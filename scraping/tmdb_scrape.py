import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import time
import threading
import gc
from requests.adapters import HTTPAdapter, Retry

API_KEY = '124f5ace47354f3dacc11b0b3c024c7a'
BASE_FIND_URL = "https://api.themoviedb.org/3/find/"
BASE_MOVIE_URL = "https://api.themoviedb.org/3/movie/"
BASE_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

# rate limiter using token bucket algorithm
class TokenBucket:
    def __init__(self, rate, per):
        self._capacity = rate
        self._tokens = rate
        self._lock = threading.Lock()
        self._fill_interval = per / rate
        threading.Thread(target=self._refill, daemon=True).start()

    def _refill(self):
        while True:
            time.sleep(self._fill_interval)
            with self._lock:
                if self._tokens < self._capacity:
                    self._tokens += 1

    def consume(self):
        while True:
            with self._lock:
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
            time.sleep(self._fill_interval)

# initialize limiter for 50 requests per second
bucket = TokenBucket(rate=50, per=1)

def limited_get(url, **kwargs):
    bucket.consume()
    return session.get(url, **kwargs)
# end rate limiter

# HTTP session with retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)
session.timeout = (5, 10)


# read movie metadata
df = pd.read_csv("combined_imdb_movies.csv")
movie_metadata = df.set_index("tconst")[['primaryTitle','startYear']].to_dict('index')
all_imdb_ids = df['tconst'].dropna().unique()

# API call counter
api_call_count = 0
api_call_lock = threading.Lock()
def increment_api_call_count():
    global api_call_count
    with api_call_lock:
        api_call_count += 1

# fetch details in one call using append_to_response
def _fetch_movie_details(imdb_id, tmdb_id):
    try:
        url = (
            f"{BASE_MOVIE_URL}{tmdb_id}" 
            f"?api_key={API_KEY}&language=en-US"
            "&append_to_response=credits,keywords,release_dates,watch/providers"
        )
        increment_api_call_count()
        resp = limited_get(url, timeout=session.timeout)
        resp.raise_for_status()
        data = resp.json()
        # parse core movie data
        movie = data
        # parse credits
        cast = data.get('credits', {}).get('cast', [])[:5]
        top_cast = ", ".join(c['name'] for c in cast)
        crew = data.get('credits', {}).get('crew', [])
        director = ", ".join(c['name'] for c in crew if c.get('job') == 'Director')
        # parse keywords
        keywords = ", ".join(k['name'] for k in data.get('keywords', {}).get('keywords', []))
        # parse release_dates for certification
        certification = ''
        for entry in data.get('release_dates', {}).get('results', []):
            if entry.get('iso_3166_1') == 'US':
                for rel in entry.get('release_dates', []):
                    if rel.get('certification'):
                        certification = rel['certification']
                        break
        # parse watch providers
        providers = data.get('watch/providers', {}).get('results', {}).get('US', {}).get('flatrate', [])
        watch_providers = ", ".join(p['provider_name'] for p in providers)
        # spoken languages & production companies
        spoken_languages = ", ".join(l['name'] for l in movie.get('spoken_languages', []))
        production_companies = ", ".join(pc['name'] for pc in movie.get('production_companies', []))
        collection_name = movie.get('belongs_to_collection', {}).get('name','') if movie.get('belongs_to_collection') else ''

        return {
            'imdb_id': imdb_id,
            'tmdb_id': tmdb_id,
            'title': movie.get('title'),
            'release_date': movie.get('release_date'),
            'genres': ", ".join(g['name'] for g in movie.get('genres', [])),
            'revenue': movie.get('revenue'),
            'budget': movie.get('budget'),
            'runtime': movie.get('runtime'),
            'vote_average': movie.get('vote_average'),
            'vote_count': movie.get('vote_count'),
            'top_cast': top_cast,
            'director': director,
            'keywords': keywords,
            'spoken_languages': spoken_languages,
            'collection_name': collection_name,
            'watch_providers': watch_providers,
            'production_companies': production_companies,
            'certification': certification,
            'overview': movie.get('overview'),
            'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
            'error': None
        }
    except Exception as e:
        return {'imdb_id': imdb_id, 'error': str(e)}

# fetch tmdb id for a given imdb id
def fetch_tmdb_data(imdb_id):
    meta = movie_metadata.get(imdb_id, {})
    # find by imdb id
    try:
        find_url = f"{BASE_FIND_URL}{imdb_id}?external_source=imdb_id&api_key={API_KEY}"
        increment_api_call_count()
        resp = limited_get(find_url, timeout=session.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get('movie_results'):
            tmdb_id = data['movie_results'][0]['id']
            return _fetch_movie_details(imdb_id, tmdb_id)
        # fallback search
        title, year = meta.get('primaryTitle'), meta.get('startYear')
        if title and year:
            q = urllib.parse.quote_plus(title)
            search_url = f"{BASE_SEARCH_URL}?api_key={API_KEY}&query={q}&year={year}"
            increment_api_call_count()
            resp = limited_get(search_url, timeout=session.timeout)
            resp.raise_for_status()
            results = resp.json().get('results', [])
            if results:
                tmdb_id = results[0]['id']
                return _fetch_movie_details(imdb_id, tmdb_id)
        return {'imdb_id': imdb_id, 'error': 'No TMDb match found'}
    except Exception as e:
        return {'imdb_id': imdb_id, 'error': str(e)}

if __name__ == '__main__':
    start = time.time()
    # prepare output
    open('tmdb_movie_data_full.csv','w').close()
    open('tmdb_fetch_errors_full.csv','w').close()

    # disable GC for speed
    gc.disable()
    batch_size = 1000
    with ThreadPoolExecutor(max_workers=10) as pool:
        for i in tqdm(range(0, len(all_imdb_ids), batch_size), desc='Batches'):
            batch = all_imdb_ids[i:i+batch_size]
            futures = [pool.submit(fetch_tmdb_data, mid) for mid in batch]
            results = []
            errors = []
            for f in as_completed(futures):
                r = f.result()
                (errors if r.get('error') else results).append(r)
            if results:
                pd.DataFrame(results).to_csv('tmdb_movie_data_full.csv', mode='a', header=(i==0), index=False)
            if errors:
                pd.DataFrame(errors).to_csv('tmdb_fetch_errors_full.csv', mode='a', header=(i==0), index=False)
    gc.enable()
    duration = time.time() - start
    print(f"Done in {duration:.2f}s, total calls: {api_call_count}, avg rps: {api_call_count/duration:.1f}")
