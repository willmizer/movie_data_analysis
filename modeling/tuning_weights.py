import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import optuna

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer




# load and process dataset
df = pd.read_csv('final_tmdb.csv', keep_default_na=False)
df.drop_duplicates(subset=['title'], inplace=True)
df.reset_index(drop=True, inplace=True)

# cache embeddings for faster re-runs
EMB_CACHE = "embeddings.npz"
if os.path.exists(EMB_CACHE):
    data = np.load(EMB_CACHE)
    overview_embeds = data["overview"]
    keywords_embeds = data["keywords"]
    print(f"Loaded embeddings from {EMB_CACHE}")
else:
    model = SentenceTransformer('all-mpnet-base-v2') # all-mpnet-base-v2 is used for its good balance of speed and accuracy over all-MiniLM-L6-v2 model
    overview_embeds = model.encode(df['overview'].tolist(), show_progress_bar=True)
    keywords_embeds = model.encode(df['keywords'].tolist(), show_progress_bar=True)
    np.savez(EMB_CACHE, overview=overview_embeds, keywords=keywords_embeds)
    print(f"Computed and cached embeddings to {EMB_CACHE}")

# compute tf-idf vectors for categorical features
vect = TfidfVectorizer(token_pattern='[^,]+')
genre_tfidf      = vect.fit_transform(df['genres'])
themes_tfidf     = vect.fit_transform(df['themes'])
cast_tfidf       = vect.fit_transform(df['top_cast'])
director_tfidf   = vect.fit_transform(df['director'])
collection_tfidf = vect.fit_transform(df['collection_name'])

numeric_cols = ['runtime_log','budget_log','revenue_log','vote_average','vote_count_log']
num_scaled   = StandardScaler().fit_transform(df[numeric_cols])

# recommendations and evaluations
def build_feature_matrix(weights):
    blocks = [
        genre_tfidf    * weights['genres'],
        themes_tfidf   * weights['themes'],
        cast_tfidf     * weights['cast'],
        director_tfidf * weights['director'],
        collection_tfidf * weights['collection'],
        overview_embeds * weights['overview'],
        keywords_embeds * weights['keywords'],
        num_scaled      * weights['numeric'],
    ]
    X = hstack(blocks, format='csr')
    return normalize(X, norm='l2', axis=1)

def recommend(seed, X_norm, k=3):
    idx = df.index[df['title'] == seed][0]
    similarities = cosine_similarity(X_norm[idx], X_norm).flatten()
    top_indices = similarities.argsort()[::-1][1:k+1]  # skip self
    return df.iloc[top_indices]['title'].tolist()

def precision_at_k(recs, relevant, k=3):
    return len(set(recs[:k]) & set(relevant)) / k

def score_seed(seed, relevant, X_w, k=3):
    recs = recommend(seed, X_w, k=k)
    return precision_at_k(recs, relevant, k=k)

def evaluate_weights(weights, ground_truth, k=3):
    X_w = build_feature_matrix(weights)
    scores = Parallel(n_jobs=-1)(
        delayed(score_seed)(seed, relevant, X_w, k)
        for seed, relevant in ground_truth.items()
    )
    return sum(scores) / len(scores)

# list of ground truths to test different weights
ground_truth = {
    "Donnie Darko": [
        "Pi", "Primer", "Coherence", "A Scanner Darkly", "The Box",
        "The Butterfly Effect", "Take Shelter"
    ],
    "Inception": [
        "Memento", "Paprika", "Shutter Island", "The Prestige", "Source Code",
        "Tenet", "Edge of Tomorrow"
    ],
    "The Matrix": [
        "Dark City", "Ghost in the Shell", "Equilibrium", "They Live", "eXistenZ",
        "The Thirteenth Floor", "Upgrade"
    ],
    "Primer": [
        "Timecrimes", "Predestination", "Triangle", "The Infinite Man", "ARQ",
        "The Man from Earth"
    ],
    "Coherence": [
        "The Invitation", "The One I Love", "The Endless", "I'm Thinking of Ending Things", "Synchronic",
        "The Vast of Night"
    ],
    "Eternal Sunshine of the Spotless Mind": [
        "Her", "Synecdoche, New York", "The Science of Sleep", "Anomalisa", "Lost in Translation",
        "Never Let Me Go", "I'm Thinking of Ending Things"
    ],
    "Under the Skin": [
        "Enemy", "The Man Who Fell to Earth", "Upstream Color", "Antiviral", "The Lobster",
        "Beyond the Black Rainbow"
    ],
    "Black Swan": [
        "Perfect Blue", "Requiem for a Dream", "Whiplash", "The Machinist", "Enemy",
        "The Night House"
    ],
    "Arrival": [
        "Interstellar", "Contact", "The Fountain", "Children of Men", "Midnight Special",
        "The Midnight Sky"
    ],
    "Annihilation": [
        "Stalker", "Solaris", "The Signal", "The Vast of Night", "Beyond the Black Rainbow",
        "Coherence"
    ],
    "The Lobster": [
        "Dogtooth", "The Killing of a Sacred Deer", "Synecdoche, New York", "Anomalisa", "Being John Malkovich",
        "The Square"
    ],
    "Enemy": [
        "Prisoners", "The Double", "Perfect Blue", "The Machinist", "Lost Highway",
        "Dead Ringers"
    ],
    "A Ghost Story": [
        "Columbus", "The Tree of Life", "After Yang", "Wendy and Lucy", "First Reformed",
        "All the Real Girls"
    ],
    "Her": [
        "Eternal Sunshine of the Spotless Mind", "Lost in Translation", "Never Let Me Go", "Anomalisa", "Melancholia",
        "The Double Life of Véronique"
    ],
    "Ex Machina": [
        "Annihilation", "Blade Runner 2049", "The Machine", "Tau", "Transcendence",
        "I Am Mother"
    ],
    "The Fountain": [
        "The Tree of Life", "Cloud Atlas", "2001: A Space Odyssey", "Another Earth", "Solaris",
        "Melancholia"
    ],
    "The Man from Earth": [
        "Coherence", "Primer", "The Sunset Limited", "Exam", "The Arrival",
        "The Infinite Man"
    ],
    "The Killing of a Sacred Deer": [
        "The Lobster", "The House That Jack Built", "Martha Marcy May Marlene", "Dogtooth"
    ],
    "The Platform": [
        "Snowpiercer", "Cube", "Circle", "The Belko Experiment", "Exam",
        "The Divide"
    ],
    "Perfect Blue": [
        "Black Swan", "Paprika", "Tokyo Godfathers", "Millennium Actress", "Requiem for a Dream",
        "The Double"
    ],
    "The Truman Show": [
        "Eternal Sunshine of the Spotless Mind", "Her", "Pleasantville", "The Secret Life of Walter Mitty", "Big Fish",
        "Synecdoche, New York"
    ],
    "Black Swan": [
        "Requiem for a Dream", "Whiplash", "Perfect Blue", "Gone Girl", "The Machinist"
    ],
    "Joker": [
        "Taxi Driver", "You Were Never Really Here", "American Psycho", "Nightcrawler", "Fight Club",
        "The King of Comedy"
    ],
    "The Grand Budapest Hotel": [
        "The Royal Tenenbaums", "Moonrise Kingdom", "Amélie", "The Darjeeling Limited", "Fantastic Mr. Fox",
        "The French Dispatch"
    ],
    "Ex Machina": [
        "Annihilation", "Blade Runner 2049", "Her", "Upgrade", "I Am Mother",
        "Automata"
    ],
    "Arrival": [
        "Interstellar", "Contact", "The Midnight Sky", "Ad Astra", "The Man from Earth",
        "The Signal"
    ],
    "Shutter Island": [
        "The Sixth Sense", "Prisoners", "The Others", "Mystic River", "Gone Baby Gone",
        "The Number 23"
    ],
    "The Social Network": [
        "Steve Jobs", "Moneyball", "The Big Short", "Spotlight", "The Founder",
        "Tetris"
    ],
    "Whiplash": [
        "Black Swan", "Amadeus", "The Pianist", "Sound of Metal", "Shine",
        "Inside Llewyn Davis"
    ],
    "Amélie": [
        "The Science of Sleep", "The Royal Tenenbaums", "Big Fish", "Lost in Translation", "Frances Ha"
    ],
    "There Will Be Blood": [
        "No Country for Old Men", "The Master", "The Power of the Dog", "Citizen Kane", "Giant",
        "Days of Heaven"
    ],
    "Blade Runner 2049": [
        "Blade Runner", "Ex Machina", "Children of Men", "Ghost in the Shell", "Under the Skin",
        "Dune"
    ],
    "Lady Bird": [
        "Eighth Grade", "Frances Ha", "Little Women", "Booksmart", "The Edge of Seventeen",
        "The Spectacular Now"
    ],
    "Inside Llewyn Davis": [
        "A Serious Man", "Her", "Blue Valentine", "Lost in Translation", "The Rider",
        "Paterson"
    ],
    "Requiem for a Dream": [
        "Trainspotting", "Enter the Void", "Pi", "Spun", "Candy",
    ],
    "Waking Life": [
        "Slacker", "Before Sunrise", "A Scanner Darkly", "Mind Game", "The Congress",
        
    ],
    "Fantastic Mr. Fox": [
        "Isle of Dogs", "The Grand Budapest Hotel", "Coraline", "Kubo and the Two Strings", "The Little Prince",
        "ParaNorman"
    ],
        "The Dark Knight": [
        "Batman Begins",
        "Joker",
        "Logan",
        "V for Vendetta",
        "Watchmen",
        "Heat",
        "Se7en"
    ],
    "Pulp Fiction": [
        "Reservoir Dogs",
        "Snatch",
        "Lock, Stock and Two Smoking Barrels",
        "Trainspotting",
        "Natural Born Killers",
        "True Romance",
        "Go"
    ],
    "Get Out": [
        "Us",
        "The Invitation",
        "Rosemary's Baby",
        "Midsommar",
        "Hereditary",
        "The Stepford Wives",
        "The Babadook"
    ],
    "The Silence of the Lambs": [
        "Se7en",
        "Zodiac",
        "Prisoners",
        "Red Dragon",
        "Hannibal",
        "Gone Girl",
        "Manhunter"
    ],
    "Fight Club": [
        "American Psycho",
        "Se7en",
        "Taxi Driver",
        "The Machinist",
        "Memento",
        "The Game",
        "Oldboy"
    ],

}



# checking ground truth titles against the dataset
missing = {}

# fast lookup of all titles in the dataset
title_set = set(df['title'].tolist())

for seed, relevant in ground_truth.items():
    # find any titles not in your df
    not_found = [r for r in relevant if r not in title_set]
    if not_found:
        missing[seed] = not_found

# report missing titles
if missing:
    for seed, not_found in missing.items():
        print(f"Seed '{seed}' missing labels: {not_found}")
else:
    print("All ground-truth titles were found in the dataset")



def objective(trial):
    w = {
        'genres':     trial.suggest_float('genres',     0.0, 8.0),
        'themes':     trial.suggest_float('themes',     0.0, 8.0),
        'cast':       trial.suggest_float('cast',       0.0, 8.0),
        'director':   trial.suggest_float('director',   0.0, 8.0),
        'collection': trial.suggest_float('collection', 0.0, 8.0),
        'overview':   trial.suggest_float('overview',   0.0, 8.0),
        'keywords':   trial.suggest_float('keywords',   0.0, 8.0),
        'numeric':    trial.suggest_float('numeric',    0.0, 8.0),
    }
    return evaluate_weights(w, ground_truth)



if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    n_trials = 500
    trial_precisions = []

    for i in range(n_trials):
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)
        trial_precisions.append(value)

        # print progress every 1 trials
        if (i + 1) % 1 == 0 or (i + 1) == n_trials:
            print(f"Trial {i+1}/{n_trials} – precision@3: {value:.3f} – best so far: {study.best_value:.3f}")

    # printing results
    print("\nAll trial precisions:")
    print(trial_precisions)
    print(f"\nBest precision@3: {study.best_value:.3f}")
    print("Best weights:", study.best_params)
