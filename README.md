# MovieMatch AI: Content-Based Movie Recommendation System

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://movie-match-ai.streamlit.app/)

**A content-based movie recommendation engine that suggests films based on what you already like, not what other users rated.**

Rather than relying on collaborative filtering or user reviews, MovieMatch analyzes what makes a movie tick: its cast, themes, plot, genre, and structure, then compares those attributes against thousands of other films. It combines semantic depth (natural language embeddings) with structural precision (cast, crew, runtime, themes) using optimized feature weighting and a KNN backend.

---

## Overview

- Scrapes and enriches movie metadata using IMDb's public dataset and TMDb's API.
- Generates semantic embeddings for plot/overview text using `SentenceTransformer`.
- Extracts high-signal features with TF-IDF (genres, cast, director, collection, themes) and normalized numeric fields.
- Tunes a weighted feature-fusion model with Optuna for optimal similarity.
- Serves recommendations through a fast KNN backend in an interactive Streamlit app, with posters.

## Tech Stack

- **App:** Python, Streamlit
- **Machine Learning:** scikit-learn (KNN, TF-IDF, TruncatedSVD), SentenceTransformer, Optuna
- **Data Collection:** TMDb API, IMDb public datasets, Letterboxd themes

## Data Pipeline

### 1. Data Collection (`merging_imdb/movie_join.ipynb`, `scraping/tmdb_scrape.py`)
- Downloaded IMDb's public datasets and merged the title/ratings files, preserving `tconst` IDs and filtering out movies longer than 5 hours or shorter than 45 minutes.
- Scraped TMDb for metadata per IMDb ID: title, release date, genre, budget, revenue, runtime, cast, director, keywords, spoken languages, production companies, certification, and poster URLs.
- Merged everything into a single combined dataset.

### 2. Data Cleaning (`clean/data_cleanup.ipynb`)
- Standardized variables, handled null values, and joined TMDb metadata with Letterboxd themes into a single structured dataset with columns like `genres`, `revenue`, `budget`, `runtime`, `vote_average`, `top_cast`, `director`, `keywords`, `themes`, and a derived `profit_in_millions`.

### 3. Exploratory Data Analysis (`eda/eda_clean.ipynb`)
- Filtered outliers and log-transformed skewed numeric fields (budget, revenue, runtime, vote count) for cleaner model input.
- Revenue had a much broader spread than budget: most films cluster around standard budgets, but earnings range from barely-recouped to massive hits, producing a long right tail.
- Vote count and vote average correlated only weakly (0.44), suggesting engagement doesn't strongly predict quality. Runtime showed a slight positive correlation with rating.
- Drama and Comedy dominated the genre distribution, with Thriller and Horror also common; theme data was less reliable due to missing values.

### 4. Feature Engineering (`modeling/model_prep.ipynb`)
- TF-IDF vectors for genres, cast, director, collection, and themes.
- Sentence embeddings (`all-mpnet-base-v2`) for overview and keywords.
- Scaled numeric features (log-transformed runtime, budget, revenue, rating, votes).
- Weighted fusion of all feature types, dimensionality reduction via TruncatedSVD (125 components), then cosine similarity + KNN for recommendations.

### 5. Weight Tuning (`modeling/tuning_weights.py`)
- Used Optuna to sample weights across 8 feature blocks (genres, themes, cast, director, collection, overview, keywords, numeric) and maximize precision@3 against a curated ground-truth set, over roughly 500 trials.

## Key Results

- **Precision@3** — the share of the top 3 recommendations that match a manually selected relevant film — improved from ~10% to 35%+, a ~3.5x lift, after switching from TF-IDF-only features to SentenceTransformer semantic embeddings with Optuna-tuned feature weights. Evaluated against ~40 hand-curated seed films, each paired with 5–7 expert-selected relevant titles.
- Feature weights were tuned with Optuna (500 trials, TPE sampler) to maximize precision@3 on that same evaluation set; the reported improvement reflects performance on the tuning set — results on unseen seed films may vary.
- Built a weighted hybrid recommendation engine combining semantic and structured data across 8 feature components (genres, themes, cast, director, collection, overview, keywords, numeric) via cosine similarity.
- Cleaned and unified over 70,000 movie records from IMDb, TMDb, and Letterboxd into one enriched dataset.
- Reduced high-dimensional embeddings with TruncatedSVD for an efficient, responsive KNN backend.
- Deployed as an interactive Streamlit app where users enter a movie and get personalized recommendations with metadata and posters in real time.

## Project Structure

```
movie_match/
├── movie_recommender.py          # Streamlit app (entry point)
├── scraping/
│   ├── tmdb_scrape.py             # TMDb metadata scraper
│   └── combined_imdb_movies.csv
├── merging_imdb/
│   └── movie_join.ipynb           # Merges IMDb title/ratings datasets
├── clean/
│   └── data_cleanup.ipynb         # Cleans and joins collected data
├── eda/
│   └── eda_clean.ipynb            # Exploratory analysis, log transforms
├── modeling/
│   ├── model_prep.ipynb           # Builds the TF-IDF/embedding feature matrix
│   └── tuning_weights.py          # Optuna feature-weight tuning
├── models/
│   ├── knn_model.joblib           # Trained KNN recommender
│   ├── feature_matrix_reduced.npz # SVD-reduced feature matrix
│   ├── movies_df.pkl.gz           # Movie metadata used at inference time
│   ├── scaler.joblib
│   └── tfidf_*.joblib             # Per-feature TF-IDF vectorizers
├── images/                        # EDA charts
└── requirements.txt
```

## Run Locally

```bash
git clone https://github.com/willmizer/movie_data_analysis.git
cd movie_data_analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run movie_recommender.py
```

## Limitations

- Precision@3 was measured on the same ~40 seed films used to tune the feature weights; performance on unseen movie types may be lower.
- Purely content-based — no collaborative filtering or user preference signal; two structurally similar films can feel tonally very different to a specific viewer.
- Films with sparse or low-quality metadata (niche releases, foreign titles with limited TMDb coverage, older catalogue titles) will produce weaker recommendations.
- Letterboxd theme data had significant missing values and was treated as a lower-weight feature as a result.
- No mechanism to avoid "franchise bubbles" — a film may recommend primarily other entries in the same series even when thematic similarity across franchises would better serve the user.

## Future Improvements

- Add a feedback mechanism on the recommender so users can rate suggestions.
- Track liked-movie history and factor it into future recommendations.
- Dig further into EDA to uncover relationships not yet reflected in the feature weights.
- Keep tuning the model as more ground-truth examples become available.
- Improve the mobile UI.

## License

This project is shared for portfolio and educational purposes. Feel free to explore the code, but please reach out before reusing it commercially.

© 2026 Will Mizer
