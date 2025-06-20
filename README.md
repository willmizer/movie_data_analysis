# **MovieMatch AI: Content-Based Movie Recommendation System** 
## [Try It Out!](https://movie-match-ai.streamlit.app/)

This project delivers an end-to-end pipeline for collecting, enriching, cleaning, tuning, and deploying movie metadata from TMDb, IMDb, and Letterboxd to build a personalized recommendation engine. Designed to provide intelligent movie suggestions based on content similarity, the system uses sentence embeddings, TF-IDF features, and structured metadata combined with optimized feature weighting and a KNN backend to recommend films similar to those users already love.

A key goal of this system is to combine *semantic depth* (via natural language embeddings) with *structural precision* (cast, crew, runtime, themes, etc.), all tuned for maximum real-world recommendation accuracy.

---

## Code Files Included (Ctrl+Click to view respective file)

| File                              | Description                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`movie_join.ipynb`](https://github.com/willmizer/movie_data_analysis/blob/main/merging_imdb/movie_join.ipynb)           | Uses TMDb API to enrich movies with metadata, crew, keywords, and providers from IMDb IDs.         |
| [`tmdb_scrape.py`](https://github.com/willmizer/movie_data_analysis/blob/main/scraping/tmdb_scrape.py)           | Uses TMDb API to enrich movies with metadata, crew, keywords, and providers from IMDb IDs.         |
| [`data_cleanup.ipynb`](https://github.com/willmizer/movie_data_analysis/blob/main/clean/data_cleanup.ipynb)  | Cleans and merges all collected data for modeling (handles duplicates, NA values, joins, etc).      |
| [`eda_clean.ipynb`](https://github.com/willmizer/movie_data_analysis/blob/main/eda/eda_clean.ipynb)        | Explores dataset, filters out adult content, and prepares numeric/categorical columns.              |
| [`model_prep.ipynb`](https://github.com/willmizer/movie_data_analysis/blob/main/modeling/model_prep.ipynb)      | Constructs full feature matrix using TF-IDF, SentenceTransformer embeddings, and normalized scaling. |
| [`tuning_weights.py`](https://github.com/willmizer/movie_data_analysis/blob/main/modeling/tuning_weights.py)     | Tunes feature weights using Optuna to improve recommendation precision@3.                           |
| [`movie_recommender.py`](https://github.com/willmizer/movie_data_analysis/blob/main/movie_recommender.py) | Deployable Streamlit app with real-time recommendations and posters.                                |

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Collection](#data-collection)  
3. [Feature Engineering](#feature-engineering)  
4. [Weight Tuning & Optimization](#weight-tuning--optimization)  
5. [Modeling Pipeline](#modeling-pipeline)  
6. [Streamlit Interface](#streamlit-interface)  
7. [Key Results](#key-results)  
8. [Future Improvements](#future-improvements)

---

## Project Overview

MovieMatch AI recommends movies based on *what you liked*. Rather than relying on collaborative filtering or user reviews, it analyzes what makes a movie tick—its cast, themes, plot, genre, and structure—and compares those attributes with thousands of others.

Core features include:

- Scraping and enriching movie metadata using TMDb and Letterboxd
- Generating semantic embeddings using `SentenceTransformer`
- Extracting high-signal features using TF-IDF and numerical normalization
- Tuning a weighted feature fusion model for optimal similarity
- Deploying a fast KNN-based recommendation engine
- Serving results through an interactive Streamlit app

---

## Data Collection

### `tmdb_scrape.py`  
Scrapes metadata for each IMDb movie by using TMDb's API. Includes:
- Title, release date, genre, budget, revenue, runtime, cast, director
- Keywords, spoken languages, production company, certification
- Poster URLs for front-end display

### `themes_scrape.py`  
Uses BeautifulSoup to extract *themes* from Letterboxd genre pages using custom slug rules and fallbacks.

All collected data is merged into a single dataset, `final_cleaned_tmdb.csv`.

---

## Feature Engineering

In `model_prep.ipynb`, we build the full feature matrix:
- **TF-IDF** vectors for genres, cast, themes, director, and collection
- **Sentence embeddings** (`all-mpnet-base-v2`) for plot overview and keywords
- **Scaled numerical features**: log-runtime, log-budget, log-revenue, average rating, and vote count
- **Weighted fusion** of all blocks using optimized parameters
- **Dimensionality reduction** using TruncatedSVD (125 components)
- **Normalization** and fitting a cosine-based KNN model

---

## Weight Tuning & Optimization

In `tuning_weights.py`, we use Optuna to:
- Randomly sample feature weights across all 8 feature blocks
- Measure precision@3 based on a curated ground truth dictionary of related movies
- Maximize the number of relevant recommendations per trial
- Achieved >26.4% precision@3 across ~500 trials

Example best weights:
```python
{'genres': 1.15, 'themes': 0.73, 'cast': 1.85, 'director': 2.63, 
 'collection': 3.91, 'overview': 1.99, 'keywords': 3.84, 'numeric': 0.83}
```

---

## Future Improvements
- Add feedback section on movie recommender
- Explore more in EDA to try and uncover other important realationships that need to be implemented into modeling
- Improve models with more tuning
- Improve mobile ui design for movie match
