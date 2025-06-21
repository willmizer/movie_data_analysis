**`unique_city_ids` Ex output**: 

| city_name     | city_id | url                                                       |
| ------------- | ------- | ---------------------------------------------------------- |
| Abbs Valley   | 37317   | https://www.redfin.com/city/37317/VA/Abbs-Valley          |
| Abingdon      | 22      | https://www.redfin.com/city/22/VA/Abingdon                |
| Accomac       | 28      | https://www.redfin.com/city/28/VA/Accomac                 |
| Aden          | 29022   | https://www.redfin.com/city/29022/VA/Aden                 |
| Adwolf        | 21132   | https://www.redfin.com/city/21132/VA/Adwolf               |


side by side 
<table>
  <tr>
    <td align="center">
      <img src="images/correlation_matrix1.png" width="400" alt="Correlation Matrix Before Cleaning"/>
      <br><em>Before cleaning</em>
    </td>
    <td align="center">
      <img src="images/correlation_matrix2.png" width="400" alt="Correlation Matrix After Cleaning"/>
      <br><em>After cleaning</em>
    </td>
  </tr>
</table>

big main
<div align="center">
  <img src="images/price-sqft.png" width="1000" alt="price-sqft distribution"/>
</div>
--------------------------------------------------------------------------------------------------------------------------------------------------------------
<div align="center">
  <img src="images/price-sqft-chart.png" width="1000" alt="price-sqft correlation"/>
</div>
--------------------------------------------------------------------------------------------------------------------------------------------------------------
<div align="center">
  <img src="images/price-sqft-property_type.png" width="1000" alt="price-sqft by property type"/>
</div>
--------------------------------------------------------------------------------------------------------------------------------------------------------------
<div align="center">
  <img src="images/hoa-price-chart.png" width="1000" alt="HOA vs price"/>
</div>
--------------------------------------------------------------------------------------------------------------------------------------------------------------

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
3. [Data Clean](#data-clean)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)  
6. [Weight Tuning & Optimization](#weight-tuning--optimization)  
7. [Modeling Pipeline](#modeling-pipeline)  
8. [Streamlit Interface](#streamlit-interface)  
9. [Key Results](#key-results)  
10. [Future Improvements](#future-improvements)

---

## Project Overview

MovieMatch AI recommends movies based on *what you liked*. Rather than relying on collaborative filtering or user reviews, it analyzes what makes a movie tick—its cast, themes, plot, genre, and structure—and compares those attributes with thousands of others.

Core features include:

- Scraping and enriching movie metadata using IMDb publicly available dataset and TMDbs public API pulls
- Generating semantic embeddings using `SentenceTransformer`
- Extracting high-signal features using TF-IDF and numerical normalization
- Tuning a weighted feature fusion model for optimal similarity
- Deploying a fast KNN-based recommendation engine
- Serving results through an interactive Streamlit app
  
There were a lot of adjustments along the way:
- Different `SentenceTransformer' models were used.
- Along with different tuning methods
- I tried a content column as well that contained all features per row combined. Then I used `SentenceTransformer` on it to predict movies from just one column
- I had also tried using different data sources and different features
- Ultimately I decided with the proposed methods and what seemed to work the best
  
---

## Data Collection

### `movie_join.ipynb`
Downloaded IMDbs publicly available datasets and merged nessisary title and ratings files
- Made sure IMDbs tconst ids were collected
- made very broad cleaning descisons such as cutting off movies longer than 5 hours and shorter than 45 min to remove bulk

### `tmdb_scrape.py`  
Scrapes metadata for each IMDb movie using the tconst by using TMDb's API. Includes:
- Title, release date, genre, budget, revenue, runtime, cast, director
- Keywords, spoken languages, production company, certification
- Poster URLs for front-end display

All collected data is merged into a single dataset, `combined_imdb_movies.csv`.

---

## Data Clean

The main files involved with cleaning are:

| File Name                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `tmdb_movie_data_full.csv` | Contains enriched metadata from TMDb based on IMDb IDs                    |
| `movie_themes.csv`         | Contains movie "themes", linked by title + slug.                          |
| `final_tmdb.csv`           | Final cleaned dataset merged together combining the two sources, with added profit column. |

---

The main purpose of the clean was to standardize/remove and null values and ensure variables needed were available and fully ready for modeling

## Final Output: `final_tmdb.csv`

This is the result of merging all sources into a single, structured, and cleaned movie dataset.

### Sample Columns

| Column Name            | Description                                               |
|------------------------|-----------------------------------------------------------|
| `imdb_id`              | IMDb unique movie identifier (e.g., `tt0011237`)          |
| `tmdb_id`              | TMDb movie ID                                             |
| `title`                | Full movie title                                          |
| `release_date`         | Date of original release                                  |
| `genres`               | Comma-separated list of genres                            |
| `revenue`, `budget`    | Financials (USD)                                          |
| `runtime`              | Movie runtime in minutes                                  |
| `vote_average`, `vote_count` | Rating score and number of votes                   |
| `top_cast`, `director` | Main cast members and director(s)                         |
| `keywords`, `themes`   | TMDb tags and scraped Letterboxd themes                   |
| `spoken_languages`     | Languages spoken in the film                              |
| `collection_name`      | If part of a franchise or collection                      |
| `watch_providers`      | Platforms where the movie is available to stream          |
| `poster_url`           | Direct link to poster image                               |
| `overview`             | Plot summary                                              |
| `profit_in_millions`   | Custom-calculated profit in millions (`revenue - budget`) |

---

## Exploratory Data Analysis

This section represents the final cleaned and feature-enhanced dataset, outlier filtering, and log transformation of numerical variables. It is the last stage before creating the recommendation model.

---

## File: `final_cleaned_tmdb.csv`

- **Log-transformed numeric fields** (for improved distribution and modeling)
- **Data formating and null handling**

<table>
  <tr>
    <td align="center">
      <img src="images/correlation_matrix1.png" width="400" alt="Correlation Matrix Before Cleaning"/>
      <br><em>Before cleaning</em>
    </td>
    <td align="center">
      <img src="images/correlation_matrix2.png" width="400" alt="Correlation Matrix After Cleaning"/>
      <br><em>After cleaning</em>
    </td>
  </tr>
</table>

---

## Key Columns

| Column                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `imdb_id`, `tmdb_id`  | Unique identifiers for linking across IMDb and TMDb                         |
| `title`               | Movie title                                                                 |
| `release_date`        | Official release date                                                       |
| `genres`              | Comma-separated genres                                                      |
| `revenue`, `budget`   | Raw financial values (USD)                                                  |
| `runtime`             | Runtime in minutes                                                          |
| `vote_average`        | TMDb average user rating                                                    |
| `vote_count`          | TMDb vote count                                                             |
| `top_cast`, `director`| Primary cast and directors                                                  |
| `keywords`            | TMDb tags or keywords                                                       |
| `themes`              | Letterboxd themes (pipe-separated)                                          |
| `spoken_languages`    | Languages spoken in the film                                                |
| `collection_name`     | Franchise or film series (if applicable)                                    |
| `watch_providers`     | Streaming services available                                                |
| `production_companies`| Main production studios                                                     |
| `certification`       | Content rating (G, PG, R, etc.)                                             |
| `overview`            | Plot synopsis                                                               |
| `poster_url`          | Link to movie poster image                                                  |
| `error`               | Any scrape/load errors (if present)                                         |
| `profit_in_millions`  | Calculated as `(revenue - budget) / 1e6`                                    |

---

## Feature Engineering

In `model_prep.ipynb`, I build the full feature matrix:
- **TF-IDF** vectors for genres, cast, themes, director, and collection
- **Sentence embeddings** (`all-mpnet-base-v2`) for plot overview and keywords (nuance values)
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
