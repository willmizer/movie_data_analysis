# EDA Output

This file represents the final cleaned and feature-enhanced dataset, outlier filtering, and log transformation of numerical variables. It is the last stage before creating the recommendation model.

---

## File: `final_cleaned_tmdb.csv`

- **Log-transformed numeric fields** (for improved distribution and modeling)
- **Data formating and null handling**

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

