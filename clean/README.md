# Movie Dataset Clean

This folder contains **3 core CSV files** used to create a unified movie metadata dataset combining TMDb, IMDb, and Letterboxd information.

---

## File Overview

| File Name                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `tmdb_movie_data_full.csv` | Contains enriched metadata from TMDb based on IMDb IDs                    |
| `movie_themes.csv`         | Contains movie "themes", linked by title + slug.                          |
| `final_tmdb.csv`           | Final cleaned dataset merged together combining the two sources, with added profit column. |

---

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


