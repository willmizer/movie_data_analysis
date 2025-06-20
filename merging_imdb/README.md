# IMDb Dataset Merge: Basics + Ratings

We take two key IMDb data sources:
- `title.basics.tsv` — Core movie metadata
- `title.ratings.tsv` — Rating and vote counts


---

## Files Used

| File                | Description                                                                   |
|---------------------|--------------------------------------------------------------------------------|
| `title.basics.tsv`  | Contains metadata on every IMDb title (e.g. name, year, runtime, genres).      |
| `title.ratings.tsv` | Provides average user rating and number of votes per title.                    |

---

## Sample Input

### `title.basics.tsv`

| tconst      | titleType | primaryTitle                       | startYear | runtimeMinutes | genres             |
|-------------|------------|------------------------------------|------------|------------------|---------------------|
| tt0000001   | short      | Carmencita                         | 1894       | 1                | Documentary,Short   |
| tt0000003   | short      | Poor Pierrot                       | 1892       | 5                | Animation,Comedy    |

### `title.ratings.tsv`

| tconst      | averageRating | numVotes |
|-------------|----------------|----------|
| tt0000001   | 5.7            | 2163     |
| tt0000003   | 6.5            | 2213     |

---

## Final Output: `combined_imdb_movies.csv`

| tconst      | primaryTitle              | startYear | runtimeMinutes | genres                     | averageRating | numVotes |
|-------------|---------------------------|-----------|----------------|-----------------------------|----------------|-----------|
| tt0012349   | The Kid                   | 1921.0    | 68.0           | Comedy,Drama,Family         | 8.2            | 141,521   |
| tt0010323   | The Cabinet of Dr. Caligari| 1920.0    | 67.0           | Horror,Mystery,Thriller     | 8.0            | 73,629    |
| tt0009968   | Broken Blossoms           | 1919.0    | 90.0           | Drama,Romance               | 7.2            | 11,487    |

> This file serves as the backbone for pulling api informaion from tmdb and the remaining of the project

---

## Cleaned Fields Included

- `tconst` — IMDb title ID
- `primaryTitle` — Movie title
- `startYear` — Release year
- `runtimeMinutes` — Duration in minutes
- `genres` — Comma-separated list of genres
- `averageRating` — IMDb average user score (0–10)
- `numVotes` — Number of user votes

> These fields won't mean much to us after we pull from tmdb other than the tconst field but it was useful to pull all of the information for some early filtering
---
