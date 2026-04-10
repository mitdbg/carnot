# Movie Scenario

*Data Modalities: text, tables.*

## Overview

A movie review analysis scenario that focuses on sentiment analysis and review comparisons. The queries in this scenario analyze movie reviews to identify positive and negative sentiments, compare sentiments between different reviews, and calculate various metrics based on review sentiment. This scenario tests systems' ability to understand and classify the emotional tone of textual content while performing traditional SQL operations on the results.

## Schema

The schema has two tables:
```
Movies(id text, title text, audienceScore int, tomatoMeter int, rating text, ratingContents text, releaseDateTheaters date, releaseDateStreaming date, runtimeMinutes int, genre text, originalLanguage text, director text, writer text, boxOffice text, distributor text, soundMix text)
Reviews(id text, reviewId text, creationDate date, criticName text, isTopCritic boolean, originalScore text, reviewState text, publicationName text, reviewText text, *scoreSentiment text, reviewUrl text)
```

The column marked with an asterisk (*) is not visible to the system used for multimodal data processing. The `scoreSentiment` column contains the ground truth sentiment labels ('POSITIVE', 'NEGATIVE') and is used to generate ground truth by executing traditional SQL queries.

The `Movies` table contains information about movies including their titles, scores, ratings, release dates, runtime, and other metadata. The `Reviews` table contains movie reviews with associated metadata including the review text and sentiment. The `scoreSentiment` column contains the actual sentiment of the review (not visible to the benchmarked system).

## Queries

Those are the benchmark queries expressed in natural language:

- Five clearly positive reviews (any movie). Return reviewId.
- Five positive reviews for movie `taken_3`. Return reviewId.
- Count of positive reviews for movie `taken_3`. Return the count as positive_review_cnt.
- Positivity ratio of reviews for movie `taken_3`. Return the ratio as positivity_ratio.
- Ten Pairs of reviews that express the *same* sentiment (both are positive or both are negative) for movie with id `ant_man_and_the_wasp_quantumania`. Return id, reviewId from first review, reviewId from second review.
- Ten Pairs of reviews that express the *opposite* sentiment (one is positive and the other is negative) for movie with id `ant_man_and_the_wasp_quantumania`. Return id, reviewId from first review, reviewId from second review.
- All Pairs of reviews that express the *opposite* sentiment (one is positive and the other is negative) for movie with id `ant_man_and_the_wasp_quantumania`. Return id, reviewId from first review, reviewId from second review.
- Calculate the number of positive and negative reviews for movie `taken_3`. Return scoreSentiment and count.
- Score from 1 to 5 how much did the reviewer like the movie based on the movie reviews for movie `ant_man_and_the_wasp_quantumania`. Return reviewId, reviewScore.
- Rank the movies based on movie reviews. For each movie, score every review of it from 1 to 5, then calculate the average score of these reviews for each movie. Return movieId, movieScore.

## Data Generation Logic

The data generation process creates two tables from a Kaggle movie reviews dataset:
- **Movies table**: Movie metadata including titles, scores, ratings, and release information
- **Reviews table**: Movie reviews with sentiment analysis and critic information

### Table Size Relationships

The tables have a dependent relationship where movies are derived from reviews:
- `Reviews_{scale_factor}.csv`: scale_factor rows (primary table, up to 1,375,738 available)
- `Movies_{scale_factor}.csv`: Variable rows (only movies that have reviews)

### Scaling Factor

The `scale_factor` parameter controls the number of reviews to generate. The reviews table will have exactly scale_factor rows, while the movies table contains only the unique movies referenced in those reviews.

**Dataset Constraints:**
- Available reviews (after filtering nulls): 1,375,738
- Available movies: 143,258
- **Recommended range: 1-1375738**

**Example with scale_factor = 5000:**
- Reviews_5000.csv: 5000 rows (controlled by scale_factor)
- Movies_5000.csv: ~200-400 rows (depends on unique movies in selected reviews)
- Maximum table size: 5000 rows (Reviews table)

**Example with scale_factor = 100000:**
- Reviews_100000.csv: 100000 rows
- Movies_100000.csv: ~3000-5000 rows (more diverse movie selection)
- Maximum table size: 100000 rows (Reviews table)

### Strategic Review Selection

The generation logic ensures:
- Includes one movie with the largest number of reviews following the same score pattern (e.g., /5 or /10)
- Includes one movie with mostly negative reviews (hardcoded as 'taken_3')
- Prioritizes movies with many reviews and longer review texts
- Filters out reviews with null reviewText for data quality

## Ground Truth

The following SQL queries generate ground truth for the questions above (using the `scoreSentiment` column that is invisible to the benchmarked system). Note that `limit` is not included in these queries for evaluation because `limit` assumes no order and we should get all satisfied data from ground truth queries.

- `SELECT reviewId FROM Reviews WHERE scoreSentiment = 'POSITIVE';`
- `SELECT reviewId FROM Reviews WHERE id = 'taken_3' AND scoreSentiment = 'POSITIVE';`
- `SELECT COUNT(*) AS positive_review_cnt FROM Reviews WHERE id = 'taken_3' AND scoreSentiment = 'POSITIVE';`
- `SELECT CAST(SUM(CASE WHEN scoreSentiment = 'POSITIVE' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) AS positivity_ratio FROM Reviews WHERE id = 'taken_3';`
- `SELECT R1.id, R1.reviewId, R2.reviewId FROM Reviews AS R1 JOIN Reviews AS R2 ON R1.id = R2.id AND R1.reviewId <> R2.reviewId WHERE R1.id = 'ant_man_and_the_wasp_quantumania' AND R1.scoreSentiment = R2.scoreSentiment;`
- `SELECT R1.id, R1.reviewId, R2.reviewId FROM Reviews AS R1 JOIN Reviews AS R2 ON R1.id = R2.id AND R1.reviewId <> R2.reviewId WHERE R1.id = 'ant_man_and_the_wasp_quantumania' AND R1.scoreSentiment <> R2.scoreSentiment;`
- `SELECT R1.id, R1.reviewId, R2.reviewId FROM Reviews AS R1 JOIN Reviews AS R2 ON R1.id = R2.id AND R1.reviewId <> R2.reviewId WHERE R1.id = 'ant_man_and_the_wasp_quantumania' AND R1.scoreSentiment <> R2.scoreSentiment;`
- `SELECT scoreSentiment, COUNT(*) FROM Reviews WHERE id = 'taken_3' GROUP BY scoreSentiment;`
- `SELECT reviewId, CAST(SPLIT_PART(originalScore, '/', 1) AS FLOAT) AS reviewScore FROM Reviews WHERE id = 'ant_man_and_the_wasp_quantumania';`
- `SELECT M.id AS movieId, M.audienceScore AS movieScore FROM Movies AS M;`

## Query Summary
Here's a summary table of the features for each query:

| **Query** | **Data Types** | | **Semantic Operations** | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Table** | **Text** | **Filter** | **Join** | **Group-by** | **Score** |
|**Q1**| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|**Q2**| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|**Q3**| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|**Q4**| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|**Q5**| ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
|**Q6**| ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
|**Q7**| ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
|**Q8**| ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
|**Q9**| ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
|**Q10**| ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |


---