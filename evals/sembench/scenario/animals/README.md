# Animals Scenario

*Data Modalities: images, audio, tables.*

## Overview

To study animals, biologists often rely on automated means to detect the presence of animals. In particular, camera traps and audio recorders, installed in different habitats, are popular mechanisms for detection. This scenario studies questions related to the occurrence or co-occurrence of different animal species at specific locations, using language models to detect animal species in images or audio recordings.

## Schema

The schema has two tables:
```
ImageTable(image Image, *species Text, city Text, stationID Text)
AudioTable(*animal Text, audio Audio, city Text, StationID Text)
```

Columns that are marked with an asterisk (*) are not visible to the system used for multimodal data processing. They provide the animal species directly and can be used for easily generating ground truth via traditional SQL queries.

The `ImageTable` contains the paths of images (`image` column) as well as the name of the city (`city` column) and the ID of the station (`stationID` column) at which the camera is placed. The `species` column contains a comma-separated list of species present in the associated picture. This column is not visible to the benchmarked system.

The `AudioTable` contains the path of sound recordings (`audio` column) of animal calls, along with the name of the city (`city` column) and the station ID (`stationID` column) where the recorder is located. The `animal` column contains the name of (one single) species that the call is associated with. The latter column is not visible to the benchmarked system.

## Queries

- How many pictures of zebras do we have in our database?
- How many sound recordings of elephants do we have in our database?
- What is the city where we captured most pictures of zebras (ties are broken arbitrarily, consider only cities with at least one zebra picture)?
- What is the city for which we have most recordings of elephants (ties are broken arbitrarily, consider only cities with at least one elephant recording)?
- What is the list of cities for which we have either images or audio recordings of elephants?
- For which cities do we have images of monkeys but no audio recordings of monkeys?
- What are the cities for which zebras and impala co-occur, i.e., we have at least one image of a zebra and at least one image of an impala?
- What is the list of cities for which we found at least one image or audio recording of elephants and at least one image or audio recording of monkeys?
- What is the list of cities for which we have both images and audio recordings of monkeys?
- What is the city and station with most associated pictures showing zebras (ties are broken arbitrarily, only consider cities with at least one zebra picture)?

## Data Generation Logic

The data generation process creates two tables from Kaggle datasets:
- **Audio data**: Animal sound recordings from various species (elephant, monkey, etc.)
- **Image data**: Camera trap images with animal species annotations

### Table Size Relationships

The tables have different scaling relationships based on dataset availability:
- `image_data.csv`: scaling_factor rows (up to 8,718 available)
- `audio_data.csv`: min(scaling_factor/3, 650) rows (limited by available audio files)

### Scaling Factor

The `scaling_factor` parameter controls the number of image rows. Audio rows are calculated as 1/3 of scaling_factor but capped at 650 due to dataset limitations.

**Dataset Constraints:**
- Available image files: 8,718
- Available audio files: 650
- **Recommended range: 1-8718**

**Example with scaling_factor = 1500:**
- image_data.csv: 1500 rows
- audio_data.csv: 500 rows (1500/3)
- Maximum table size: 1500 rows

**Example with scaling_factor = 3000:**
- image_data.csv: 3000 rows  
- audio_data.csv: 650 rows (capped at maximum available)
- Maximum table size: 3000 rows

### Strategic Data Distribution

The generation logic ensures specific patterns for query validation:
- Strategic assignment of animals to cities based on species and data type
- Ensures specific co-occurrence patterns (e.g., cities with monkey images but no monkey audio)
- Maintains relationships needed for complex join queries

## Ground Truth

- `select count(*) from ImageData where Species LIKE '%ZEBRA%';`
- `select count(*) from AudioData where Animal = 'Elephant';`
- `WITH zebra_counts AS (SELECT city, count(*) as cnt FROM ImageData WHERE Species LIKE '%ZEBRA%' GROUP BY city) SELECT city FROM zebra_counts WHERE cnt = (SELECT MAX(cnt) FROM zebra_counts);`
- `WITH elephant_counts AS (SELECT city, count(*) as cnt FROM AudioData WHERE Animal = 'Elephant' GROUP BY city) SELECT city FROM elephant_counts WHERE cnt = (SELECT MAX(cnt) FROM elephant_counts);`
- `select distinct city from (select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant');`
- `select distinct city from ImageData I where Species LIKE '%MONKEY%' and not exists (select * from AudioData A where A.city = I.city and A.animal = 'Monkey');`
- `select city from (select city from ImageData where Species LIKE '%ZEBRA%') INTERSECT (select city from ImageData where Species LIKE '%IMPALA%');`
- `select city from ((select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant'))  INTERSECT ((select city from ImageData where Species LIKE '%MONKEY%') UNION (select city from AudioData where Animal = 'Monkey'));`
- `select distinct city from (select city from ImageData where Species LIKE '%MONKEY%') INTERSECT (select city from AudioData where Animal = 'Monkey');`
- `WITH zebra_station_counts AS (SELECT city, stationID, count(*) as cnt FROM ImageData WHERE Species LIKE '%ZEBRA%' GROUP BY city, stationID) SELECT city, stationID FROM zebra_station_counts WHERE cnt = (SELECT MAX(cnt) FROM zebra_station_counts);`
