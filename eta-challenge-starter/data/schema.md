# Data Schema

All parquet files share the same columns.

| Column | Type | Description |
|---|---|---|
| `pickup_zone` | int32 | NYC taxi zone ID (1–265) |
| `dropoff_zone` | int32 | NYC taxi zone ID (1–265) |
| `requested_at` | string | ISO 8601 timestamp when the ride was requested |
| `passenger_count` | int8 | Number of passengers |
| `duration_seconds` | float64 | Trip duration — **this is the target you predict** |

At inference time the grader sends you the first four columns only. Your
`predict(request: dict) -> float` function returns predicted
`duration_seconds`.

## Zone metadata (optional but useful)

NYC TLC publishes a zone lookup (zone ID → borough, neighborhood name):

    https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

The full shapefile (with polygons) is:

    https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip

Computing zone centroids from the shapefile is a common first move — it gives
you per-zone latitude/longitude, which unlocks haversine distance features,
road-network routing, and neighborhood embeddings.

## What's NOT in the schema (and why)

- **Trip distance**: excluded. Riders don't know actual trip distance at
  request time, so neither does your model. Estimated distance from zone
  centroids is fair game.
- **Weather**: not included in v1. NOAA has public hourly observations for
  Central Park / JFK / LGA if you want to join them yourself:
  https://www.ncei.noaa.gov/access/services/data/v1
- **Fare / tip / tolls**: irrelevant for duration prediction.

## Cleaning applied before you see the data

Performed by `download_data.py`:

- Trips shorter than 30 seconds or longer than 3 hours dropped
- Trips with invalid zone IDs (outside 1–265) dropped
- Missing `passenger_count` filled with 1
- Rows with a pickup timestamp outside calendar 2023 dropped

If you discover additional garbage (we leave plenty), clean it yourself —
that's a fair part of the work.
