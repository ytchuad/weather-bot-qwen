# Phase 2C extension: minute weather and rebuild-safe history

## Architecture

The existing five-minute canonical cycle remains the sole model/inference
path. `CanonicalCycleCollector` warms/persists TMAX and TMIN shared cycles
every 300 seconds without an account context. `WeatherSnapshotCollector` runs separately every 60 seconds and reads
the existing cached HKO intraday state (`get_intraday_state`, backed by the
HKO AWS intraday CSV and local parquet) plus the latest already-completed
model record. It does not import or call model inference, canonical cycle
construction, Gamma CLOB discovery, or depth fetch.

`WeatherSnapshotStore` and `MarketSnapshotStore` use independent temporary
JSONL chunks and the shared 5–15 minute close setting (default 10). Model
cycle storage is unchanged. `HistoricalStore` downloads private Dataset
partitions into `/tmp/layer_a_remote_cache` and merges them with local
`data/layer_a`, `data/layer_a_market` and `data/layer_a_weather` read paths.

## Runtime guarantees

* Weather and market capture failures are isolated from each other, model
  capture, serving and remote upload.
* Close writes SHA-256 manifests and uses atomic rename; incomplete temporary
  files remain discoverable at startup.
* Remote files are never uploaded again, never placed in writable capture
  roots and never treated as local pending partitions.
* Minute model joins are backward as-of only; inference is not rerun.
* No strategy, threshold, Kelly, paper-account or real-money state is part of
  the new records.

## Validation

The implementation adds unit coverage for deterministic weather IDs, zero vs
missing values, truthful source age, collector isolation, ten-minute close and
recovery, remote cache failure/de-duplication, backward model joins, minute
projection, admin authorization, weather-inclusive export and replay delay
probes. Runtime-state and artifact-integrity checks are performed separately so
the model files and paper-account files are not part of the feature diff.

Known limitation: when a historical date has no weather stream, the minute
projection can only report missing weather for minutes represented by stored
model or market records; it does not invent a complete weather series.
