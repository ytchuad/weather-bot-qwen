# Layer A historical UI

## Data path

The UI reads the read-only `HistoricalStore`, not only the current process
files. The store loads cached private Dataset history and current local
capture, then deduplicates by:

```text
decision_cycle_id     model cycles
market_snapshot_id    one-minute market snapshots
weather_snapshot_id   one-minute weather snapshots
```

When a key is present in both places, a schema-valid local record wins over a
remote record; otherwise the valid remote record wins. This deterministic
priority lets current local data appear immediately while remote refresh is
still running and prevents duplicates after a Space rebuild.

## Minute join contract

`GET /api/history/minute` creates a read-only row for every minute represented
by a stored model, market or weather timestamp in the query range.

* Weather uses the same-minute snapshot or latest prior valid snapshot. It
  exposes `weather_source_timestamp`, `weather_age_seconds` and
  `weather_quality_status`.
* Market uses only same-minute snapshots. Each bucket exposes Gamma reference
  price, best bid, best ask, spread, book timestamp, book age and validation
  status.
* Model uses the latest `decision_timestamp <= timestamp` only. Every row
  exposes `model_cycle_timestamp`, `model_cycle_id` and `model_age_seconds`.

Consequently a 09:01 row can use the 09:00 model, but it can never use a
09:05 model. Missing historical weather remains an explicit missing status;
no interpolation is performed.

## API

```text
GET  /api/history/minute
GET  /api/history/model-cycles
GET  /api/history/market-snapshots
GET  /api/history/weather-snapshots
POST /admin/layer-a-history-refresh   # X-Layer-A-Admin-Token or Bearer token
```

The minute endpoint accepts `date`, `start`, `end`, repeated or comma-separated
`bucket` and `model` filters, and `limit`. Public responses contain only
records and sanitized remote status; tokens and private repository settings
are never returned.

## Rebuild behavior

Startup initializes local stores and starts serving immediately. Market and
weather collectors start independently, while remote refresh runs in a daemon
thread. Local rows remain visible during `loading`; after refresh the cache
generation invalidates the read index and remote/local rows become visible in
one deduplicated view. A remote outage changes health to `degraded` or
`unavailable`, but does not stop serving or collecting current data.
