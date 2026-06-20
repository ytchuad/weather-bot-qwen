# Changelog

## [Unreleased] - 2026-06-19

### Fixed
- **HKO Daily Data source corrected**: Changed `fetch_hko_data()` to use `HKO_AWS_CSV_URL` (`https://www.hko.gov.hk/wxinfo/awsgis/hko.csv`) for calculating `max_since_midnight` and `min_since_midnight` instead of the defunct `HKO_MAXMIN_URL`
- **Cache key collision resolved**: Added `_intraday_cache` as separate TTLCache for `get_intraday_state()` function, preventing result overwriting when sharing `_medium_cache` with `fetch_hko_data()`
- **Predictions endpoint fixed**: Changed `forecast = hko.get("forecast")` to `forecast_aws = hko.get("forecast_max")` in `app/api/predictions.py` to use correct key from `fetch_hko_data()`
- **Intraday model integration fixed**: Added `forecast_max` and `forecast_min` parameters to `run_all_models()` and properly forwarded them to `predict_intraday_all()` for intraday model predictions

### Added
- **Rainfall Cards in Hub**: Added 4 rainfall cards (60m, 120m, accumulated today, nowcast) in `weatherElems` with hide/show toggle in dropdown
- **WeatherCards component**: Created `/app/frontend/src/components/WeatherCards.tsx` with individual card toggle functionality
- **Backend support**: Added `get_accumulated_rain_today()` and `get_nowcast_rainfall()` functions in `weather_service.py`
- **API schema**: Extended `WeatherNow` model with `rain_60m`, `rain_120m`, `rain_accumulated_today`, `rain_nowcast` fields

### Changed
- `fetch_hko_data()` now parses AWS CSV data with 24-hour temperature history, filtering by target date to compute max/min since midnight
- `get_intraday_state()` now uses dedicated `_intraday_cache` to avoid cache key collision with `fetch_hko_data()`

### Verified
- All 5 data source health checks in `/api/diagnostics/sources` endpoint now return "ok" status
- Model predictions now return all variants: `9d`, `aws`, `baseline`, `model_a`, `model_b`, `model_c`, `rain_nowcast`