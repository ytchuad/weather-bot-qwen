# Changelog

## [Unreleased] - 2026-06-28

### Added
- **Real-Time Inference Parity Framework**: Generic production ML parity framework
  - `config/generic_realtime_parity_framework.yaml` — Framework pipeline & schema definitions
  - `features/source_adapters_base.py` — Generic canonical source adapter (`standardize_source`)
  - `features/shared_feature_builder_base.py` — Shared feature builder contract + validation
  - `inference/realtime_inference_base.py` — 11-step generic inference flow with stop conditions
  - `monitoring/inference_parity_check_base.py` — Replay parity check with tolerances
  - `monitoring/daily_shadow_eval_base.py` — Post-outcome shadow evaluation
  - `monitoring/data_quality_checks_base.py` — 8-category data quality monitoring
  - All generic modules are reusable for future models (rainfall, nowcast, UV, warning)

- **Model 2A Feature Spec & Framework Implementation**:
  - `config/model_2a_feature_spec.yaml` — Complete spec: active hours 06:00-23:50, 10-min grid, feature groups, tolerances, guardrails
  - `features/model_2a_source_adapters.py` — Weather/wind/forecast canonical adapters with temp cleaning & spike detection
  - `features/model_2a_feature_builder.py` — Shared feature builder (46+ features) used by historical, live, and replay modes
  - `inference/model_2a_realtime_inference.py` — End-to-end inference; pred_tmax = max_so_far + upside_qXX (strict)
  - `monitoring/model_2a_inference_parity_check.py` — Replay parity check for Model 2A
  - `monitoring/model_2a_daily_shadow_eval.py` — Shadow evaluation against actual daily highs
  - `monitoring/model_2a_data_quality_checks.py` — Temp spikes, wind coverage, forecast consistency checks

- **Model 2A (Core+Wind)**: New intraday tmax model combining minute observations, forecast gap, wind station data (Ref, Victoria Harbour, Highland, King's Park), pressure, dew point, and temporal features
  - Feature store builder: `data/build_model_2a_feature_store.py`
  - Training script: `models/train_model_2a.py`
  - Inference: `predict_intraday_tmax_model_2a()` added to `models/intraday_inference.py`
  - OOT: MAE=0.222°C, cov80=93.2%, PIW=0.340°C, PR-AUC=0.992
- **UI integration**: Model 2A registered in `app/config.py` (key, label, colour) and `app/frontend/src/components/ModelGrid.tsx` (label)

### Changed
- **Model G target definition fixed**: `remaining_upside` now uses `daily_max_temp - max_so_far` (not `temp_current`); `forecast_gap` uses `forecast_max_temp - max_so_far`; rolling features replaced with shift-based; added `is_upside_zero` classifier target with ≤0.05 threshold
- **.gitignore**: Added `data/wind_data/`, `data/weather_minute_wide.parquet`, `data/wind_features_10min.parquet`, `data/forecast_features.parquet`, `data/model_2a_feature_store.parquet`
- **Model F removed**: Removed `model_f` from `app/config.py`, `intraday_inference.py`, `strategy_builder.py`, `strategy_card.py`, `page_health.py`, `strategy_gate.py`
- **build_model_2a_feature_store.py**: 9 fixes — raw-minute `actual_high_today`/`max_so_far`, `merge_asof` forecast matching, wind rolling max from max col, real timestamp freshness, explicit Victoria Harbour stations, missing flags, validation
- **Documentation**: README.md updated with framework modules in repo structure; CHANGELOG.md tracks all additions

## [2026-06-19]

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