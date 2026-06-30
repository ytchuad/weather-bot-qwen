# Changelog

## [Unreleased] - 2026-06-30

### Added
- **Auto-seed default strategy accounts on startup**: if `data/strategy_accounts.json` is empty, `start_scheduler()` copies from `config/default_strategy_accounts.json` (two accounts: `enhanced_v1_paper` with model_a, `enhanced_v2_paper` with baseline)
- **`context_json` column in snapshot DB**: stores weather state (temp lags, rh, time_since_*), forecast, rainfall data quality flags, and per-model stds for offline debugging
- **Hub KPI summary cards**: Model Expected (°C), Market Expected (°C), Expected Bucket row above the temperature tracking chart
- **ModelGrid confidence interval track**: visual ±1σ bar + mean dot per model card
- **WeatherCards redesign**: from card grid to compact horizontal bar layout

### Changed
- `app/api/strategies.py`:
  - `_build_strategy_context()` now assembles `context_json` dict with ~20 diagnostic variables
  - `start_scheduler()` calls `_seed_default_accounts_if_empty()` before launching the thread
  - Snapshot write passes `context_json` to the logger
- `features/strategy_snapshot_logger.py`: added `context_json TEXT` column + migration; updated `write_snapshot()` and `read_snapshots()`
- `execution/auto_runner.py`: assembles and passes `context_json` on snapshot write
- `app/frontend/src/pages/Hub.tsx`: removed ConsensusTrack; added KPI cards, bucket midpoint parser, tempRange computation for ModelGrid
- `app/frontend/src/components/BucketChart.tsx`: default view changed to "prob", simplified JSX
- `app/frontend/src/components/ModelGrid.tsx`: added `tempRange` prop, confidence interval track bar, top bucket label, Model G label
- `app/frontend/src/components/ModelsComparisonChart.tsx`: reduced height 400→300px, simplified returns
- `app/frontend/src/components/WeatherCards.tsx`: redesigned to horizontal bar layout
- `config/default_strategy_accounts.json` (new): tracked default accounts file for HF Spaces seeding
- `app/frontend/src/pages/Hub.tsx`: "Expected Bucket" now shows Polymarket highest-probability bucket (was model)
- **Scroll fix**: removed nested double scroll containers; Hub no longer has its own `overflow-y-auto`, Layout is the single scroll layer; scrollbar hidden via `[&::-webkit-scrollbar]:hidden`

## [2026-06-29]

### Added
- **Strategy Snapshot Logger**: SQLite-based persistence for per-cycle snapshots
  - `features/strategy_snapshot_logger.py` — writes/reads snapshots (timestamp, Polymarket weighted temp, model predicted temp, actual temp) after each strategy cycle
  - Enabled by piggybacking on the existing background scheduler — no separate data collection thread needed
  - Snapshots recorded automatically every ~5 min per running strategy
- **Models Comparison Chart (Hub page)**: New ECharts line chart showing all models vs Polymarket vs actual temperature
  - `GET /api/charts/models-comparison` endpoint — reads from SQLite, returns all model prediction arrays
  - `app/frontend/src/components/ModelsComparisonChart.tsx` — renders 9+ lines with color-coded legend
  - Snapshot logger now stores `all_model_predictions` (JSON column) from all models in `run_all_models()` results
- **Strategy Temperature Tracking Chart**: New ECharts line chart in the Strategies page
  - Three lines: Polymarket weighted-average temperature, strategy model predicted temperature, actual HKO observed temperature
  - Trade markers on the actual-temperature line (buy/sell annotations)
  - Accessible via "Chart" button on each StrategyCard
  - API endpoint: `GET /api/strategies/{sid}/chart?date=YYYY-MM-DD`
  - Frontend component: `app/frontend/src/components/StrategyChart.tsx`
  - Data served from SQLite — zero model loading, zero live API calls on page load

### Changed
- `app/api/strategies.py`:
  - `_build_strategy_context()` now returns `markets`, `post_mean`, `is_min_temp`, `target_date_str` for snapshot writing
  - `_scheduler_loop()` writes a snapshot to SQLite after each successful strategy cycle
  - New `GET /{sid}/chart` endpoint returns pure time-series arrays for the frontend
  - New `_load_chart_trades()` helper reads trade events from paper_trade_audit.parquet for chart markers
- `app/frontend/src/pages/Strategies.tsx` — added "Chart" button to each StrategyCard, toggling the new StrategyChart component
- `app/frontend/src/components/StrategyChart.tsx` — new ECharts line chart with Polymarket/model/actual temperature series and trade markers
- `app/frontend/src/api/client.ts` — added `fetchStrategyChart()` function
- `app/frontend/src/types/index.ts` — added `StrategyChartData`, `StrategyChartTrade` interfaces
- `models/intraday_inference.py` — removed `_maybe_st_cache_resource()` (Streamlit `st.cache_resource` shim, no longer needed)

## [2026-06-28]

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