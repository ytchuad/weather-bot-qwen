# Changelog

## [Unreleased] - 2026-07-04

### Fixed
- **Rain data sources: removed stale parquet fallback entirely**: `compute_rain_kwargs()` in `weather_service.py` and `get_nowcast_features()` in `nowcast_loader.py` both had parquet-based fallbacks (`load_rain_15min()` / `_read_wide_all()`) that served month-old data on HF Spaces (ephemeral filesystem). Removed both fallbacks. Live fetch is the only path now; on failure, `rain_data_ok=False` / `rain_nowcast_missing_flag=1` signals models that rainfall features are unavailable.
- **Model visibility persistence across refresh**: `useEffect` in `Hub.tsx` unconditionally re-added all model keys to `visibleKeys` on every page load, silently reverting user's hidden-model choices. Removed the re-add logic; saved preferences are now respected.
- **Wind + pressure cache TTL**: `fetch_wind_stations()` and `fetch_pressure_data()` cache TTL bumped from 60s→300s, with stale-data fallback when fetch returns empty, reducing wind-driven prediction jumps from 0.96°C to ≤0.35°C.
- **Bucket mapping `_market_question_to_bucket()` upper-case loop**: `if temp_val >= lo` was always true on first iteration (lo=23), causing ALL "or higher" markets to return `"23-24"`. Changed to `if lo == temp_val` so "33°C or higher" correctly maps to `">=33"`.
- **Hardcoded bucket ranges removed**: Replaced `range(23, 34)` loops with dynamic `TMAX_BUCKETS` + `_bucket_bounds()`. "or higher" markets return `">=N"` format instead of being forced into range buckets.

### Added
- **model_2a1 (i-lens variant)**: New model variant that uses i-lens forecast data as input. Registered in `MODEL_COLORS`/`MODEL_LABELS` (teal `#0d9488`) in `ModelsComparisonChart.tsx`, `BucketProbsChart.tsx`, and `LABEL_MAP` in `ModelGrid.tsx`.
- **Bucket sort order in dropdown**: `sortBuckets()` function in `BucketProbsChart.tsx` — `<23` and `or below` buckets appear first, `>=N` and `or higher` last, numeric range buckets in ascending order.
- **Global model visibility settings UI**: Gear icon ⚙ in Hub.tsx header opens a settings modal with per-model checkboxes (Show All / Hide All buttons). Reads/writes `"visibleKeys"` in localStorage — same key as ModelGrid toggle. All charts (`ModelsComparisonChart`, `BucketProbsChart`) filter model series to visible keys.

## [Unreleased] - 2026-07-03

### Fixed
- **Model 2A 5-10 min prediction fluctuation (root cause)**: `predict_intraday_tmax_model_2a()` was recomputing `temp_change_30m`, `temp_change_60m`, `temp_volatility_60m`, `temp_acceleration_60m`, `rh_change_60m`, `dew_point_change_60m`, and `dew_point_spread_change_60m` from the raw `temp_buffer`/`rh_buffer` arrays every call. These buffers changed between scheduler ticks because `get_intraday_state()` cache expires every ~60s and the HKO CSV merge+dropped-duplicates produce slightly different `df_today` each time. At 09:03→09:08, identical scalar context features produced a 0.62°C swing due to buffer content changes.
  - `get_intraday_state()` now pre-computes all buffer-derived features (`temp_volatility_60m`, `temp_acceleration_60m`, `rh_change_60m`, `dew_point_change_60m`, `dew_point_spread_change_60m`) as deterministic scalars from `df_today` using `.iloc[-30]` / `.iloc[-60]`
  - `predict_intraday_tmax_model_2a()` now accepts `temp_change_30m_pre`, `temp_change_60m_pre`, `temp_volatility_60m_pre`, `temp_acceleration_60m_pre`, `rh_change_60m_pre`, `dew_point_change_60m_pre`, `dew_point_spread_change_60m_pre` parameters — when provided, uses them instead of buffer-based computation (fallback preserved for other callers)
  - New state fields passed through `model_service.py` `common` dict → `predict_intraday_tmax_all` → `predict_intraday_tmax_model_2a`
- **`time_since_max` binary flip not fully smoothed**: When `temp_now == max_so_far`, `time_since_max` drops from N to 0 instantly. Pre-computed features eliminate indirect buffer coupling, but this feature is still computed as-is (will benefit from future smoothing).
- **Dew point delta computation**: Moved from always-buffer-based (`idx`/`rh_idx`) to pre-computed preferred + buffer fallback, fixing a `NameError` when pre-computed features were provided.

### Added
- **Buffer debug info in context_json**: `buffer_len`, `temp_at_idx30`, `temp_at_idx60`, `rh_at_idx60` now logged in each snapshot for monitoring Model 2A buffer stability.
- New pre-computed state fields exposed in context_json: `temp_volatility_60m`, `temp_acceleration_60m`, `rh_change_60m`, `dew_point_change_60m`, `dew_point_spread_change_60m`.

## [Unreleased] - 2026-07-02

### Changed
- **Snapshot writing separated from strategy cycle status**: `_scheduler_loop` now writes snapshots before calling `run_single_strategy_cycle()`, so snapshots are recorded from the earliest available weather data (~00:15 HKT) regardless of `min_hour` or `check_entry_rules` gates. Extracted snapshot logic into `_write_cycle_snapshot()` helper.
- **`enhanced_v1_paper` min_hour reverted to 8**: Was temporarily 0 for early snapshots; no longer needed after snapshot/strategy separation.

## [Unreleased] - 2026-07-01

### Fixed
- **Model 2A `temp_change_30m`/`temp_change_60m` lookback indices**: `predict_intraday_tmax_model_2a` used `idx-3` / `idx-6` (3/6 minute lookback) instead of `idx-30` / `idx-60` (30/60 minute), making these features ~10x more sensitive to HKO CSV jitter than intended. Same fix applied to `temp_volatility_60m` (window widened from 6 to 60 readings), `temp_acceleration_60m`, `rh_change_60m`, and dew-point deltas (`_t6`→`_t60`). This was the root cause of Model 2A/Model G predictions fluctuating more than their std between 30-second calls, causing Per-Bucket Probability chart to disagree with live Model vs Market Dynamics.
- **`GET /api/charts/bucket-probs` array alignment**: Single-pass padding logic caused model probability arrays to be shorter than timestamps when older snapshots lacked `model_probs`. Rewritten to two-pass: first discover all model keys, then pre-allocate arrays with `None` and fill at correct indices.
- **Bucket-probs endpoint collecting bucket names as model keys**: First pass used `probs.keys()` (bucket names like "33-34") instead of `mk` (model key), causing all model data to be lost and legend to display bucket names instead of model names.
- **BucketProbsChart y-axis formatting**: Formatter used `"{value}%"` which displayed raw values (0→"0%", 0.5→"0.5%") instead of multiplying by 100. Fixed to `(v) => \`${(v*100).toFixed(0)}%\``. Also changed static `max: 1` to dynamic `max(allValues)*1.15` for better scale when data is clustered at low probabilities.
- **Missing `forecast_max`/`forecast_min` in snapshot `run_all_models` calls**: Both `_build_strategy_context()` (strategies.py) and `auto_runner.py` were calling `run_all_models()` without `forecast_max`/`forecast_min`, causing intraday models to use `None` defaults. The live `/api/predictions` endpoint correctly passed these values, leading to systematic probability mismatches between Per-Bucket snapshot chart and Model vs Market Dynamics chart (e.g. Model G showed 8% vs 28.6% for >=34).
- **Model 2A late-night upside quantization**: Quantile predictions (`remaining_upside_q*`) are now conditioned on the classifier output via `remaining_upside_q* *= (1 - prob_max_reached)`. Previously the classifier correctly detected the daily max was reached (zero_prob ~99% after 22:00), but the quantile models independently predicted positive remaining upside, inflating bucket probabilities (e.g. 76% instead of 99%+ for the realized bucket). This is a principled Bayesian composition: classifier gives P(upside=0), quantiles predict conditional distribution when upside > 0.
- **Forecast fallback for late hours**: In `model_service.py`, when the HKO forecast API is unavailable after 20:00, the fallback now uses `max_so_far` directly instead of `max_so_far + 2°C`. At late hours the daily max is finalized, so adding +2°C created a false `forecast_gap` signal that pushed quantile predictions upward.

### Added
- **Bucket Probability Time-Series Chart**: new `BucketProbsChart.tsx` component + `GET /api/charts/bucket-probs` endpoint. Reads per-bucket model probabilities and Polymarket prices from `context_json['model_probs']` + `context_json['market_prices']` in snapshot DB. Includes bucket dropdown selector and ECharts multi-line chart (models + market price over time).
- **Trajectory / Bucket toggle on Hub**: the "Models vs Market — Temperature Tracking" card now has a `[Trajectory │ Bucket]` toggle to switch between historical prediction trajectory (existing `ModelsComparisonChart`) and per-bucket probability time series (new `BucketProbsChart`). Both share the same card, x-axis format, and 120s refetch interval.
- **Per-bucket probabilities stored in snapshots**: `_build_strategy_context()` and `auto_runner.py` now save `model_probs` (per-model per-bucket probabilities) and `market_prices` in `context_json` at every snapshot write.
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
- `app/frontend/src/pages/Hub.tsx`: "Model Expected" now shows 1/std-weighted average of all visible models (was active model's mean); order & visible keys persisted to localStorage

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