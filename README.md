---
title: Weather Prediction Dashboard
emoji: 🌦️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
---

# Weather Bot Qwen — Probabilistic Temperature Nowcasting & Research System

**HKO daily maximum / minimum temperature probabilistic forecasting and intraday nowcasting research system, with optional paper‑trading simulation layer.**

---

## Disclaimer

This project is for **research, weather modelling, backtesting, and paper‑trading simulation only**.  
It does **not** provide financial advice, investment advice, trading recommendation, or betting recommendation.

- Any real‑money use, exchange integration, or automated order placement must be independently reviewed for legal, regulatory, operational, and personal risk considerations.
- The trading‑related modules are **optional simulation components**. The default operating mode is paper trading.
- The authors assume no liability for any financial loss or other damages arising from the use of this software.

---

## Project Objective

This project builds a **probabilistic weather forecasting and intraday nowcasting engine** for Hong Kong Observatory (HKO) daily maximum and minimum temperature.

The system combines:
- HKO official historical observations
- HKO forecast archives (from i‑lens.hk)
- Long‑horizon XGBoost models for daily mean & spread prediction
- Intraday high‑frequency (10‑minute) HKO temperature observations
- Intraday **LightGBM quantile models** (q10, q25, q50, q75, q90) for remaining upside / downside
- **Rainfall‑aware intraday model** utilising 15‑minute accumulated rainfall data
- **Model A — Minute‑level LightGBM** (temp + RH only, 5‑min resolution, 38 features)
- Empirical baseline fallback
- Calibration, probability mapping, and Bayesian fusion between prior and posterior

The **trading‑related modules** are separated as an optional **research, paper‑trading, and execution‑simulation layer**.

---

## System Layers

### Layer A – Weather Data & Feature Layer
- HKO official daily observations (max / min / mean temperature)
- HKO forecast archives (text, wind, humidity, forecast max/min)
- Intraday 10‑minute HKO temperature observations (from data.gov.hk)
- **15‑minute HKO rainfall** (accumulated since midnight)
- Live HKO 9‑day forecast & live intraday state builder

### Layer B – Probabilistic Forecast Layer
- **Long‑horizon XGBoost model** → daily mean & std (Gaussian prior)
- **Intraday LightGBM quantile model** → remaining upside / downside quantiles
- **Rainfall‑aware intraday model** → same‑day nowcast aware of rain intensity and regime
- **Model A (minute‑level, temp+RH)** → 5‑min resolution quantile model (38 features, temp + RH only)
- **Model B (Rain)** → Model A + rainfall history features at minute granularity
- **Model C (Nowcast)** → Model B + 37 spatial rainfall nowcast features
- **Model D / E** → Enhanced minute-level variants
- **Model G (Gap+Max)** → forecast-gap + max_so_far based intraday model
- **Model 2A (Core+Wind)** → Baseline + forecast + wind station data + pressure + dew point (45 features, OOT MAE=0.222°C, PR-AUC=0.992)
- **Empirical baseline fallback** (lookup tables) for quick reference
- Calibration & probability mapping → bucket probabilities
- **Bayesian fusion** between prior (long‑horizon) and posterior (intraday)

### Layer C – Research / Paper‑Trading Layer
- Market price ingestion (Polymarket Gamma API)
- Kelly allocation simulation (mutually exclusive events, YES/NO dual‑side)
- CLOB slippage simulation (order‑book depth)
- Paper positions & PnL tracking
- Forward‑test performance tracking
- **Strategy-Centric Architecture**: Self-contained strategies with per‑strategy capital, model selection, market template, and gate pipeline
- **Headless Auto-Runner**: GitHub Actions cron (every 5 min) runs enabled strategies outside Streamlit

---

## Tech Stack

- **Language**: Python 3.14 (Streamlit Cloud)
- **Data**: pandas, numpy, xarray, Parquet
- **Machine Learning**: XGBoost (long‑horizon), LightGBM (intraday quantile & classifiers)
- **Dashboard**: Streamlit (modular `app/` package), Plotly
- **APIs**: requests (HKO, Polymarket Gamma, Polymarket CLOB)
- **Deployment**: GitHub + Streamlit Cloud, GitHub Actions

---

## Data Sources

| Data | Source |
|------|--------|
| HKO official daily Tmax / Tmin / Tmean | HKO Climate Data Service (CSV) |
| Historical HKO forecasts | i‑lens.hk daily extracts |
| Intraday 10‑min HKO temperature | data.gov.hk (ZIP files, 1‑min / 10‑min) |
| **Intraday 15‑min HKO rainfall** | i‑lens.hk (history / instant charts) |
| **Intraday 1‑min HKO temperature + RH** | i‑lens.hk (scraped history, 4.9M rows) + HKO live CSV |
| Live 9‑day forecast | HKO Open Data API |
| Live intraday temperature | HKO AWS GIS CSV (`hko.csv`) |
| Polymarket market prices & token IDs | Polymarket Gamma API |
| Polymarket order books | Polymarket CLOB API |

---

## Data Pipeline

1. **Official observations** → `build_hko_obs.py` → `hko_tmax_historical.parquet`
2. **Forecast archive** → `build_hko_forecast_dataset.py` → `hko_historical_forecasts.parquet`
3. **Intraday temperature** → `build_intraday_obs.py` → `intraday_hko_10min.parquet`
4. **Rainfall** → `download_rainfall.py` → `hko_rainfall_15min.parquet`
5. **Rainfall features** → `build_rainfall_features.py` → `hko_rainfall_15min_features.parquet`
6. **Training sets**:
   - Long‑horizon: `build_training_set.py` → `training_set_max.parquet` / `training_set_min.parquet`
   - Intraday ML (rain‑aware): `build_intraday_ml_dataset.py` → `intraday_ml_train.parquet`
   - Empirical lookup: `build_intraday_lookup.py` → `lookup_upside.parquet` / `lookup_downside.parquet`
   - **Minute‑level (Model A)**: `scrape_hko_history.py` → `hko_history.parquet` → `build_intraday_minute_features.py` → `intraday_minute_ml_features.parquet`

All Parquet files are stored under `data/`.

---

## Model Architecture

### Long‑Horizon XGBoost Model

- **Goal**: Predict daily Tmax / Tmin conditional Gaussian distribution (μ, σ)
- **Features**: HKO forecast max/min, wind direction, weather keywords, humidity, delta features, lead days, seasonality
- **Method**: Two‑step XGBoost (mean regressor + absolute error regressor → std)
- **Output**: `predict_distribution()` → (mean, std) → `predict_bucket_probabilities()` → per‑bucket probabilities

### Intraday LightGBM Quantile Model

#### Intraday Target Definition

For each intraday snapshot time *t* on date *D*:
- `max_so_far_t = max HKO temperature observed from 00:00 to t`
- `min_so_far_t = min HKO temperature observed from 00:00 to t`

- `remaining_upside_t = official_Tmax_D - max_so_far_t`
- `remaining_downside_t = min_so_far_t - official_Tmin_D`

- `remaining_upside_t = max(0, remaining_upside_t)`
- `remaining_downside_t = max(0, remaining_downside_t)`
Predictions are reconstructed as:
- `pred_Tmax_quantile = max_so_far_t + predicted_remaining_upside_quantile`
- `pred_Tmin_quantile = min_so_far_t - predicted_remaining_downside_quantile`


This enforces `pred_Tmax >= max_so_far` and `pred_Tmin <= min_so_far` by construction.

#### Anti‑leakage Rules (Point‑in‑Time)

For any intraday snapshot *t*, the model must **only use information available at or before t**.

Do not use:
- future temperature observations after *t*
- final daily Tmax / Tmin as features
- forecast updates issued after *t*
- full‑day derived statistics as intraday features
- settlement result or market outcome in live prediction features

Official daily Tmax / Tmin may **only** be used as training labels or backtesting labels.

#### Model Design

- **Input features**: current temp, max_so_far, min_so_far, temp_change_10/30/60/120 min, rolling means, minutes_since_midnight, hour, month, seasonal sin/cos, forecast_tmax, forecast_tmin, time_since_max/min_so_far, **rainfall_60m, rainfall_120m, rainfall_max_30m/60m, rain_cooling_60m, post_peak_rain_flag, morning_peak_rain_flag, drop_from_max**.
- **Outputs**:
  - 5 LightGBM quantile regressors (`objective='quantile'`, α = 0.10, 0.25, 0.50, 0.75, 0.90) for remaining upside / downside
  - 2 binary classifiers for `is_upside_zero` / `is_downside_zero`
- **Monotonicity**: Predicted quantiles are sorted per sample to guarantee q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90
- **Early‑hour handling**: Longer lags are imputed with shorter lags or current values; training data includes all hours (00:00–23:50)
- **Rainfall‑aware**: The model learns the effect of rain on temperature, especially the “morning peak then heavy rain” regime where daily max temperature is reached early and followed by cooling. This is captured through rolling rainfall, rain‑cooling interactions, and regime flags.

### Model A — Minute‑Level Temp+RH Quantile Model

A separate model family trained on scraped HKO minute-level temperature and relative humidity data (4.9M rows, 2016–2026).

- **Architecture**: 5 LightGBM quantile regressors (α=0.10/0.25/0.50/0.75/0.90) + 1 binary classifier for upside_zero.
- **Hyperparams**: `max_depth=6`, `num_leaves=31`, `lr=0.03`, `n_estimators=1500`, `subsample=0.8`, `colsample=0.8`, `min_data_in_leaf=300`, `reg_lambda=1.0`.
- **Features**: 38 features — temp trends (5m/15m/30m/60m), temp acceleration, RH trends (15m/30m/60m), dew point, cyclic time encoding. No rainfall or forecast features.
- **5-min deterministic grid** (every row where `minute % 5 == 0`).
- **OOT results**: MAE=0.614°C, 80% PI coverage=84.5%, avg PIW=1.93°C, PR-AUC=0.877.
- **Known limitation**: Coverage drops from 84.8% (no-rain) to 79.1% (rain) — to be addressed by Model B.

### Model B & C (Planned)

- **Model B**: Adds 8 rainfall history features (accumulation + cooling interactions) to Model A to close the rain/no-rain gap.
- **Model C**: Adds 37 spatial rainfall nowcast features on top of Model B for the most comprehensive minute-level predictor.

### Fusion & Calibration

`combine_with_prior()` merges the long‑horizon prior (mean, std) with the intraday posterior (quantile outputs). Current default weight is **0.0** (full intraday), validated by hourly fusion analysis. A standard‑deviation scale factor (`std_scale=0.9`) is applied to improve coverage.

---

## Model Validation

### Long‑horizon XGBoost
- MAE / RMSE for Tmax and Tmin mean prediction
- PIT histogram & reliability diagram
- Brier score for bucket probabilities

### Intraday LightGBM
- Pinball loss for q10 / q25 / q50 / q75 / q90
- p10–p90 interval coverage (target ~80%)
- Interval width by hour
- `upside_zero` / `downside_zero` classifier calibration
- Separate validation for early hours (00:00–02:00)
- **Rain‑regime validation**: slices for no‑rain, rain, heavy rain, post‑peak rain, morning‑peak rain

All metrics are generated by `models/validate_models.py` and exported to `reports/model_validation_report.xlsx`.

---

## Paper‑Trading & Execution Simulation

**All execution modules currently operate as paper‑trading or simulation components** unless explicitly configured otherwise. Real order placement is **disabled by default**.

### Strategy-Centric Architecture (New)

The system was rebuilt from a **portfolio-centric** to a **strategy-centric** model:

- **StrategyAccount** (`data/strategy_accounts.json`): Each strategy is self-contained with its own `id`, `label`, `model`, `capital`, `market_template`, params (`bias`, `std_mult`, `kelly_fraction`), and `status` (running/paused).
- **Market Templates** (`hk-tmax`, `hk-tmin`): Auto-resolve to today's Polymarket slug. No manual event discovery needed.
- **Strategy Dashboard**: Live cards with ON/OFF toggle, per-strategy PnL, expandable positions/trades/params.
- **Background Scheduler**: Session-level daemon thread (30s poll, 5-min cooldown) for interactive use.
- **Headless Auto-Runner** (`execution/auto_runner.py`): CLI entry point for GitHub Actions cron (every 5 min). Reads strategy_accounts.json, resolves markets, runs cycles, updates data files, commits back.
- **Per-Strategy Gate Pipelines**: Entry/Exit/Sizing/Rebalance gates are configured in `config/paper_strategies.json` (V2 format) or overridden per-strategy.

### Kelly Allocation (Mutually Exclusive Events)
- `compute_multi_kelly_bets()`: numerical optimisation maximising expected log wealth
- Supports both BUY YES and BUY NO (short side)
- Constraints: max 15% per bucket, total exposure ≤ 50%, half‑Kelly fraction
- Independent simple Kelly also available for quick dashboard display

### CLOB Slippage Simulation
- Fetches order book depth from Polymarket CLOB API (or mock mode)
- Walks the asks/bids to estimate average fill price and actual contracts acquired
- Adjusts bet quantities accordingly

### Dynamic Rebalancing
- Recalculates target positions every time new forecast/intraday data arrives
- Compares with current paper positions (`current_positions.json`)
- Generates buy/sell orders to reach target
- Tracks unrealised PnL correctly for both YES and NO positions

### Performance Tracking
- Dashboard displays cumulative PnL curves, ROI, Brier scores
- Separate accounts for 9‑day, AWS, and intraday‑fusion engines
- Audit log (`paper_trade_audit.parquet`) with `run_id`, `paper_model_key`, `scheduler_source`, `strategy_version`

### Strategy Registry & Runner
- **8 paper strategies** defined in `config/paper_strategies.json` (baseline, rain‑observed, rain‑nowcast, gated‑ensemble, enhanced‑v1, enhanced‑v2, enhanced‑v2‑aggressive, enhanced‑v2‑conservative)
- **V2 config-driven strategies**: All gate thresholds adjustable via JSON — no code changes needed
- **`strategy_account.py`** — StrategyAccount dataclass + StrategyAccountStore (CRUD for `data/strategy_accounts.json`)
- **`auto_runner.py`** — headless CLI for cron-based unattended execution
- **`market_templates.py`** — auto-resolve market slugs from date
- Each strategy has `paper_only: true` (enforced by `validate_strategy_config`)
- Multi‑account isolation: positions stored as `current_positions[strategy_key][slug]`
- **Night skip**: scheduler skips execution when `hkt_now().hour < 8`

---

## Dashboard (Streamlit)

The dashboard has been restructured from the old portfolio-centric layout into a **strategy-centric** interface:

### Pages
- **Hub** (`📊`) — Overview dashboard with key metrics
- **Intraday** (`📈`) — Intraday temperature path, remaining upside/downside, rainfall metrics
- **Strategies** (`⚡`) — Unified strategy dashboard with three sub-tabs:
  - **Live** — Toggle strategies ON/OFF, view per‑strategy PnL, positions, trade history, gate evaluations
  - **Builder** — Create/edit strategy config: select model, set market template, tune gate parameters, save
  - **Lab** — Synthetic backtest (PaperTradeHarness) with parameter sweeps and cross-strategy comparison
- **Analytics** (`📉`) — Forward test performance, PnL charts, Brier scores
- **Health** (`🏥`) — Runtime checks (compilation, smoke tests, feature schemas, model registry)

### Key Design Principles
- **Strategy-Centric**: Each strategy is a self-contained entity with its own capital, model, market template, and gate pipeline. No portfolio bundling.
- **Per-Strategy Parameters**: `bias`, `std_mult`, `kelly_fraction`, `capital` are set per-strategy in card expanders, not globally in the sidebar.
- **Auto-Resolved Markets**: Market templates (`hk-tmax`, `hk-tmin`) auto-resolve to the correct Polymarket slug for today's date — no manual event discovery.
- **Market cards**: Each strategy is rendered as a card with toggle, PnL summary, expandable position table, and trade log.
- **Auto-refresh**: Background scheduler daemon (session-level thread) polls `strategy_accounts.json` every 30 seconds; only strategies with `scheduler_on: true` and 5-minute cooldown are run.

### Sidebar
- Date picker for weather data alignment
- Force-refresh and cache-clearing buttons
- Sync HKO forecast button

**Model labeling**: Each model's decision detail section is labelled with its model name (基準線 Baseline / 降雨觀測 / 即時降雨預報 / 閘門集成) for easy identification.

**Live demo** (if available): *[Streamlit Cloud URL]*

---

## Memory Management

The app runs on Streamlit Cloud's 1 GB RAM limit. The following optimisations are applied:

- **Background scheduler** runs as a daemon thread (not `streamlit-autorefresh`) — avoids extra page re-renders
- **Lazy library imports**: `xgboost` and `scipy.stats.norm` are imported on first call rather than at module load — saves ~70–140 MB of peak startup RAM
- **`@st.cache_resource`** on `load_models()` and `_load_models()` ensures model objects are created once per process (via `_maybe_st_cache_resource` helper that works both in and out of Streamlit)
- **`@st.cache_data(ttl=120)`** on `get_intraday_state()` and `_load_rain_data_cached()` — parquet reads are cached, preventing 3 MB of I/O per auto‑refresh
- **`.gitignore` exclusions** — ~150 MB of unused files (ECMWF GRIB, candidate model directories, duplicate `active/`/`archive/` models, HTML/PNG visualisations) are excluded from deployment
- **Trade log & PnL history** are capped (1000 and 500 entries respectively) in `st.session_state`
- **`__init__.py`** in `models/` and `features/` ensures regular package behaviour in Python 3.14 (avoids namespace‑package resolution failures)

## Deployment & Automation

- **Streamlit Cloud**: Automatically deploys from `main` branch on push.
- **GitHub Actions**:
  - `daily_update.yml` — Daily data sync (intraday, forecast, model performance) at 00:30 and 12:30 UTC.
  - `hourly_update.yml` — Forward test logger and auto-rebalancer every hour.
  - `run_strategies.yml` — Headless strategy execution every 5 minutes (reads strategy_accounts.json, runs enabled strategies, trades paper positions, commits data changes).
- **Secrets**: API keys, wallet credentials, or trading keys must **never** be committed. Use Streamlit secrets and GitHub Actions secrets.

---

## Quick Start

### Minimal Run (using pre‑built data & models)
```bash
pip install -r requirements.txt
streamlit run app/main.py                       # modular app (recommended)
# or
streamlit run dashboard.py                      # legacy entry point

## Full Rebuild (from raw inputs)
1. Prepare historical observations
python data/build_hko_obs.py
python data/build_hko_forecast_dataset.py

2. Long‑horizon model
python features/build_training_set.py
python models/train_probabilistic_model.py

3. Intraday temperature data
python features/build_intraday_obs.py

4. Rainfall data
python data/download_rainfall.py
python features/build_rainfall_features.py

5. Intraday models (rain‑aware)
python features/build_intraday_ml_dataset.py
python models/train_intraday_ml.py   # or retrain_full_rain_model.py
python models/copy_rain_model.py

6. Minute‑level Model A (temp + RH only)
python scripts/scrape_hko_history.py           # scrape minute history (4.9M rows)
python features/build_intraday_minute_features.py
python models/train_minute_model_a.py

7. Launch dashboard
streamlit run app/main.py

### Strategy Runner (Headless CLI)
```bash
python -m execution.auto_runner                 # run all due strategies
python -m execution.auto_runner --force          # skip cooldown check
python -m execution.auto_runner --list           # list enabled strategies
```

## Current Status & Roadmap
### Completed
✅ Full data pipeline & feature engineering

✅ Long‑horizon XGBoost model (mean & std)

✅ Intraday LightGBM quantile model (including early‑hour support)

✅ Rainfall‑aware intraday model with interval and regime features

✅ Empirical baseline lookup

✅ Multi‑outcome Kelly + CLOB slippage simulation

✅ Dynamic rebalancing & paper PnL tracking

✅ Streamlit dashboard with live rainfall integration

✅ Strategy runner system (8 paper‑only strategies, config-driven gates)

✅ Strategy-Centric Architecture — per-strategy accounts, market templates, auto-resolved slugs

✅ Strategy Builder — create/edit strategy config with model selector, gate tuning, save

✅ Headless auto-runner (GitHub Actions cron, 5-minute cycles)

✅ Deprecated old Portfolio + Execute pages in favour of unified Strategy Dashboard

✅ Multi‑account paper trading with audit trail

✅ Comprehensive smoke tests (92 gate tests passing)

✅ Manual testing helpers (pyboard)

✅ Model A — minute-level temp+RH quantile model (38 features, 5-min grid)

✅ Model A inference integrated in dashboard (model selector, comparison tab)

✅ Model B — rainfall-augmented minute model (8 rainfall features, 46 features total)

✅ Model C — full nowcast minute model (37 spatial nowcast features, 83 features total)

✅ Model G (Gap+Max) — forecast-gap + max_so_far based intraday model

✅ Model 2A (Core+Wind) — 45 features incl. wind station data, pressure, dew point; OOT MAE=0.306°C, cov80=88.7%

✅ Real-Time Inference Parity Framework — generic production ML parity framework separating (A) generic framework from (B) model-specific specs

## Roadmap

### Near‑Term: Model & Framework

- **Model D/E Tmin** — Cross-midnight and morning-minimum minute models (already trained)
- Backtest Models B/C through paper‑trader pipeline to measure PnL impact
- Dashboard integration for Models B/C (model selector + comparison tab)
- Scheduled weekly retraining of all minute models
- **Framework rollout** — Apply real-time inference parity framework to Models B/C/D/E/G

### Model & Data Quality

- Automated data‑quality reports for intraday observations

- By‑hour validation report for intraday LightGBM

- Scheduled model retraining with locked validation period

- Integrate radar / lightning nowcast features for convective weather regimes

### Paper‑Trading & Strategy

- Improved paper‑trading audit log with per-strategy filtering
- Scenario stress tests for Kelly allocation
- Max drawdown & exposure monitoring per strategy
- PnL attribution by model source and strategy
- More strategy templates in the Builder (drag-and-drop gate composition)
- Enhanced Lab tab — parameter sweeps, automatic "promote to live"

### Optional Execution Layer (disabled by default)

- Real CLOB order placement (with manual approval workflow)
- Secrets management & operational kill switch
- Execution audit trail

## Known Limitations

- HKO official daily extrema may differ slightly from 10‑minute intraday derived extrema.

- Early‑day predictions carry higher uncertainty because limited intraday observations are available.

- Rainfall data starts from 2023‑06‑01, so models trained on earlier periods lack rain features.

- Intraday LightGBM models require strict anti‑leakage controls; always validate with point‑in‑time splits.

- Probability outputs should be monitored with reliability diagrams and forward‑test logs.

- Trading‑related components are for simulation and research only.

- Polymarket market definitions and settlement rules may change; validate independently.

## Repository Structure (abbreviated)
```
Weather_Bot_Qwen/
├── data/                   # raw & processed data (parquet, state JSON)
│   ├── strategy_accounts.json    # per-strategy accounts & state
│   ├── current_positions.json    # paper positions by strategy_key
│   ├── pnl_history/              # per-strategy PnL snapshots
│   └── auto_runner_log.json      # headless execution audit trail
├── app/                    # Streamlit modular app package
│   ├── main.py                  # entry point, navigation, scheduler
│   ├── pages/
│   │   ├── page_hub.py          # overview dashboard
│   │   ├── page_intraday.py     # intraday temperature visualization
│   │   ├── page_strategies.py   # Live / Builder / Lab tabs
│   │   ├── page_analytics.py    # forward-test performance
│   │   └── page_health.py       # runtime checks
│   └── components/
│       ├── sidebar.py           # global sidebar (date picker, sync)
│       ├── strategy_card.py     # per-strategy card with toggle & PnL
│       └── strategy_builder.py  # strategy creation form & gate tuning
├── features/               # feature builders & dataset constructors
│   ├── source_adapters_base.py   # generic canonical source adapter
│   ├── shared_feature_builder_base.py  # shared feature builder contract
│   ├── model_2a_source_adapters.py  # Model 2A canonical source adapters
│   └── model_2a_feature_builder.py  # Model 2A shared feature builder
├── inference/              # real-time inference framework
│   ├── realtime_inference_base.py  # generic inference flow
│   └── model_2a_realtime_inference.py # Model 2A inference
├── monitoring/             # production ML monitoring
│   ├── inference_parity_check_base.py  # generic replay parity
│   ├── daily_shadow_eval_base.py       # generic shadow eval
│   ├── data_quality_checks_base.py     # generic data quality checks
│   ├── model_2a_inference_parity_check.py  # Model 2A parity check
│   ├── model_2a_daily_shadow_eval.py       # Model 2A shadow eval
│   └── model_2a_data_quality_checks.py     # Model 2A data quality
├── models/                 # training, inference, saved models
│   ├── train_minute_model_a.py   # Model A (temp+RH, 5-min)
│   ├── train_minute_model_b.py   # Model B (+rain hist)
│   ├── train_minute_model_c.py   # Model C (+nowcast)
│   ├── train_model_2a.py         # Model 2A (+wind + forecast)
├── execution/              # strategy runner, Kelly, slippage, rebalancer
│   ├── strategy_account.py       # StrategyAccount dataclass + persistence
│   ├── strategy_runner.py        # cycle execution dispatch
│   ├── market_templates.py       # auto-resolve Polymarket slugs from date
│   ├── auto_runner.py            # headless CLI for cron execution
│   ├── gates/                    # 30+ pluggable gate functions
│   ├── paper_trade_harness.py    # synthetic backtest harness
│   ├── strategy_config.py        # Strategy dataclass, pipeline builder
│   ├── strategy_factory.py       # factory from paper_strategies.json
│   ├── kelly_betting.py
│   ├── clob_slippage.py
│   └── rebalancer.py
├── config/                 # paper_strategies.json (8 strategies)
│   ├── generic_realtime_parity_framework.yaml  # generic framework spec
│   └── model_2a_feature_spec.yaml              # Model 2A feature spec
├── tests/                  # pytest test suite (92 gate tests)
├── .github/workflows/
│   ├── daily_update.yml         # daily data sync
│   ├── hourly_update.yml        # hourly forward-test & rebalancer
│   └── run_strategies.yml       # headless strategy execution (every 5 min)
├── app/main.py                  # Streamlit entry point (recommended)
├── dashboard.py                 # legacy single-file entry point (deprecated)
├── config.yaml                  # execution.allow_live_orders: false
├── requirements.txt
└── README.md
```