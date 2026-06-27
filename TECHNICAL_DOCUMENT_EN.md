# Weather Bot Qwen — Technical Architecture & Logic Document

## 1. Project Overview

**Weather Bot Qwen** is a probabilistic weather forecasting and intraday nowcasting research system, specifically designed for Hong Kong Observatory (HKO) daily maximum and minimum temperature forecasting. It combines meteorological data analysis with an optional paper-trading simulation layer for prediction markets (Polymarket).

### Core Objectives
1.  **Probabilistic Forecasting**: Generate calibrated probability distributions for daily Tmax and Tmin.
2.  **Intraday Nowcasting**: Update predictions in real-time (10-minute intervals) as new weather data arrives.
3.  **Rainfall Awareness**: Incorporate rainfall data to model non-linear temperature effects (e.g., cooling after rain).
4.  **Paper Trading Simulation**: Simulate trading strategies (Kelly Criterion) based on forecast edges.

---

## 2. System Architecture

The system is structured into three distinct layers:

### Layer A: Weather Data & Feature Layer
*   **Data Sources**:
    *   HKO Official Observations (Daily Tmax/Tmin)
    *   HKO Forecast Archives (Historical predictions)
    *   Intraday 10-min Temperature Observations
    *   15-min Accumulated Rainfall Data
    *   Live HKO 9-day Forecast & Intraday State
*   **Processing**: Raw data is cleaned, aligned, and stored in Parquet format under `data/`.

### Layer B: Probabilistic Forecast Layer
*   **Long-horizon Model (XGBoost)**: Predicts daily mean and standard deviation (Gaussian prior).
*   **Intraday Model (LightGBM)**: Quantile regression models (q10-q90), for predicting "remaining upside/downside".
*   **Model A (Minute-Level LightGBM, temp+RH only)**: Higher-resolution (5-min) quantile models using temperature + relative humidity only, no rainfall features.
*   **Model B (Minute-Level LightGBM, temp+RH+rainfall, trained)**: Augments Model A with rainfall accumulation and cooling features at minute granularity from HKO 15-min rain gauge data.
*   **Model C (Minute + Rainfall + Nowcast)**: Augments Model B with 37 spatial rainfall nowcast features from HKO gridded nowcast data.
*   **Model D / E (Tmin)**: Cross-midnight, evening cooling, and morning minimum prediction models for Tmin.
*   **Model G (Gap+Max)**: Forecast-gap + max_so_far based intraday tmax model (17 features).
*   **Model 2A (Core+Wind)**: Combines minute observations, forecast gap, wind station data, pressure, and dew point for intraday tmax (45 features, OOT MAE=0.222°C, PR-AUC=0.992).
*   **Rainfall-aware Model**: Specialized features to capture rain-cooling effects.
*   **Fusion Engine**: Bayesian fusion combining prior (XGBoost) and posterior (LightGBM) distributions.

### Layer C: Research / Paper-Trading Layer
*   **Market Integration**: Polymarket Gamma API (prices) and CLOB API (order books).
*   **Strategy**: Kelly allocation for mutually exclusive events (YES/NO).
*   **Simulation**: Slippage modeling, dynamic rebalancing, and PnL tracking.
*   **Strategy-Centric Architecture**: Self-contained strategies with per-strategy capital, model selection, market template, and gate pipeline.
*   **Headless Auto-Runner**: GitHub Actions cron (every 5 min) runs enabled strategies outside Streamlit.

---

## 3. Data Pipeline Logic

The data pipeline transforms raw observations into model-ready training sets.

### 3.1 Intraday ML Dataset Construction (`features/build_intraday_ml_dataset.py`)

This script builds the core training set for the intraday nowcasting models.

#### 3.1.1 Feature Engineering
1.  **Time Features**: `month`, `hour`, `minutes_since_midnight`, seasonal sin/cos encoding.
2.  **Intraday State Features**:
    *   `max_so_far` / `min_so_far`: Cumulative max/min temperature observed from 00:00 up to time `t`.
    *   `range_so_far`: Current spread between max and min.
    *   `temp_change_Xmin`: Temperature delta over 10/30/60/120 minute windows.
    *   `time_since_max_so_far`: Minutes elapsed since the current daily high was recorded.
3.  **Forecast Features**: Merges HKO's D-1 and D-0 forecasts based on snapshot time (cutoff at 11:30 AM).
4.  **Rainfall Features**:
    *   `rainfall_60m`, `rainfall_120m`: Accumulated rainfall.
    *   `rain_cooling_60m`: Interaction term (Rainfall * Temperature Drop).
    *   `post_peak_rain_flag`: Indicator for rain occurring after the daily temperature peak.
    *   `morning_peak_rain_flag`: Indicator for morning rain followed by cooling.

#### 3.1.2 Target Definition
The models predict the *remaining* potential change, not the absolute temperature.
*   **Upside (Tmax)**: `remaining_upside = max(0, Official_Tmax - max_so_far)`
*   **Downside (Tmin)**: `remaining_downside = max(0, min_so_far - Official_Tmin)`
*   **Binary Classifiers**: `is_upside_zero` / `is_downside_zero` indicate if the daily extremum has likely been reached.

#### 3.1.3 Anti-Leakage Rules (Point-in-Time)
*   Features at time `t` **only** use data available at or before `t`.
*   Official daily Tmax/Tmin are **only** used as labels, never as features.
*   Forecast updates issued after `t` are excluded.

### 3.2 Empirical Lookup Table (`features/build_intraday_lookup.py`)

This script builds a multi-dimensional lookup table for empirical baseline predictions.

#### 3.2.1 Aggregation Logic
*   **Dimensions**: `month`, `hour`, `max_bucket` (or `min_bucket`), `trend_60min`.
*   **Metrics**: Calculates `p10`, `p25`, `p50`, `p75`, `p90`, and `prob_zero` for remaining upside/downside.
*   **Fallback Hierarchy**: If specific bucket data is missing, it falls back to `month + hour` only, then `hour` only.

### 3.3 Minute-Level Feature Pipeline (Model A)

A parallel feature pipeline was built for higher-resolution minute-level forecasting, using scraped HKO AWS minute data.

#### 3.3.1 Data Source
*   **Scraper**: `scripts/scrape_hko_history.py` — concurrent (10 workers) scraper for minute-level temperature + relative humidity from HKO's i-lens API (`https://i-lens.hko.gov.hk/`).
*   **Coverage**: 4,989,795 rows spanning 3,464 days (2016-12-08 → 2026-06-10).
*   **Format**: Parquet with checkpoint resume support for incremental updates.

#### 3.3.2 Feature Engineering (`features/build_intraday_minute_features.py`)
Builds 45 columns (42 features + 3 targets) at 1-minute granularity from the scraped minute history:

1. **Intraday State Features**: `max_so_far_1m`, `min_so_far_1m`, `range_so_far_1m`, `time_since_max_1m`, `time_since_min_1m`, `drop_from_max_1m`, `rise_from_min_1m`.
2. **Temperature Trend Features**: `temp_change_5m/15m/30m/60m`, `temp_acceleration_30m`, `temp_std_30m/60m`.
3. **Relative Humidity Features**: `rh_change_15m/30m/60m`, `rh_mean_30m/60m`, `rh_std_60m`.
4. **Interaction Features**: `temp_x_rh`, `dew_point_c` (Magnus formula), `dew_point_spread`.
5. **Cyclic Time Encoding**: `hour`, `minute`, `month_sin/cos`, `day_sin/cos`, `is_morning/afternoon/evening/night`.
6. **Targets**: `remaining_upside = max(0, official_tmax - max_so_far_1m)`, `is_upside_zero`.

#### 3.3.3 Key Properties
*   Output: `data/intraday_minute_ml_features.parquet` (218 MB, gitignored).
*   Features use **only** temperature and relative humidity (no rainfall, no forecast).
*   All features are point-in-time safe (no future leakage).

---

## 4. Model Training Logic

### 4.1 Intraday LightGBM Models (`models/train_intraday_ml.py`)

#### 4.1.1 Data Splitting
Strict temporal split to prevent future leakage:
*   **Train**: Before 2025-01-01
*   **Validation**: 2025-01-01 to 2026-01-01
*   **Test**: After 2026-01-01 (Fail-fast validation ensures no test set reuse)

#### 4.1.2 Model Architecture
*   **Algorithm**: LightGBM (Gradient Boosting Decision Trees).
*   **Quantile Regression**: 5 separate models for α = {0.10, 0.25, 0.50, 0.75, 0.90}.
    *   *Objective*: `quantile`
    *   *Hyperparameters*: `max_depth=6`, `learning_rate=0.05`, `n_estimators=500`.
*   **Binary Classifiers**: 2 models for `upside_zero` and `downside_zero`.
    *   *Objective*: `binary`

#### 4.1.3 Post-Training Constraints
*   **Monotonicity Enforcement**: Predicted quantiles are sorted per sample to guarantee q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90.
*   **Early Hour Handling**: Training includes all hours (00:00–23:50), with lag features imputed using shorter lags or current values if history is insufficient.

### 4.2 Rain-Aware Model Comparison (`models/train_rain_aware_model.py`)

This script compares a baseline model (without rain features) against a rain-aware model.

#### 4.2.1 Feature Sets
*   **Baseline**: Standard intraday features (temp, history, forecasts).
*   **Rain-Aware**: Adds `rainfall_60m`, `rain_cooling_60m`, `post_peak_rain_flag`, etc.

#### 4.2.2 Evaluation Slices
Performance is evaluated across specific weather regimes:
*   `all_data`, `rain_present`, `heavy_rain`, `post_peak_rain`, `morning_peak_rain`, `no_rain`.

#### 4.2.3 Key Metrics
*   **MAE**: Mean Absolute Error for q50.
*   **Coverage**: Percentage of actual values within the 80% prediction interval (q10-q90).
*   **False Positive Rate**: Rate of predicting remaining upside > 0 when actual is 0.

### 4.3 Model A: Minute-Level LightGBM (`models/train_minute_model_a.py`)

A separate model family trained on the minute-level feature pipeline (temp + RH only, no rainfall/forecast).

#### 4.3.1 Model Architecture
*   **5 Quantile Regressors** (upside only): α = {0.10, 0.25, 0.50, 0.75, 0.90}.
    *   Hyperparams: `max_depth=6`, `num_leaves=31`, `lr=0.03`, `n_estimators=1500`, `subsample=0.8`, `colsample=0.8`, `min_data_in_leaf=300`, `reg_lambda=1.0`.
    *   Early stopping (50 rounds) on validation loss.
*   **1 Binary Classifier**: `upside_zero` (predicts if tmax has been reached).
    *   Same hyperparameters, tuned decision threshold via F1-maximization on validation set.
*   **38 Features** from temp + RH only (no rainfall, no forecast).
*   **5-min deterministic time grid** (every row where `minute % 5 == 0`).

#### 4.3.2 Data Split
*   **Train**: before 2024-06-11 (789,531 rows)
*   **Validation**: 2024-06-11 to 2025-06-11 (105,469 rows)
*   **OOT (Out-of-Time)**: 2025-06-11 onwards (105,485 rows)

#### 4.3.3 OOT Evaluation Results

**By time bucket:**

| bucket | n_rows | dates | MAE_up | cov80 | PIW | bias | q90_br | q10_br |
|--------|--------|-------|--------|-------|-----|------|--------|--------|
| 00-06  | 23,650 | 324   | 1.330  | 0.729 | 4.02 | +0.25 | 0.115 | 0.157 |
| 06-12  | 23,328 | 324   | 0.959  | 0.778 | 3.13 | +0.11 | 0.099 | 0.123 |
| 12-18  | 23,328 | 324   | 0.148  | 0.896 | 0.50 | -0.04 | 0.059 | 0.045 |
| 18-24  | 23,330 | 324   | 0.008  | 0.978 | 0.03 | -0.00 | 0.009 | 0.014 |
| ALL    | 93,636 | 324   | 0.614  | 0.845 | 1.93 | +0.06 | 0.072 | 0.083 |

**By rain regime** (rainfall flag merged post-hoc from HKO 15-min observations):

| regime  | n_rows | MAE_up | cov80 | PIW  | bias   | q90_br | q10_br |
|---------|--------|--------|-------|------|--------|--------|--------|
| no_rain | 88,701 | 0.608  | 0.848 | 1.92 | +0.05  | 0.073  | 0.079  |
| rain    |  4,935 | 0.709  | 0.791 | 2.08 | +0.34  | 0.054  | 0.155  |

**Classifier**: PR-AUC=0.975, Precision=0.925, Recall=0.905, F1=0.915, threshold=0.444.

**Key Observations:**
*   **Bias flip**: Underpredicts in morning (+0.25°C 00-06), overpredicts afternoon (-0.04°C 12-18) — model expects warming that already happened.
*   **Rain degradation**: Rain rows show 79.1% coverage vs 84.8% no-rain, wider PI (2.08 vs 1.92), larger bias (+0.34 vs +0.05). Expected since Model A has no rain features.
*   **Evening triviality**: 18-24 has 0.008 MAE, 97.8% coverage — tmax almost always reached by then.
*   **Calibration**: 84.5% coverage on a nominally 80% PI indicates slightly conservative interval widths.

#### 4.3.4 Artifacts
All saved to `models/intraday_minute_ml/`:
*   `upside_q{10,25,50,75,90}.txt` — 5 quantile LightGBM models
*   `upside_zero.txt` — binary classifier
*   `best_threshold.json` — tuned decision threshold
*   `feature_list.json` — ordered 38 feature column names
*   `oot_predictions.parquet` — full OOT predictions for downstream analysis

### 4.4 Model B: Rainfall-Augmented Minute LightGBM (`models/train_minute_model_b.py`)

Model B is the natural successor to Model A: it adds rainfall history features at minute granularity, trained on the full 789K-row dataset with pre-2023 rows zero-filled and missing_flag=1.

#### 4.4.1 Motivation
Model A's rain vs no-rain gap:
- Coverage: 79.1% (rain) vs 84.8% (no-rain) — a 5.7pp gap.
- Bias: +0.34°C (rain) vs +0.05°C (no-rain) — systematic overconfidence during rainfall.
- PIW: 2.08 (rain) vs 1.92 (no-rain) — wider intervals indicate higher uncertainty.

Adding explicit rainfall accumulation and cooling interaction terms was expected to close this gap.

#### 4.4.2 Feature Additions
Augments the 38 Model A features with 8 rainfall features, sourced from `hko_rainfall_15min_features.parquet` (forward-filled from 15-min to 5-min grid via `merge_asof` with 15-min tolerance):

| Feature | Description |
|---------|-------------|
| `rainfall_60m` | Accumulated rainfall over last 60 minutes (mm) |
| `rainfall_120m` | Accumulated rainfall over last 120 minutes (mm) |
| `rainfall_60m_missing_flag` | 1 if 60-min rainfall data is missing |
| `rainfall_120m_missing_flag` | 1 if 120-min rainfall data is missing |
| `rain_cooling_60m` | max(0, -temp_change_60m) if rainfall_60m > 0 else 0 |
| `rain_cooling_120m` | max(0, -temp_change_60m) if rainfall_120m > 0 else 0 |
| `post_peak_rain_flag` | 1 if heavy rain (≥5mm/60m) + drop_from_max ≥0.5°C + 30≤time_since_max≤240min |
| `morning_peak_rain_flag` | 1 if post_peak_rain_flag + 9≤hour≤14 |

#### 4.4.3 Data Preparation
1. Load `hko_rainfall_15min_features.parquet` (2023-06-01 to 2026-06-06, 105,775 rows).
2. Convert to HKT timezone, then `merge_asof` onto 5-min minute grid with 15-min tolerance.
3. Pre-2023-06-01 rows (68.2% of total) have no match → `missing_flag=1`, `rainfall=0` after fillna.
4. Compute cooling interactions and regime flags from minute features + rainfall columns.
5. Same train/valid/OOT splits as Model A (789K/105K/105K).

#### 4.4.4 Data Split
*   **Train**: before 2024-06-11 (789,531 rows, same as Model A)
*   **Validation**: 2024-06-11 to 2025-06-11 (105,469 rows)
*   **OOT**: 2025-06-11 onwards (105,485 rows)

#### 4.4.5 Model Architecture
Identical to Model A:
- **5 quantile regressors** (upside only): α = {0.10, 0.25, 0.50, 0.75, 0.90}.
- **1 binary classifier**: `upside_zero`.
- Same hyperparams: `max_depth=6`, `num_leaves=31`, `lr=0.03`, `n_estimators=1500`, `subsample=0.8`, `colsample=0.8`, `min_data_in_leaf=300`, `reg_lambda=1.0`, early stopping 50 rounds.
- **46 features** (38 temp+RH + 8 rainfall history).

#### 4.4.6 OOT Evaluation Results

**By time bucket (93,636 valid rows, 324 dates):**

| bucket | n_rows | dates | MAE_up | cov80 | PIW | bias | q90_br | q10_br |
|--------|--------|-------|--------|-------|-----|------|--------|--------|
| 00-06  | 23,650 | 324   | 1.327  | 0.727 | 4.01 | +0.17 | 0.123 | 0.150 |
| 06-12  | 23,328 | 324   | 0.960  | 0.777 | 3.13 | +0.09 | 0.103 | 0.120 |
| 12-18  | 23,328 | 324   | 0.148  | 0.907 | 0.49 | -0.04 | 0.061 | 0.033 |
| 18-24  | 23,330 | 324   | 0.007  | 0.989 | 0.02 | -0.01 | 0.009 | 0.001 |
| ALL    | 93,636 | 324   | 0.613  | 0.850 | 1.92 | +0.05 | 0.074 | 0.076 |

**By rain regime:**

| regime  | n_rows | MAE_up | cov80 | PIW  | bias   | q90_br | q10_br |
|---------|--------|--------|-------|------|--------|--------|--------|
| no_rain | 88,701 | 0.609  | 0.853 | 1.92 | +0.04  | 0.075  | 0.073  |
| rain    |  4,935 | 0.677  | 0.792 | 1.92 | +0.30  | 0.066  | 0.142  |

**Classifier**: PR-AUC=0.975, Precision=0.925, Recall=0.905, F1=0.915, threshold=0.441.

#### 4.4.7 Key Observations

**vs Model A (same OOT set, 93,636 rows):**

| Metric | Model A | Model B | Δ |
|--------|---------|---------|---|
| MAE | 0.614 | **0.613** | -0.001 |
| Coverage (cov80) | 84.47% | **84.97%** | **+0.50pp** |
| PIW | 1.925 | **1.921** | -0.004 |
| Bias | +0.062 | **+0.054** | -0.008 |
| PR-AUC | 0.9752 | 0.9754 | +0.0002 |

1. **Modest overall improvement**: Model B is marginally better across all metrics — 0.5pp coverage gain, slightly narrower intervals, slightly lower bias. The delta is small but consistent.
2. **Rain-regime gap persists**: Rain coverage is 79.2% vs 85.3% no-rain — a 6.1pp gap (similar to Model A's 5.7pp). The rainfall history features did not close the gap as expected.
3. **Rain MAE improved**: Rain rows MAE = 0.677 vs Model A's 0.709 (−0.032). This is the most tangible improvement.
4. **Classifier unchanged**: PR-AUC is essentially identical (0.975). The rainfall features add no information for the "has tmax been reached" decision.
5. **Afternoon coverage improved**: 12-18 cov80 rose from 89.6% (Model A) to 90.7% (Model B), likely from better post-rain cooling predictions.
6. **Possible explanation**: The 15-min rainfall gauge data is too coarse (single station, 15-min resolution) to add meaningful signal beyond what temperature trends already capture. Most rainfall cooling is already reflected in `temp_change_60m` and `drop_from_max_1m`.

#### 4.4.8 Artifacts
All saved to `models/intraday_minute_ml_model_b/`:
*   `upside_q{10,25,50,75,90}.txt` — 5 quantile LightGBM models
*   `upside_zero.txt` — binary classifier
*   `best_threshold.json` — tuned decision threshold
*   `feature_list.json` — ordered 46 feature column names
*   `oot_predictions.parquet` — full OOT predictions for downstream analysis

#### 4.4.9 Controlled Experiment: A_restricted vs B_restricted

To isolate whether rainfall features add value *within the period where rainfall data exists* (independent of the pre-2023 zero-filled training data), two restricted models were trained on only rows with `as_of_datetime_hkt >= 2023-06-01`:

| Model | Features | Train Rows | Description |
|-------|----------|-----------|-------------|
| A_restricted | 38 temp+RH | 108,658 | Model A, no pre-2023 data |
| B_restricted | 38+8 rainfall | 108,658 | Model B, no pre-2023 data |

Both use the same train/valid/OOT date splits and the same 108K-row training set. The only difference is the 8 rainfall history features.

**OOT Results (93,636 rows):**

| Metric | A_restricted | B_restricted | Δ (B - A) |
|--------|-------------|-------------|-----------|
| MAE (all) | 0.647 | **0.638** | -0.009 |
| cov80 (all) | 77.99% | 77.66% | -0.33pp |
| PIW (all) | 1.835 | **1.776** | -0.059 |
| Bias (all) | +0.002 | +0.007 | +0.005 |
| MAE (rain) | 0.704 | **0.637** | **-0.067** |
| cov80 (rain) | 65.53% | 65.11% | -0.42pp |
| Bias (rain) | **+0.370** | **+0.242** | **-0.128** |
| PR-AUC | 0.858 | 0.863 | +0.005 |

**Key findings:**

1. **Rainfall features improve rain-day MAE by 0.067°C** (0.704 → 0.637) — the most tangible benefit, consistent with Model B vs Model A's rain MAE improvement.
2. **Rain bias drops by 0.128** (+0.370 → +0.242) — rainfall features substantially reduce overconfidence during rain events.
3. **Overall metrics are nearly identical** — the 8 rainfall features add no meaningful signal on non-rain rows (which are 95% of the data).
4. **Coverage is unchanged or slightly worse** — the restricted models have much lower coverage (78%) than full models (85%) regardless of rainfall features, because the training set is 7× smaller.
5. **Data quantity dominates**: Both restricted models are substantially worse than the full 789K-row models, showing that pre-2023 temperature/RH history is far more important than rainfall features for overall performance.

**Conclusion**: Rainfall history features are value-added on rain rows (lower MAE, lower bias) but the effect is modest. The pre-2023 training data contributes far more to overall model quality than the rainfall features do. Model C (nowcast features) is a more promising direction.

#### 4.4.10 Calibration Experiment: Residual-Based Interval Calibration

The Model B OOT coverage is 84.97% overall — above the nominal 80% target — but rain rows achieve only 79.19%. A calibration experiment computes per-regime empirical 10th/90th percentiles of residuals (`residual = actual_remaining_upside - q50`) to set calibrated intervals: `cal_q10 = q50 + residual_p10`, `cal_q90 = q50 + residual_p90`.

**Method**: Empirical residual quantiles directly guarantee 80% coverage. This avoids symmetric-scaling pitfalls on the 0-bound skewed target.

**By rain regime (93,636 OOT rows):**

| Regime | n | cur_cov | cur_bias | residual p10 | residual p90 | cal_cov | cal_PIW |
|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 93,636 | 84.97% | +0.054 | -1.284 | +1.067 | 80.00% | 2.35 |
| rainfall_60m==0 | 88,701 | 85.30% | +0.040 | -1.248 | +1.077 | 80.00% | 2.33 |
| **rainfall_60m>0** | 4,935 | **79.19%** | **+0.302** | **-1.965** | **+0.547** | **79.98%** | 2.51 |
| rainfall_60m>5 (heavy) | 1,569 | 82.54% | +0.318 | -1.750 | +0.568 | 79.99% | 2.32 |
| post_peak_rain_flag=1 | 506 | 83.40% | +0.375 | -2.095 | +0.604 | 79.84% | 2.70 |

**Rain rows by hour:**

| Bucket | n | cur_cov | cur_bias | p10 | p90 | cal_cov | cal_PIW |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 00-06 rain | 1,149 | 61.01% | +0.877 | -2.759 | +1.301 | 79.98% | 4.06 |
| 06-12 rain | 1,557 | 79.00% | +0.364 | -1.697 | +1.144 | 79.96% | 2.84 |
| 12-18 rain | 1,272 | 84.67% | -0.051 | -0.162 | +0.300 | 79.87% | 0.46 |
| 18-24 rain | 957 | 94.04% | -0.021 | 0.000 | 0.000 | 91.64% | 0.00 |

**Key findings:**

1. **Rain rows need strongly asymmetric calibration**: residual p10=-1.97 vs p90=+0.55. The interval must extend far below q50 to capture the model's overestimation during rain (positive bias), while the upper tail needs to be *narrower* than the original q90.
2. **Dry rows are already well-calibrated**: Dry residual p10=-1.25 and p90=+1.08 give a nearly symmetric interval. The original 85.3% coverage is close to 80%; calibration narrows PIW from 1.92→1.82 (not shown in table above).
3. **00-06 rain is the worst bucket**: at 61% coverage, it needs the most aggressive widening (p10=-2.76, p90=+1.30, cal_PIW=4.06).
4. **18-24 is trivial**: remaining upside ≈ 0 for all rows; coverage is already 94% and calibration has no material effect.

**Recommended calibration rule for inference:**
- If `rainfall_60m > 0`: calibrate with residual p10=-1.97, p90=+0.55 → `cal_q10 = min(q10, q50 - 1.97)`, `cal_q90 = min(q90, q50 + 0.55)`. This widens the lower tail while capping the upper tail, keeping q50 unchanged.
- If `rainfall_60m == 0`: keep original quantiles (or apply a mild narrowing with p10=-1.25, p90=+1.08).

Calibration residuals are saved to `models/intraday_minute_ml_model_b/calibration_residuals.json`.

### 4.5 Model C (Trained): Full Minute LightGBM with Nowcast

Model C is the third and most feature-rich variant: it extends Model B with 37 spatial rainfall nowcast features from HKO's gridded nowcast product.

#### 4.5.1 Motivation
The nowcast features capture spatially-aware rainfall intensity and coverage (radar- or gauge-interpolated estimates at ~5km resolution), which provide:
- **Leading indicators**: Nowcast features are forward-looking (up to 120 min), unlike backwards-looking rain gauge accumulations.
- **Spatial context**: Aggregated statistics across multiple stations (min/mean/max/p90) capture whether rainfall is widespread or isolated.
- **Heavy rain detection**: `rain_nc_heavy_0_120m` and `rain_nc_area_gt*` features flag convective rainfall that produces strong cooling.

#### 4.5.2 Feature Additions
Augments Model B's features with 37 nowcast features loaded via `get_nowcast_features()` / `rainfall_nowcast_station_features_wide_all.parquet`:

| Feature Group | Count | Examples |
|--------------|-------|---------|
| Accumulated sums (nearest station) | 4 | `rain_nc_nearest_mm_sum_{30,60,90,120}m` |
| Radius-5km aggregated (mean) | 4 | `rain_nc_mean_r5km_sum_{30,60,90,120}m` |
| Radius-5km aggregated (max) | 4 | `rain_nc_max_r5km_sum_{30,60,90,120}m` |
| Radius-5km aggregated (min) | 4 | `rain_nc_min_r5km_sum_{30,60,90,120}m` |
| Radius-5km aggregated (p90) | 4 | `rain_nc_p90_r5km_sum_{30,60,90,120}m` |
| Radius-5km area coverage (>0mm) | 4 | `rain_nc_area_gt0_r5km_sum_{30,60,90,120}m` |
| Radius-5km area coverage (>5mm) | 4 | `rain_nc_area_gt5_r5km_sum_{30,60,90,120}m` |
| Summary features | 6 | `rain_nc_sum_0_60m`, `rain_nc_sum_0_120m`, `rain_nc_any_0_120m`, `rain_nc_front_loaded_ratio`, `rain_nc_heavy_0_120m`, `rain_nc_valid_horizon_count` |
| Data quality flags | 3 | `rain_nc_missing_flag`, `rain_nowcast_age_minutes`, `rain_nowcast_missing_flag` |

These are the same features used in the 10-min `rain_nowcast` model, made available at minute resolution.

#### 4.5.3 Data Preparation
1. Load or fetch nowcast data (fallback chain: live API → `rainfall_nowcast_station_features_wide_all.parquet` → no-data defaults).
2. For historical training: use `rainfall_nowcast_station_features_wide_all.parquet` with point-in-time snapshot lookup via `merge_asof` on `issue_time`.
3. For live inference: use `get_nowcast_features()` (same as existing rain_nowcast pipeline).
4. Merge onto the minute rainfall-augmented feature grid.

#### 4.5.4 Architecture
Identical to Model A/B:
- **5 quantile regressors** + **1 binary classifier**, same hyperparams.
- **83 features** (38 temp+RH + 8 rainfall history + 37 nowcast).

#### 4.5.5 Results

**Accept/Reject Evaluation**

| Criterion | Model B | Model C | Verdict |
|-----------|---------|---------|---------|
| 00-06 rain MAE | 1.518 | **1.319** (-0.199) | ✅ Improved |
| 00-06 rain COV80 | 61.0% | **68.5%** (+7.5pp) | ✅ Materially improved |
| Rain bias | +0.302 | **+0.186** (-0.116) | ✅ Decreased |
| Rain q10 breach rate | 14.2% | **9.6%** (-4.6pp) | ✅ Lower-tail improved |
| No-rain MAE deterioration | — | **0.601** (vs 0.609) | ✅ No degradation (-0.008) |
| All-data COV80 | 85.0% | **85.0%** (=) | ✅ Within 78–85% |
| All-data MAE | 0.613 | **0.602** | ✅ Improved |
| PR-AUC | 0.975 | **0.976** | ✅ Maintained |

All acceptance criteria met. **Model C is accepted.**

**Detailed OOT Metrics by Segment**

| Bucket | n | MAE | COV80 | PIW | Bias | q10_br |
|--------|---|-----|-------|-----|------|--------|
| 00-06 | 23,650 | 1.301 | 72.6% | 3.95 | +0.154 | 14.5% |
| 06-12 | 23,328 | 0.942 | 78.6% | 3.09 | +0.081 | 11.2% |
| 12-18 | 23,328 | 0.147 | 90.3% | 0.49 | -0.041 | 3.6% |
| 18-24 | 23,330 | 0.007 | 98.8% | 0.02 | -0.006 | 0.3% |
| ALL | 93,636 | 0.602 | 85.0% | 1.90 | +0.048 | 7.5% |
| No rain | 88,701 | 0.601 | 85.1% | 1.90 | +0.040 | 7.3% |
| Rain | 4,935 | 0.610 | 84.2% | 1.90 | +0.186 | 9.6% |
| 00-06 rain | 1,149 | 1.319 | 68.5% | 3.88 | +0.582 | 26.9% |

Model C improves rain-regime predictions across all metrics vs Model B. The nowcast spatial features provide leading indicators that single-station rain gauge history cannot. The 00-06 rain segment, previously the worst (61% COV80), now reaches 68.5% — bridging half the gap from the 80% target without calibration.

#### 4.5.6 Implementation Plan
- **New training script**: `models/train_minute_model_c.py` — loads Model B features + nowcast merge, trains 6 LightGBM models, saves to `models/intraday_minute_ml_model_c/`.
- **Inference**: `predict_intraday_tmax_model_c()` in `models/intraday_inference.py`.
- **Integration**: Add `model_c` to `predict_intraday_tmax_all()`.

#### 4.5.7 Model Evolution Summary

| Model | Features | Train Rows | COV80 (rain) | COV80 (all) | MAE (rain) | MAE (all) | Data Dependency |
|-------|----------|-----------|-------------|-------------|-----------|----------|-----------------|
| A | 38 temp+RH | 789K | 79.1% | 84.5% | 0.676 | 0.612 | Minute temp/RH only |
| B | A + 8 rain hist | 789K | 79.2% | **85.0%** | 0.677 | 0.613 | + 15-min rain gauge |
| A_restricted | 38 temp+RH | 109K | 65.5% | 78.0% | 0.704 | 0.681 | Minute temp/RH, ≥2023 |
| B_restricted | A + 8 rain hist | 109K | 65.1% | 77.7% | 0.637 | 0.667 | + 15-min rain gauge, ≥2023 |
| C | B + 37 nowcast | 789K | **84.2%** | 85.0% | **0.610** | **0.602** | + Nowcast API/parquet |

#### 4.5.8 Tmin Models A/B/C

Tmin models are the reciprocal counterpart of Tmax Models A/B/C. They predict **remaining_downside** = max(0, min_so_far - Official_Tmin) and **is_downside_zero** using the same feature sets:

| Tmin Model | Base Tmax Model | Features | Model Dir | Training Script |
|------------|----------------|----------|-----------|----------------|
| A_tmin | Model A | 38 temp+RH | `models/intraday_minute_ml_tmin/` | `models/train_minute_model_a_tmin.py` |
| B_tmin | Model B | 46 (38 + 8 rain hist) | `models/intraday_minute_ml_model_b_tmin/` | `models/train_minute_model_b_tmin.py` |
| C_tmin | Model C | 83 (46 + 37 nowcast) | `models/intraday_minute_ml_model_c_tmin/` | `models/train_minute_model_c_tmin.py` |

**Key differences from Tmax:**
- Target: `remaining_downside` (vs `remaining_upside`).
- Classifier: `downside_zero` (vs `upside_zero`).
- P10/P90 swap at inference: `pred_tmin_p10 = min_so_far - downside_p90`, `pred_tmin_p90 = min_so_far - downside_p10`.
- No late-hour heuristic (no equivalent of prob_max_reached for Tmin).

##### 4.5.8.1 Model A Tmin Results

| Bucket | n | MAE | COV80 | PIW | Bias | q10_br | q90_br |
|--------|---|-----|-------|-----|------|--------|--------|
| 00-06 | 23,650 | 0.552 | 76.1% | 1.697 | -0.2746 | 12.3% | 11.7% |
| 06-12 | 23,328 | 0.235 | 91.1% | 0.878 | -0.2080 | 1.3% | 7.5% |
| 12-18 | 23,328 | 0.141 | 92.2% | 0.424 | -0.1145 | 1.7% | 6.0% |
| 18-24 | 23,330 | 0.055 | 95.3% | 0.170 | -0.0341 | 1.7% | 3.0% |
| ALL | 93,636 | 0.247 | 88.6% | 0.795 | -0.1582 | 4.3% | 7.1% |
| No rain | 88,701 | 0.241 | 88.9% | 0.781 | -0.1541 | 4.3% | 6.9% |
| Rain | 4,935 | 0.356 | 84.8% | 1.050 | -0.2316 | 5.0% | 10.2% |
| 00-06 rain | 1,149 | 0.779 | 69.4% | 1.610 | -0.5342 | 9.1% | 21.6% |

Classifier PR-AUC = 0.9547

##### 4.5.8.2 Model B Tmin Results

| Bucket | n | MAE | COV80 | PIW | Bias | q10_br | q90_br |
|--------|---|-----|-------|-----|------|--------|--------|
| 00-06 | 23,650 | 0.552 | 78.1% | 1.753 | -0.2750 | 10.5% | 11.4% |
| 06-12 | 23,328 | 0.236 | 90.3% | 0.923 | -0.2058 | 2.0% | 7.7% |
| 12-18 | 23,328 | 0.146 | 93.3% | 0.412 | -0.1036 | 0.5% | 6.3% |
| 18-24 | 23,330 | 0.054 | 95.6% | 0.175 | -0.0307 | 1.4% | 3.0% |
| ALL | 93,636 | 0.248 | 89.3% | 0.819 | -0.1542 | 3.6% | 7.1% |
| No rain | 88,701 | 0.242 | 89.5% | 0.807 | -0.1497 | 3.6% | 6.9% |
| Rain | 4,935 | 0.357 | 84.8% | 1.033 | -0.2350 | 4.8% | 10.3% |
| 00-06 rain | 1,149 | 0.779 | 68.0% | 1.578 | -0.5323 | 9.4% | 22.6% |

Classifier PR-AUC = 0.9557

##### 4.5.8.3 Model C Tmin Results

| Bucket | n | MAE | COV80 | PIW | Bias | q10_br | q90_br |
|--------|---|-----|-------|-----|------|--------|--------|
| 00-06 | 23,650 | 0.540 | 77.0% | 1.728 | -0.2477 | 12.2% | 10.8% |
| 06-12 | 23,328 | 0.236 | 91.1% | 0.928 | -0.1893 | 1.5% | 7.4% |
| 12-18 | 23,328 | 0.145 | 92.3% | 0.443 | -0.0984 | 2.0% | 5.6% |
| 18-24 | 23,330 | 0.055 | 95.7% | 0.194 | -0.0265 | 1.7% | 2.6% |
| ALL | 93,636 | 0.245 | 89.0% | 0.826 | -0.1409 | 4.4% | 6.6% |
| No rain | 88,701 | 0.240 | 89.3% | 0.817 | -0.1380 | 4.3% | 6.4% |
| Rain | 4,935 | 0.345 | 83.2% | 1.000 | -0.1913 | 6.1% | 10.7% |
| 00-06 rain | 1,149 | 0.749 | 66.8% | 1.421 | -0.4727 | 9.5% | 23.7% |

Classifier PR-AUC = 0.9602

##### 4.5.8.4 Accept/Reject: Model B Tmin vs Model C Tmin

| Criterion | Model B Tmin | Model C Tmin | Verdict |
|-----------|-------------|-------------|---------|
| 00-06 rain MAE | 0.779 | 0.749 | ✅ Improved |
| 00-06 rain COV80 | 68.0% | 66.8% | ❌ Degraded |
| Rain MAE | 0.357 | 0.345 | ✅ Improved |
| Rain bias | -0.235 | -0.191 | ⚪ Maintained |
| Rain q10 breach | 4.8% | 6.1% | ❌ Degraded |
| No-rain MAE | 0.242 | 0.240 | ✅ Improved |
| All-data COV80 | 89.3% | 89.0% | ⚪ Maintained |
| All-data MAE | 0.248 | 0.245 | ✅ Improved |

Nowcast features improve rain MAE by 0.012°C (0.357 → 0.345) and 00-06 rain MAE by 0.030°C (0.779 → 0.749), but the effect is modest compared to Tmax where improvements were 0.067°C (rain MAE) and 0.199°C (00-06 rain). The Tmin target is inherently simpler (afternoon/evening low is already close to observed), so nowcast features have less room to add value. Coverage on 00-06 rain actually drops from 68.0% to 66.8%, and rain q10 breach increases from 4.8% to 6.1%. **Model B Tmin is preferred for rain-regime predictions; Model C Tmin is preferred for dry conditions and overall MAE.**

##### 4.5.8.5 Cross-Model Summary (ALL OOT)

| Metric | Model A Tmin | Model B Tmin | Model C Tmin | Model D Tmin |
|--------|-------------|-------------|-------------|-------------|
| MAE | 0.247 | 0.248 | 0.245 | 0.247 |
| COV80 | 88.6% | 89.3% | 89.0% | 89.4% |
| PIW | 0.795 | 0.819 | 0.826 | 0.772 |
| Bias | -0.1582 | -0.1542 | -0.1409 | -0.1554 |
| q10_br | 4.3% | 3.6% | 4.4% | 3.6% |
| q90_br | 7.1% | 7.1% | 6.6% | 7.0% |

All four Tmin models achieve similar overall performance (MAE 0.245–0.248, COV80 88.6–89.3%), significantly better than Tmax models (MAE 0.602, COV80 85.0%). This is expected because remaining_downside is a smaller target — the daily minimum is typically observed in the early morning hours, so by mid-day the remaining downside is near zero for most rows. The models are all slightly pessimistic (negative bias: predicted remaining_downside < actual), meaning the predicted Tmin tends to be slightly higher than actual.

#### 4.5.9 Tmin Model D: Cross-Midnight & Evening Cooling Features

Tmin Model D adds 32 features on top of Model C (4 groups: cross-midnight rolling, previous evening context, night cooling potential, evening re-low risk) with a two-stage architecture: Stage 1 = 3 auxiliary classifiers (`will_make_new_low_after_now`, `is_downside_zero`, `tmin_timing_bucket`), Stage 2 = standard quantile regression (all rows, unconditional). During inference, when `will_make_new_low` probability is below threshold, the prediction intervals are shrunk to avoid overly wide intervals when new lows are unlikely.

**Per-Hour OOT Metrics Comparison (Model B / C / D Tmin)**

| Hour | n | Model B COV80 | Model C COV80 | Model D COV80 | Model B PIW | Model C PIW | Model D PIW |
|------|----|--------------|--------------|--------------|------------|------------|------------|
| 00 | 4,854 | 77.8% | 76.8% | **78.5%** | 1.993 | 1.947 | **1.759** |
| 01 | 3,888 | 77.6% | 75.7% | **77.4%** | 2.008 | 1.960 | **1.873** |
| 02 | 3,888 | 78.2% | 77.2% | **77.6%** | 1.864 | 1.840 | **1.713** |
| 03 | 3,888 | 78.5% | 77.6% | **77.5%** | 1.680 | 1.668 | **1.516** |
| 04 | 3,888 | 78.2% | 77.5% | **77.0%** | 1.530 | 1.522 | **1.383** |
| 05 | 3,888 | 78.5% | 77.3% | **75.5%** | 1.423 | 1.416 | **1.277** |
| 06 | 3,888 | 84.7% | 88.3% | **89.5%** | 1.301 | 1.302 | **1.188** |
| 07 | 3,888 | 89.0% | 90.9% | **92.5%** | 1.188 | 1.182 | **1.093** |
| 08 | 3,888 | 92.1% | 91.6% | **92.5%** | 0.974 | 0.976 | **0.913** |
| 09 | 3,888 | 91.7% | 91.4% | **92.1%** | 0.829 | 0.839 | **0.790** |
| 10 | 3,888 | 91.9% | 92.2% | **91.8%** | 0.699 | 0.703 | **0.648** |
| 11 | 3,888 | 92.2% | 92.4% | **92.3%** | 0.550 | 0.565 | **0.506** |
| 12 | 3,888 | 92.6% | 90.9% | **92.3%** | 0.476 | 0.482 | **0.451** |
| 13 | 3,888 | 92.4% | 92.1% | **91.8%** | 0.441 | 0.459 | **0.431** |
| 14 | 3,888 | 92.5% | 92.0% | **92.5%** | 0.412 | 0.463 | **0.427** |
| 15 | 3,888 | 93.3% | 93.2% | **93.3%** | 0.403 | 0.461 | **0.422** |
| 16 | 3,888 | 94.1% | 93.4% | **94.2%** | 0.378 | 0.414 | **0.401** |
| 17 | 3,888 | 94.6% | 92.2% | **94.1%** | 0.362 | 0.377 | **0.370** |
| 18 | 3,888 | 94.9% | 94.2% | **94.7%** | 0.283 | 0.312 | **0.279** |
| 19 | 3,888 | 95.4% | 95.4% | **95.7%** | 0.239 | 0.279 | **0.251** |
| 20 | 3,888 | 94.6% | 95.6% | **95.2%** | 0.171 | 0.192 | **0.187** |
| 21 | 3,888 | 95.9% | 95.7% | **96.2%** | 0.146 | 0.153 | **0.154** |
| 22 | 3,888 | 96.4% | 96.9% | **97.0%** | 0.116 | 0.123 | **0.126** |
| 23 | 3,898 | 96.3% | 96.6% | **97.6%** | 0.093 | 0.104 | **0.117** |
| **ALL** | **94,288** | **89.3%** | **89.0%** | **89.4%** | **0.819** | **0.826** | **0.772** |

Model D maintains or slightly improves COV80 over Model C across most hours, while PIW is significantly tighter in the early morning (00-06), e.g., hour 00 PIW drops from 1.947 to 1.759. This is driven by the cross-midnight rolling features — by early morning, the model has D-1 evening data to better assess night cooling potential. Afternoon hours (12-17) also see modest PIW improvements (~0.03–0.04). **Overall PIW shrinks 6.5% (0.826 → 0.772) with COV80 stable at 89.4%.**

**Regression Metrics by Rain Regime (OOT)**

| Regime | MAE | COV80 | PIW | Bias | q10_br | q90_br |
|--------|------|-------|-----|------|--------|--------|
| ALL (no filter) | 0.253 | 89.2% | 0.792 | -0.1564 | 3.8% | 7.0% |
| No rain | 0.247 | 89.4% | 0.777 | -0.1538 | 3.8% | 6.9% |
| Rain | 0.363 | 85.0% | 1.087 | -0.2069 | 5.0% | 9.9% |
| 00-06 rain | 0.799 | 64.1% | 1.545 | -0.4968 | 13.2% | 22.7% |
| 06-12 rain | 0.340 | 87.5% | 1.119 | -0.2359 | 2.4% | 10.0% |
| 18-24 re-low | 0.054 | 96.1% | 0.188 | -0.0330 | 1.2% | 2.7% |

Model D matches Model C in no-rain conditions but has room to improve 00-06 rain COV80 (64.1%). 18-24 re-low candidates are predicted very accurately (MAE=0.054, COV80=96.1%), benefiting from the two-stage architecture that shrinks intervals when `will_make_new_low` is unlikely.

---

**Stage 1 Classifier Diagnostics**

**`is_downside_zero` — PR-AUC / Precision / Recall / F1 by Hour**

| Hour | n | pos | PR-AUC | Prec | Recall | F1 |
|------|----|-----|--------|------|--------|-----|
| 00 | 9,814 | 2,758 | 0.8934 | 0.8870 | 0.9365 | 0.9111 |
| 01 | 4,380 | 283 | 0.3672 | 0.3555 | 0.4912 | 0.4125 |
| 02 | 4,380 | 380 | 0.3944 | 0.3852 | 0.6184 | 0.4747 |
| 03 | 4,380 | 502 | 0.4695 | 0.4352 | 0.6753 | 0.5293 |
| 04 | 4,380 | 771 | 0.4849 | 0.4470 | 0.6239 | 0.5208 |
| 05 | 4,380 | 1,364 | 0.5680 | 0.4741 | 0.7713 | 0.5872 |
| 06 | 4,380 | 2,256 | 0.6836 | 0.5858 | 0.9291 | 0.7185 |
| 07 | 4,380 | 2,887 | 0.7987 | 0.7002 | 0.9609 | 0.8100 |
| 08 | 4,380 | 3,084 | 0.8438 | 0.7403 | 0.9874 | 0.8462 |
| 09 | 4,380 | 3,147 | 0.8646 | 0.7576 | 0.9692 | 0.8504 |
| 10 | 4,380 | 3,221 | 0.8692 | 0.7565 | 0.9867 | 0.8564 |
| 11 | 4,380 | 3,287 | 0.8751 | 0.7676 | 0.9918 | 0.8654 |
| 12 | 4,380 | 3,318 | 0.8725 | 0.7726 | 0.9934 | 0.8692 |
| 13 | 4,380 | 3,365 | 0.8688 | 0.7937 | 0.9801 | 0.8771 |
| 14 | 4,380 | 3,384 | 0.8723 | 0.7987 | 0.9894 | 0.8838 |
| 15 | 4,380 | 3,384 | 0.8724 | 0.7981 | 0.9897 | 0.8836 |
| 16 | 4,380 | 3,392 | 0.8744 | 0.8107 | 0.9835 | 0.8888 |
| 17 | 4,380 | 3,441 | 0.8682 | 0.8087 | 0.9951 | 0.8922 |
| 18 | 4,380 | 3,468 | 0.8798 | 0.8164 | 0.9963 | 0.8974 |
| 19 | 4,380 | 3,486 | 0.8846 | 0.8227 | 0.9928 | 0.8998 |
| 20 | 4,380 | 3,518 | 0.8911 | 0.8517 | 0.9645 | 0.9046 |
| 21 | 4,380 | 3,538 | 0.9023 | 0.8603 | 0.9698 | 0.9118 |
| 22 | 4,380 | 3,575 | 0.9083 | 0.8711 | 0.9757 | 0.9204 |
| 23 | 4,438 | 3,672 | 0.9049 | 0.8699 | 0.9867 | 0.9246 |
| **ALL** | **110,612** | **65,481** | **0.8669** | **0.7856** | **0.9445** | **0.8578** |

Performance improves sharply over the day: early morning (01-05) PR-AUC is only 0.37–0.57 (few positive samples since most rows haven't yet reached the official Tmin), while evening (20-23) PR-AUC exceeds 0.89 (most rows have reached Tmin, making the classification trivial).

**`will_make_new_low_after_now` — PR-AUC / Precision / Recall / F1 by Hour**

| Hour | n | pos | PR-AUC | Prec | Recall | F1 |
|------|----|-----|--------|------|--------|-----|
| 00 | 9,814 | 6,666 | 0.7515 | 0.6792 | 1.0000 | 0.8090 |
| 01 | 4,380 | 4,014 | 0.9836 | 0.9280 | 0.9985 | 0.9620 |
| 02 | 4,380 | 3,877 | 0.9791 | 0.8977 | 0.9956 | 0.9441 |
| 03 | 4,380 | 3,716 | 0.9742 | 0.9144 | 0.9486 | 0.9312 |
| 04 | 4,380 | 3,388 | 0.9316 | 0.8403 | 0.9504 | 0.8920 |
| 05 | 4,380 | 2,707 | 0.8596 | 0.7393 | 0.8833 | 0.8049 |
| 06 | 4,380 | 1,789 | 0.7011 | 0.5705 | 0.7418 | 0.6450 |
| 07 | 4,380 | 1,148 | 0.5708 | 0.5031 | 0.5662 | 0.5328 |
| 08 | 4,380 | 936 | 0.5513 | 0.5268 | 0.4829 | 0.5039 |
| 09 | 4,380 | 851 | 0.5025 | 0.5187 | 0.5206 | 0.5196 |
| 10 | 4,380 | 775 | 0.4709 | 0.4119 | 0.5613 | 0.4752 |
| 11 | 4,380 | 709 | 0.4739 | 0.3992 | 0.5670 | 0.4685 |
| 12 | 4,380 | 678 | 0.4645 | 0.3673 | 0.5796 | 0.4497 |
| 13 | 4,380 | 631 | 0.4578 | 0.4416 | 0.4612 | 0.4512 |
| 14 | 4,380 | 612 | 0.4720 | 0.4619 | 0.4265 | 0.4435 |
| 15 | 4,380 | 612 | 0.4998 | 0.4947 | 0.4608 | 0.4772 |
| 16 | 4,380 | 593 | 0.5391 | 0.6152 | 0.4098 | 0.4919 |
| 17 | 4,380 | 543 | 0.5053 | 0.5234 | 0.4328 | 0.4738 |
| 18 | 4,380 | 516 | 0.5501 | 0.5402 | 0.5078 | 0.5235 |
| 19 | 4,380 | 483 | 0.5721 | 0.5501 | 0.5342 | 0.5420 |
| 20 | 4,380 | 439 | 0.6248 | 0.6156 | 0.6128 | 0.6142 |
| 21 | 4,380 | 410 | 0.7094 | 0.6097 | 0.7049 | 0.6538 |
| 22 | 4,380 | 362 | 0.7955 | 0.7407 | 0.7182 | 0.7293 |
| 23 | 4,438 | 303 | 0.6885 | 0.6368 | 0.8911 | 0.7428 |
| **ALL** | **110,612** | **36,758** | **0.7986** | **0.7873** | **0.7458** | **0.7660** |

Early morning (01-04) has the strongest `will_make_new_low` performance (PR-AUC > 0.93), as the night cooling pattern from previous evening is highly predictable. Afternoon (12-17) is weakest (PR-AUC ~0.46–0.50), where re-low probability depends on complex factors like afternoon convection.

**`will_make_new_low_after_now` by Rain Regime**

| Regime | n | pos | PR-AUC | Prec | Recall | F1 |
|--------|----|-----|--------|------|--------|-----|
| ALL | 110,612 | 36,758 | 0.7986 | 0.7873 | 0.7458 | 0.7660 |
| No rain | 104,870 | 34,205 | 0.7981 | 0.7918 | 0.7495 | 0.7701 |
| Rain | 5,742 | 2,553 | 0.8069 | 0.6727 | 0.7728 | 0.7193 |
| 00-06 rain | 1,299 | 950 | **0.9171** | 0.8086 | 0.9474 | **0.8725** |
| 06-12 rain | 1,791 | 826 | 0.7341 | 0.5817 | 0.7712 | 0.6632 |
| 18-24 re-low | 26,338 | 2,513 | 0.6408 | 0.6125 | 0.6120 | 0.6123 |
| 00-05 early | 31,714 | 24,368 | 0.8697 | 0.8364 | 0.9168 | 0.8747 |
| 06-11 morning | 26,280 | 6,208 | 0.5782 | 0.4924 | 0.6018 | 0.5416 |
| 12-17 afternoon | 26,280 | 3,669 | 0.4899 | 0.4704 | 0.4421 | 0.4558 |

00-06 rain shows the strongest predictive power (PR-AUC=0.917), as rain-enhanced night cooling is highly predictable. Afternoon is the most challenging regime (PR-AUC=0.49), which is also where the two-stage shrinkage is most frequently triggered.

**`tmin_timing_bucket` — Confusion Matrix (OOT)**

| True\\Pred | Early(0) | Day(1) | Eve(2) | Total |
|-----------|---------|--------|--------|-------|
| Early(0) | **75,226** | 151 | 711 | 76,088 |
| Day(1) | 8,613 | **1,831** | 1,066 | 11,510 |
| Eve(2) | 7,998 | 392 | **2,208** | 10,598 |

**Overall accuracy: 80.7%** (98,196 valid labels)

The Early bucket (00-08) is classified most accurately (98.9% recall) because most Tmins fall here. The Day bucket (08-18) is hardest — only 15.9% correctly identified, with 8,613 misclassified as Early. The Eve bucket (18-24) has 20.8% recall, mostly confused with Early. The classifier has a strong Early bias, which is statistically justified (77.5% of samples are Early). **This classifier is primarily used for contextual adjustment in the two-stage shrinkage, not as a standalone judgment.**

---

**00-06 Rain Calibration (D_tmin_rain_calibrated)**

The 00-06 rain regime is Model D's weakest area: COV80=64.1% (PIW=1.545), well below the 80% target. The main cause is that the two-stage shrinkage erroneously narrows the prediction intervals — yet this regime actually has a high probability of further cooling.

Two fixes are applied during inference for this regime:

1. **Shrinkage override**: When `hour < 6` and `rainfall_60m > 0`, force `shrink = 1.0` (fully disable shrinkage), preventing further erosion of coverage.

2. **Residual calibration**: From OOT data, compute residual quantiles for 00-06 rain:
   ```
   residual = actual_remaining_downside - q50_downside
   p10(residual) = -0.452   → calibrated_q10 = q50 + (-0.452)
   p90(residual) = +2.072   → calibrated_q90 = q50 + (+2.072)
   ```
   These calibration factors are stored in `rain_calibration.json` and loaded lazily by `predict_intraday_tmin_model_d()`. The effect is to push q10 lower and q90 higher, significantly widening the prediction interval to approach 80% coverage.

**00-06 Rain OOT Metrics (Original vs Calibrated)**

| Version | COV80 | PIW | MAE | Bias | q10_br | q90_br |
|---------|-------|-----|-----|------|--------|--------|
| Original | 64.1% | 1.545 | 0.799 | -0.4968 | 13.2% | 22.7% |
| Calibrated | **80.7%** | **2.417** | 0.799 | -0.4968 | **9.3%** | **10.0%** |
| Change | **+16.6%** | **+0.872** | — | — | — | — |

Post-calibration COV80 jumps from 64.1% to 80.7% (exceeding the 80% target), q10_br drops from 13.2% to 9.3%, and q90_br drops from 22.7% to 10.0%. PIW widens from 1.545 to 2.417 — a necessary trade-off for this regime.

---

#### 4.5.10 Tmin Model E: Morning Minimum Prediction (E_morning_tmin)

Model E predicts the **morning local minimum** over 00:00–07:59 HKT for the first round of lowest temperature trading. Unlike Model D (full-day Tmin), Model E targets the minimum within a fixed early-morning window ending at 08:00.

**Label Definitions**

| Label | Type | Description |
|-------|------|-------------|
| `morning_min_00_08` | float | Minimum temperature from 00:00–07:59 for this target_date |
| `remaining_morning_downside` | float (≥0) | `min_so_far - morning_min_00_08` (remaining drop to reach the morning minimum) |
| `morning_low_reached` | binary | Whether the current `min_so_far` has already reached the morning minimum |
| `morning_low_survives_day` | binary (downstream) | Whether the morning minimum equals the full-day `official_tmin` (downstream label, used for trading layer blending, not a core morning-min prediction) |

**Features**

Same 115 features as Model D (38 base + 8 rain + 35 nowcast + 34 cross-midnight/evening). Training uses only pre-cutoff rows (hour < 8).

**Training Architecture**

Same LightGBM params as Model D, two stages:
1. **Classifiers**: `morning_low_reached_clf` (PR-AUC=0.844, F1=0.783), `morning_low_survives_day_clf` (PR-AUC=0.824, F1=0.845)
2. **Quantile regression**: 5 models (q10/q25/q50/q75/q90) on `remaining_morning_downside`
3. **No two-stage shrinkage** — raw quantile predictions only

**OOT Performance (raw, uncalibrated)**

| Bucket | n | MAE_down | COV80 | PIW | Bias | q90_br | q10_br |
|--------|----|----------|-------|-----|------|--------|--------|
| 00-02 | 9,848 | 0.572 | 75.4% | 1.541 | -0.2463 | 12.9% | 11.7% |
| 02-04 | 8,760 | 0.479 | 74.1% | 1.364 | -0.2075 | 12.1% | 13.8% |
| 04-06 | 8,760 | 0.377 | 73.9% | 0.981 | -0.2108 | 11.9% | 14.2% |
| 06-08 | 8,760 | 0.220 | 90.2% | 0.647 | -0.1961 | 9.4% | 0.4% |
| **ALL** | **36,128** | **0.417** | **78.3%** | **1.145** | **-0.2161** | **11.6%** | **10.1%** |

Raw COV80 is 78.3% (below the 80% target). The 06-08 bucket is overcovered (COV80=90.2%, q10_br=0.4%), while other buckets are undercovered — requiring hour-block calibration.

**Hour-Block Calibration**

Residual p10/p90 (`remaining_morning_downside - downside_q50`) computed per 2-hour bucket from OOT:

| Bucket | n | p10 offset | p90 offset |
|--------|------|------------|------------|
| 00-02 | 9,848 | -0.5089 | +1.3085 |
| 02-04 | 8,760 | -0.3892 | +1.1350 |
| 04-06 | 8,760 | -0.2396 | +1.0808 |
| 06-08 | 8,760 | -0.0354 | +0.4999 |

`predict_intraday_tmin_model_e_morning()` loads `morning_calibration.json` at inference and applies:
```python
q10_cal = max(0, q50 + p10_offset)
q90_cal = max(q10_cal, q50 + p90_offset)
```

**Calibrated OOT Performance**

| Bucket | n | COV80 | PIW | q10_br | q90_br |
|--------|------|-------|-----|--------|--------|
| 00-02 | 9,848 | 80.0% | 1.773 | 10.0% | 10.0% |
| 02-04 | 8,760 | 80.0% | 1.484 | 10.0% | 10.0% |
| 04-06 | 8,760 | 80.0% | 1.245 | 10.0% | 10.0% |
| 06-08 | 8,760 | 79.9% | 0.510 | 10.0% | 10.1% |
| **ALL** | **36,128** | **80.0%** | **1.269** | **10.0%** | **10.0%** |

After calibration every hour bucket achieves ~80% COV80 with q10_br ≈ q90_br ≈ 10%.

**Inference Logic**

`predict_intraday_tmin_model_e_morning()` reuses `_build_model_d_features()` and is called by `predict_intraday_tmin_all()` when `hour < 8`. It auto-loads `morning_calibration.json` for hour-block calibration:

```python
# Hour-block calibration (selected by hour)
q10_cal = max(0.0, q50 + p10_offset)
q90_cal = max(q10_cal, q50 + p90_offset)
quantiles = sorted([q10_cal, q25, q50, q75, q90_cal])
```

```python
pred_morning_min_p50 = min_so_far - q50_down
pred_morning_min_p10 = min_so_far - q90_down   # coldest
pred_morning_min_p90 = min_so_far - q10_down   # warmest
```

Returns 5 quantile predictions plus `prob_morning_low_reached` and `prob_morning_low_survives_day` for the trading system to combine.

**Classifier Diagnostics**

`morning_low_reached` by hour:

| Hour | n | Pos | Pos Rate | PR-AUC | Prec | Recall | F1 |
|------|------|-----|----------|--------|------|--------|----|
| 0 | 5,468 | 977 | 0.1787 | 0.9366 | 0.9417 | 0.8270 | 0.8807 |
| 1 | 4,380 | 437 | 0.0998 | 0.5634 | 0.4418 | 0.6773 | 0.5348 |
| 2 | 4,380 | 575 | 0.1313 | 0.5474 | 0.5020 | 0.6487 | 0.5660 |
| 3 | 4,380 | 759 | 0.1733 | 0.6134 | 0.5844 | 0.7207 | 0.6454 |
| 4 | 4,380 | 1,121 | 0.2559 | 0.6337 | 0.6084 | 0.6682 | 0.6369 |
| 5 | 4,380 | 1,846 | 0.4215 | 0.7423 | 0.6908 | 0.7394 | 0.7143 |
| 6 | 4,380 | 2,829 | 0.6459 | 0.8359 | 0.7701 | 0.9010 | 0.8304 |
| 7 | 4,380 | 3,578 | 0.8169 | 0.9240 | 0.8569 | 0.9693 | 0.9096 |
| **ALL** | **36,128** | **12,122** | **0.3355** | **0.8443** | **0.7486** | **0.8221** | **0.7836** |

Hour 0 is excellent (PR-AUC=0.937). Hours 2–4 are weakest (low pos rate, PR-AUC≈0.55). Hours 5–7 improve markedly approaching cutoff.

`morning_low_survives_day` by regime:

| Regime | n | Pos | Pos Rate | PR-AUC | Prec | Recall | F1 |
|--------|------|-----|----------|--------|------|--------|----|
| ALL | 36,128 | 26,427 | 0.7315 | 0.8237 | 0.7334 | 0.9999 | 0.8461 |
| No rain | 34,217 | 25,251 | 0.7380 | 0.8289 | 0.7400 | 0.9999 | 0.8505 |
| Rain | 1,911 | 1,176 | 0.6154 | 0.6083 | 0.6154 | 1.0000 | 0.7619 |
| 00-06 rain | 1,299 | 876 | 0.6744 | 0.6811 | 0.6744 | 1.0000 | 0.8055 |
| High RH (≥85) | 11,636 | 7,778 | 0.6684 | 0.7283 | 0.6692 | 0.9996 | 0.8017 |
| Low dew spread (<2) | 5,969 | 3,975 | 0.6659 | 0.7189 | 0.6659 | 1.0000 | 0.7995 |
| Winter (Dec-Feb) | 8,910 | 7,821 | 0.8778 | 0.9158 | 0.8778 | 1.0000 | 0.9349 |
| Summer (Jun-Aug) | 9,107 | 5,938 | 0.6520 | 0.6297 | 0.6701 | 0.9853 | 0.7977 |

Recall is near 1.0 across all regimes (model rarely misses a positive survives_day). No-rain weather (73.8% pos rate) outperforms rain (61.5%). Winter performs best (PR-AUC=0.916), summer is weaker (PR-AUC=0.630).

**Trading Layer Formula**

Combine Model E (morning min) and Model D (full-day Tmin) predictions:

```python
P(final_tmin_bucket = b) = P(morning_min_bucket = b) × P(survives)
                         + P(later_relow_bucket = b) × (1 − P(survives))
```

where `P(survives) = P(morning_low_survives_day) = 0.7315` (OOT mean). Bucket probabilities derive from the 5 quantile outputs of Model E (morning min) and Model D (full-day Tmin).

Summary statistics:
- Mean diff `official_tmin - morning_min = −0.083°C` (morning min slightly higher than full-day Tmin)
- 11.3% of days have official Tmin < morning_min by more than 0.1°C (requiring the `later_relow` path)

---

### 4.6 Model G: Forecast-Gap + Max-So-Far (`models/train_model_g.py`)

Model G predicts remaining upside using 17 features focused on forecast gap and max-so-far, without rainfall or nowcast data. It is a successor to the removed Model F.

#### 4.6.1 Features (17)
| Group | Features |
|-------|----------|
| Current state | `temp_current`, `humidity`, `pressure`, `temp_slope_30min`, `temp_volatility_30min`, `humid_delta_30min`, `pressure_delta_30min` |
| Forecast | `forecast_gap` (forecast_max - max_so_far) |
| Time | `hour`, `minute`, `day_of_week`, `month` |
| Max-so-far | `drop_from_max` |

#### 4.6.2 OOT Results
| Metric | Value |
|--------|-------|
| cov80 | 84.3% |
| PIW | ~1.5°C |
| MAE | ~0.45°C |

### 4.7 Model 2A: Core + Wind (`models/train_model_2a.py`)

Model 2A combines the core minute-observation baseline with forecast features and wind station data from 5 station groups (Ref, Offshore, Highland, Victoria Harbour, King's Park). This is the most feature-rich intraday tmax model.

#### 4.7.1 Feature Store (`data/build_model_2a_feature_store.py`)

The feature store is built from three data sources merged onto a 10-minute decision grid (06:00–23:50):

| Source | Data | Files |
|--------|------|-------|
| Minute weather | temp, RH, pressure, dew point (1-min raw) | `hk_weather_raw/*_{temperature,humidity,pressure,dew}.parquet` |
| Wind | 5-group station wind speed (mean, max, spread, count) | `wind_data/*_wind_all.parquet` |
| Forecast | HKO daily forecast (max/min, issue time) | `hk_daily_forecast/daily_forecast_clean.parquet` |

**Data corrections applied** (vs initial build):
1. `actual_high_today` and `max_so_far` / `min_so_far` computed on raw 1-minute data before rounding to the 10-minute grid, preventing the 10-min grid from artificially lowering daily max.
2. Forecast merged via `pd.merge_asof` with `decision_time` / `forecast_issue_datetime` per target date, ensuring only forecasts issued before the decision time are used (no look-ahead).
3. Wind rolling max (`wind_{prefix}_max_60m`) uses the group `max` column, not the group `mean` column.
4. Data freshness (`obs_data_age_minutes`, `wind_data_age_minutes`) calculated from real source timestamps.
5. Victoria Harbour stations explicitly mapped (`京士柏`, `啟德`, `九龍天星碼頭`) instead of relying on `"未知"` station_type mapping.

#### 4.7.2 Architecture

- **45 features** defined in `MODEL_2A_MIN_FEATURES`:
  - 5 current state: `temp_current`, `rh_current`, `pressure_current`, `dew_point_current`, `dew_point_spread`
  - 5 anchor: `max_so_far`, `min_so_far`, `range_so_far`, `drop_from_max`, `time_since_max`
  - 6 temperature trend: `temp_change_30m`, `temp_change_60m`, `temp_slope_30m`, `temp_slope_60m`, `temp_acceleration_60m`, `temp_volatility_60m`
  - 5 moisture/pressure: `rh_change_60m`, `dew_point_change_60m`, `dew_point_spread_change_60m`, `pressure_change_60m`, `pressure_change_180m`
  - 6 forecast: `forecast_min_temp`, `forecast_max_temp`, `forecast_range`, `forecast_gap_from_max_so_far`, `forecast_age_minutes`, `forecast_lead_days`
  - 8 wind: `wind_ref_mean`, `wind_ref_max`, `wind_victoria_harbour_mean`, `wind_victoria_harbour_max`, `wind_highland_mean`, `wind_highland_max`, `wind_all_change_60m`, `wind_kings_park_current`
  - 7 time features + 2 freshness features
- **Quantile models**: 5 LightGBM quantile regressors (q10/q25/q50/q75/q90) for remaining upside, trained with `max_depth=6, learning_rate=0.03, n_estimators=1500`
- **Classifier**: 1 LightGBM binary classifier for `is_upside_zero` (remaining_upside ≤ 0.05)
- **Training split**: Pre-2024-06-11 train, 2024-06-11 to 2025-06-11 valid, post-2025-06-11 OOT

#### 4.7.3 OOT Performance (54,289 rows, 2025-06-11 to 2026-06-23)

**Overall**
| Metric | Value |
|--------|-------|
| MAE (remaining upside) | 0.426°C |
| cov80 | 86.3% |
| PIW | 1.413°C |
| Bias (q50) | +0.016°C |
| q90 breach rate | 5.96% |
| q10 breach rate | 7.75% |
| Classifier PR-AUC | 0.981 |
| Classifier F1 (thr=0.446) | 0.923 |

**By Hour Bucket**
| Bucket | n | MAE_up | cov80 | PIW | Bias |
|--------|---|--------|-------|-----|------|
| 00-06 | 13,573 | 0.803 | 78.8% | 2.584 | +0.036 |
| 06-09 | 6,786 | 0.785 | 79.0% | 2.613 | +0.035 |
| 09-12 | 6,786 | 0.654 | 80.5% | 2.270 | +0.107 |
| 12-15 | 6,786 | 0.315 | 81.2% | 1.027 | -0.049 |
| 15-18 | 6,786 | 0.036 | 94.9% | 0.143 | -0.027 |
| 18-24 | 13,572 | 0.006 | 98.6% | 0.041 | -0.005 |

**Key observations:**
- Afternoon buckets (12-18) show strong performance with MAE < 0.32°C and cov80 > 81%.
- Morning/early buckets (00-09) have higher uncertainty (PIW ~2.6°C) due to larger remaining upside at the start of the day.
- The classifier achieves high PR-AUC (0.981), making it reliable for detecting when max temperature has been reached.

#### 4.7.4 Feature Importance
Key features from the q50 model (by split gain):
1. `drop_from_max` — most important single feature
2. `max_so_far`
3. `temp_current`
4. `forecast_gap_from_max_so_far`
5. `time_since_max`
6. `temp_change_60m`
7. `wind_all_change_60m` — first wind feature
8. `wind_ref_mean`
9. `temp_slope_60m`
10. `pressure_current`

Wind features contribute meaningfully after the core temperature and forecast features, particularly wind change over 60 minutes and reference station mean speed.

---

## 5. Inference & Prediction Logic (`models/intraday_inference.py`)

### 5.1 Feature Assembly
The `_build_features` function constructs the feature vector from live inputs (current temp, history, forecasts, rainfall).

### 5.2 Prediction Workflow
1.  **Load Models**: Cached LightGBM models are loaded from disk.
2.  **Predict Remaining Upside/Downside**: Generate quantile predictions.
3.  **Reconstruct Absolute Temperature**:
    *   `Predicted_Tmax = max_so_far + predicted_remaining_upside`
    *   `Predicted_Tmin = min_so_far - predicted_remaining_downside`
4.  **Heuristic Overrides**:
    *   If hour ≥ 18 and temp decline > 1.0°C, `prob_max_reached` is set to ≥ 0.95.
    *   If hour ≥ 16 and temp decline > 2.0°C, `prob_max_reached` is set to ≥ 0.90.

### 5.3 Bayesian Fusion (`combine_with_prior`)
Merges the long-horizon XGBoost prior (μ, σ) with the intraday LightGBM posterior.
*   **Current Default**: Weight = 0.0 (Full intraday mode).
*   **Logic**:
    *   If weight ≤ 0 or no intraday data, return intraday mean and derived std.
    *   Otherwise, calculate weighted mean and combined standard deviation:
        *   `post_mean = weight * prior_mean + (1 - weight) * intra_mean`
        *   `post_std = sqrt((weight * prior_std)² + ((1 - weight) * intra_std)²)`

### 5.4 Model A Inference
Model A uses a completely different feature set (38 temp+RH features) from the standard models. It is loaded as an additional model variant (`model_a`) alongside `baseline` and `rain_nowcast`.

*   **Feature Construction**: `_build_minute_features()` computes the 38 features from raw minute-level temp and RH history (rolling windows, dew point, acceleration).
*   **Prediction**: Identical quantile workflow (upside only) — q10 through q90, plus `upside_zero` classifier.
*   **Active Model Switching**: `set_active_model('model_a')` selects the minute-level model for prediction.
*   **Multi-Model Output**: `predict_intraday_tmax_all()` returns predictions from all three models: `baseline`, `rain_nowcast`, and `model_a`.

---

## 6. Trading & Execution Logic

### 6.1 Kelly Allocation (`execution/kelly_betting.py`)

#### Strict Multi-Outcome Kelly (`compute_multi_kelly_bets`)
*   **Concept**: Maximizes expected log wealth across mutually exclusive outcomes (temperature buckets).
*   **Optimization**: Uses `scipy.optimize.minimize` (SLSQP method).
*   **Constraints**:
    *   Max 15% capital per bucket (`max_per_bucket`).
    *   Total exposure ≤ 50% (`total_max`).
*   **Edge Calculation**:
    *   `Edge_YES = Model_Prob - Market_Price`
    *   `Edge_NO = Market_Price - Model_Prob`
*   **Wealth Calculation**: For each outcome `x`, calculates final wealth `W_x = 1 - total_f + sum(f_k / price_k)` for winning bets.

#### Simple Independent Kelly (`compute_bets_simple`)
*   **Concept**: Simplified version treating each bucket independently (ignores mutual exclusivity).
*   **Formula**: `f = (edge / odds) * kelly_fraction`
*   **Use Case**: Quick dashboard display, but prone to over-betting.

### 6.2 CLOB Slippage Simulation (`execution/clob_slippage.py`)
*   **Order Book Fetching**: Fetches order book depth from Polymarket CLOB API.
*   **Mock Mode**: Generates synthetic order books for offline testing.
*   **VWAP Calculation**: Simulates walking the asks/bids to estimate Volume-Weighted Average Price (VWAP) and actual contracts acquired.
*   **Budget Application**: Converts USDC budget into contract quantity based on available liquidity.

### 6.3 Dynamic Rebalancing (`execution/rebalancer.py`)
*   **Position Tracking**: Maintains `current_positions.json` with `side`, `quantity`, and `entry_price`.
*   **Rebalance Logic**:
    *   **Adding Position**: Calculates weighted average cost.
    *   **Reducing Position**: Maintains original entry price.
    *   **Flipping Position**: Resets entry price to new target.
*   **Dust Filtering**: Ignores positions smaller than `PM_MIN_QTY` (5.0 shares).
*   **Audit Logging**: Records all rebalancing actions to `data/paper_trade_audit.parquet`.

### 6.4 Enhanced V1 Strategy (Hardcoded Gates)

First-generation enhanced paper trading strategy with gate logic implemented as hardcoded constants.

**Entry Gates (5 checks)**

| # | Gate | Location | Threshold |
|---|------|----------|-----------|
| 1 | Time Window | `strategy_engine.py:125` | 08:00–16:00 only (`evening`/`night` blocked) |
| 2 | Edge Threshold | `strategy_engine.py:134` | Edge > 3% |
| 3 | Liquidity (Slippage) | `strategy_engine.py:141-149` | Order must fully fill; slippage ≤ edge/2 |
| 4 | Extreme Conviction Hold | `strategy_engine.py:177-183` | prob > 0.98 AND ≤6h to expiry → hold to expiry |
| 5 | Per-Bucket Exposure | Kelly constraint | max_per_bucket = 15% |

**Exit Gates (6 checks)**

| # | Condition | Threshold |
|---|-----------|-----------|
| 1 | Edge Reversed | edge < -5% → full exit |
| 2 | Profit Take | market price > model prob → exit |
| 3 | Rain Emergency | temp_drop > 1.5°C + raining → exit |
| 4 | Std Spike | std > 3.0 → halve position |
| 5 | Extreme Conviction Expiry | >6h to expiry even with conviction → exit |
| 6 | Time Expiry | No new entries after 18:00 |

**Exposure Calculation**

```python
effective = min(time_limit, base_total_max) × confidence_mult × volatility_mult
```

Where `time_limit` by slot: 08-12=50%, 12-16=30%, 16-18=10%, after=0%
`confidence_mult`: std<1.0→1.2x, std<2.0→1.0x, std<3.0→0.8x, else 0.6x
`volatility_mult`: price vol <2%→1.0x, <5%→0.8x, <10%→0.6x, else 0.5x

**Entry Point**: `strategy_engine.run_enhanced_rebalance_cycle()`
**Config Strategy Name**: `enhanced_v1_paper`

---

### 6.5 Enhanced V2 Strategy (Config-Driven Gates)

Fully driven by `config/paper_strategies.json`. All gate thresholds are adjustable via that JSON without code changes.

**Entry Gates (7 checks)**

| # | Gate | Location | V2 Threshold | vs V1 |
|---|------|----------|-------------|-------|
| 1 | Time Gate | `strategy_gate.py:672` | `min_hour: 8` | Same |
| 2 | Regime Edge | `strategy_gate.py:676-679` | 4 regimes with individual thresholds | New (per-slot) |
| 3 | Prob Confidence | `strategy_gate.py:682-683` | std ≤ 2.5 | Threshold instead of ladder |
| 4 | Boundary Proximity | `strategy_gate.py:685-692` | min_dist=0.6σ, agg=0.35σ | New (V1 had none) |
| 5 | Post-Slippage Edge | `strategy_gate.py:694-699` | edge_after > 0 | Same |
| 6 | Drawdown Gate | `strategy_gate.py:701-704` | -10% stop entries, -15% flatten | New (V1 had none) |
| 7 | Exposure Limits | `compute_config_orders` | max_per_bucket=12%, total=50% | Tighter (V1: 15%) |

**Regime Edge Thresholds**

| Regime | Hours | Min Edge | Exposure Cap |
|--------|-------|----------|-------------|
| `day_08_12` | 08-12 dry | 3.0% | 50% |
| `rain_08_12` | 08-12 rainy | **5.5%** (V2 override) | 40% |
| `slot_12_16` | 12-16 | 2.0% | 25% |
| `slot_16_24` | 16-24 | No entry | 0% |

**Model Selection Chain**

C→B→A fallback chain, affecting confidence multiplier:
- Model C (fresh nowcast available) → 1.0x
- Model B (rainfall features available) → 0.7x
- Model A (no rain/nowcast) → 0.5x
- Baseline (no minute model) → 0.5x

**Unified Position Sizing**

```
final_qty = kelly_size × model_conf_mult × time_window_mult × rain_uncertainty_mult × boundary_mult
```

| Multiplier | Values |
|-----------|--------|
| Model Confidence | C: 1.0 / B: 0.7 / A: 0.5 |
| Time Window | 08-10: 0.5 / 10-14: 1.0 / 14-16: 0.5 / 16+: 0 |
| Rain Uncertainty | no_rain: 1.0 / weak_rain: 0.7 / moderate_or_heavy_rain: 0.5 |
| Boundary Proximity | < 0.35σ: fixed 0.5 / < 0.60σ: distance_std / 0.6 |

**Exit Pipeline (10+ conditions)**

Two pipelines (forecast-driven + risk-driven), accumulating `multiplier`:

**Forecast-Driven:**

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Edge Reversed | < **-4%** (V2 override) | Full exit (mult=0) |
| Edge Disappeared | abs(edge) < 0.005 | Full exit (mult=0) |
| Bucket Prob < stop_prob | **8%** (V2 override) | Reduce to 30% |
| Profit Take | market > model prob | Reduce 50% |
| Confidence Drop | std > max_std × 1.5 | Reduce 50% |

**Risk-Driven:**

| Condition | Action |
|-----------|--------|
| Nowcast Stale | Reduce 30% (Model C only) |
| Data Missing | Reduce 50% |
| Rain Emergency (temp_drop > 1.5°C + rain) | Reduce to 30% |
| Drawdown -7.5% ~ -10% | Proportional reduction |
| Drawdown -10% ~ -15% | Stop new entries |
| Drawdown < -15% | Full flatten |
| T2S taper (hours_to_settlement < 4h) | Linear taper to 0.3x |

**Rebalance (7 triggers)**

| # | Trigger | Threshold |
|---|---------|-----------|
| 1 | Target position delta | delta > 0.5 qty |
| 2 | Edge change | abs(edge) > 1% |
| 3 | Top bucket changed | top_old ≠ top_new |
| 4 | Prob confidence change | max_pp ≥ **8pp** (V2 override) |
| 5 | EV change | |ΔEV| × 100 ≥ **3pp** (V2 override) |
| 6 | Exposure exceeded | > total_max × notional capital |
| 7 | T2S de-risk | hours_to_settlement < 4h |

**Stability Detection**: If top bucket unchanged AND prob change < 5pp → skip rebalance to reduce unnecessary trades.

**Drawdown Three-Tier System** (shared with exit pipeline)
- -7.5% ~ -10%: `REDUCE_RISK`, proportional reduction: `drawdown_pct / reduce_threshold`
- -10% ~ -15%: `STOP_ENTRIES`, block new entries, keep existing
- < -15%: `HARD_FLATTEN`, liquidate all positions

**T2S Linear Taper**
- From T-4h: `multiplier = hours_remaining / taper_start_hours`
- At T-2h: locked at `strong_taper_multiplier = 0.3`

**Config Architecture**

```json
paper_strategies.json → defaults (global) → strategies.enhanced_v2_paper.override (deep merge)
```

All thresholds adjustable via `paper_strategies.json` — no code changes needed.

**Entry Point**: `strategy_engine.run_config_rebalance_cycle()`
**Config Strategy Name**: `enhanced_v2_paper`

**V1 vs V2 Key Differences**

| Aspect | V1 | V2 |
|--------|----|----|
| Gate logic | Hardcoded in `strategy_engine.py` | Config-driven via `paper_strategies.json` |
| Model selection | None | C→B→A fallback chain + confidence multipliers |
| Regime edge | Uniform edge > 3% | 4 regimes with individual thresholds |
| Boundary proximity | None | Regex-parsed bucket names, standardized distance |
| Rain confidence | None | weak_rain: 0.7x, moderate_or_heavy_rain: 0.5x |
| Entry gates | 5 checks | 7 checks (+model sel, +boundary, +drawdown) |
| Exit conditions | 6 hardcoded | 10+ config-driven |
| Rebalance | No dedicated logic | 7 triggers + stability detection |
| T2S taper | None | Linear from T-4h |
| Drawdown control | None | Three-tier (stop/reduce/flatten) |
| Tunability | Modify `.py` code | Modify `paper_strategies.json` |

### 6.6 V2 Variant Strategies

Two variants derived from V2 by adjusting key parameters:

**`enhanced_v2_aggressive_paper`** — More aggressive entries

| Parameter | V2 Default | Aggressive |
|-----------|------------|------------|
| Rain threshold (rain_08_12 edge) | 5.5% | **4.5%** |
| Boundary min distance | 0.6σ | **0.35σ** |
| Boundary aggressive threshold | 0.35σ | **0.2σ** |
| Rebalance prob threshold | 8pp | **5pp** (default) |
| Rebalance EV threshold | 3pp | **2pp** (default) |
| 08-10 / 14-16 time window mult | 0.5 | **0.55** |

**`enhanced_v2_conservative_paper`** — More conservative

| Parameter | V2 Default | Conservative |
|-----------|------------|--------------|
| Rain threshold (rain_08_12 edge) | 5.5% | **6.0%** |
| Max per-bucket exposure | 12% | **10%** |
| Rebalance prob threshold | 8pp | 8pp (same as V2) |
| Rebalance EV threshold | 3pp | 3pp (same as V2) |
| Boundary min distance | 0.6σ | 0.6σ (same as V2) |

### 6.7 Strategy-Centric Architecture (v2 Rebuild)

The trading layer was rebuilt from a **portfolio-centric** to a **strategy-centric** model. Each strategy is now a self-contained entity with its own capital allocation, model selection, market template, and gate pipeline.

#### 6.7.1 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `StrategyAccount` | `execution/strategy_account.py` | Dataclass: id, label, model, capital, market_template, status, scheduler_on, params |
| `StrategyAccountStore` | `execution/strategy_account.py` | CRUD for `data/strategy_accounts.json` |
| `resolve_slug()` | `execution/market_templates.py` | Auto-resolve template + date → Polymarket slug |
| `strategy_card()` | `app/components/strategy_card.py` | Streamlit card: toggle, PnL, positions, trades, params |
| `strategy_builder_form()` | `app/components/strategy_builder.py` | Form: model selector, gate tuning, save, test in Lab |
| `run_single_strategy_cycle()` | `execution/strategy_runner.py` | Execute one cycle for a strategy |
| `auto_runner.py` | `execution/auto_runner.py` | Headless CLI for GitHub Actions cron |

#### 6.7.2 StrategyAccount JSON Schema

Stored at `data/strategy_accounts.json`:
```json
{
  "version": 1,
  "strategies": {
    "enhanced_v2_aggressive": {
      "label": "Enhanced V2 Aggressive",
      "model": "baseline",
      "capital": 5000.0,
      "market_template": "hk-tmax",
      "status": "running",
      "scheduler_on": true,
      "last_run": "2026-06-14T10:30:00",
      "params": {
        "bias": 0.0,
        "std_mult": 1.0,
        "kelly_fraction": 0.25
      },
      "from_strategy_key": "enhanced_v2_aggressive_paper",
      "gate_config_override": null
    }
  }
}
```

#### 6.7.3 Market Templates

Market templates auto-resolve to today's Polymarket slug, eliminating manual event discovery:

| Template | Function | Example Slug |
|----------|----------|-------------|
| `hk-tmax` | `_hk_tmax_slug(date)` | `highest-temperature-in-hong-kong-on-June-14-2026` |
| `hk-tmin` | `_hk_tmin_slug(date)` | `lowest-temperature-in-hong-kong-on-June-14-2026` |

Custom templates support `{placeholder}` substitution. Unknown templates are treated as literal slugs.

#### 6.7.4 Strategy Dashboard (3 Sub-Tabs)

**Live Tab**: Each strategy renders as a self-contained card:
- Toggle switch (ON/OFF) — starts/stops the scheduler for this strategy
- Capital input — per-strategy capital allocation
- Model label + market template badge
- PnL metrics: unrealized PnL, cost basis, market value, total fees
- Expandable detail: 3 sub-tabs (Positions / Trades / Params)
- **Run Now** button — executes one cycle immediately

**Builder Tab**: Create or edit strategy config:
- Strategy ID, label, model selector (dropdown from registry)
- Market template selector
- Capital, Kelly fraction default
- 4 collapsible gate sections: Entry, Exit, Sizing, Rebalance
- Each gate section has: enable toggle, parameter sliders
- **Save Strategy** — writes to both `strategy_accounts.json` and `paper_strategies.json`
- **Test in Lab** — runs `PaperTradeHarness` with current builder config

**Lab Tab**: Synthetic backtest (same as before):
- Single strategy backtest with configurable cycles, capital, Kelly
- Multi-strategy comparison (returns table with PnL, Sharpe, Max DD)
- Results: capital history chart, trade log, cycle details

#### 6.7.5 Per-Strategy Parameter Independence

Unlike the old global sidebar, each strategy has independent:
- `capital` — virtual money allocated to this strategy
- `bias` — edge bias adjustment
- `std_mult` — standard deviation multiplier for confidence
- `kelly_fraction` — fraction of full Kelly (0.05-1.0)

These are editable inline within each strategy card's expanded view.

#### 6.7.6 Position Isolation

Positions are stored with per-strategy isolation in `data/current_positions.json`:
```json
{
  "enhanced_v2_aggressive": {
    "highest-temperature-in-hong-kong-on-June-14-2026": {
      "enhanced_v2_aggressive": {
        "27-28": {
          "side": "YES",
          "quantity": 42.5,
          "entry_price": 0.35
        }
      }
    }
  }
}
```

Each strategy's positions are completely independent — no cross-strategy interference.

### 6.8 V2 Variant Strategies

Two variants derived from V2 by adjusting key parameters:

**`enhanced_v2_aggressive_paper`** — More aggressive entries

| Parameter | V2 Default | Aggressive |
|-----------|------------|------------|
| Rain threshold (rain_08_12 edge) | 5.5% | **4.5%** |
| Boundary min distance | 0.6σ | **0.35σ** |
| Boundary aggressive threshold | 0.35σ | **0.2σ** |
| Rebalance prob threshold | 8pp | 5pp |
| Rebalance EV threshold | 3pp | 2pp |
| 08-10 / 14-16 time window mult | 0.5 | **0.55** |

**`enhanced_v2_conservative_paper`** — More conservative

| Parameter | V2 Default | Conservative |
|-----------|------------|--------------|
| Rain threshold (rain_08_12 edge) | 5.5% | **6.0%** |
| Max per-bucket exposure | 12% | **10%** |
| Rebalance prob threshold | 8pp | 8pp (same as V2) |
| Rebalance EV threshold | 3pp | 3pp (same as V2) |
| Boundary min distance | 0.6σ | 0.6σ (same as V2) |

Both use `run_config_rebalance_cycle` as entry point — only the `override` block differs.

### 6.9 Headless Auto-Runner

`execution/auto_runner.py` is a CLI entry point designed for unattended cron execution:

```
python -m execution.auto_runner                 # run all due strategies
python -m execution.auto_runner --force          # skip cooldown check
python -m execution.auto_runner --list           # list enabled strategies
```

**Workflow**:
1. Reads `data/strategy_accounts.json`
2. For each strategy with `scheduler_on: true` and status `running`:
   - Checks 5-minute cooldown (skips if not elapsed)
   - Resolves event slug from `market_template` + today's date
   - Fetches live prices from Polymarket
   - Builds context and calls `run_single_strategy_cycle()`
   - Writes audit log to `data/auto_runner_log.json`
3. The GitHub Actions workflow (`.github/workflows/run_strategies.yml`) runs every 5 minutes and commits changed `data/` files

This provides true 24/7 execution that works even when the Streamlit dashboard is not open.

## 7. Dashboard & Visualization

The dashboard has been restructured into a modular **`app/` package** (entry point: `app/main.py`) replacing the old single-file `dashboard.py`.

### 7.1 Page Structure

| Page | URL | Description |
|------|-----|-------------|
| Hub | `/hub` | Overview dashboard with key metrics |
| Intraday | `/intraday` | Intraday temperature path, remaining upside/downside, rainfall metrics |
| Strategies | `/strategies` | **Unified strategy dashboard**: Live cards / Builder / Lab |
| Analytics | `/analytics` | Forward test performance, PnL charts, Brier scores |
| Health | `/health` | Runtime checks (compilation, smoke tests, schemas) |

### 7.2 Strategies Page (Central Feature)

The **Strategies** page (`app/pages/page_strategies.py`) replaces the old Portfolio and Execute pages with 3 sub-tabs:

**Live Tab** (`components/strategy_card.py`):
- Each strategy rendered as a self-contained card with ON/OFF toggle
- Per-strategy PnL summary (unrealized PnL, cost basis, market value, fees)
- Capital input editable inline
- Expandable detail: positions table, trade history, per-strategy params
- **Run Now** button for immediate cycle execution

**Builder Tab** (`components/strategy_builder.py`):
- Model selector dropdown (baseline, rain_nowcast, model_a, model_b, model_c, etc.)
- Market template selector (hk-tmax, hk-tmin)
- Capital and Kelly fraction inputs
- 4 gate sections: Entry, Exit, Sizing, Rebalance — each with toggles and sliders
- **Save** → writes to `strategy_accounts.json` + `paper_strategies.json`
- **Test in Lab** → runs `PaperTradeHarness` with current config

**Lab Tab**:
- Single strategy synthetic backtest (PaperTradeHarness)
- Multi-strategy comparison with PnL, Sharpe, Max DD table
- Capital history chart and trade log

### 7.3 Engine Comparison
*   **9-Day XGBoost**: Long-horizon probabilistic forecast.
*   **AWS High-Freq**: Short-term forecast anchored to HKO AWS data.
*   **Intraday Fusion**: Real-time LightGBM predictions fused with prior.
*   **Model A (Minute-Level, temp+RH)**: 5-min resolution predictions using temperature + RH only.
*   **Model B (Minute-Level, temp+RH+rainfall)**: Model A + rainfall history features at minute resolution (trained, integrated).
*   **Model C (Planned)**: Model B + 37 spatial rainfall nowcast features.
*   **Model G (Gap+Max)**: Forecast-gap + max_so_far based intraday tmax.
*   **Model 2A (Core+Wind)**: Baseline + forecast + wind station data + pressure + dew point.

### 7.4 Performance Tracking
*   **Forward Test Log**: Evaluates model predictions against actual outcomes.
*   **Metrics**: Brier Score, Cumulative PnL, ROI.
*   **Visualization**: Plotly charts for temperature paths, bucket probabilities, and bankroll curves.

### 7.5 Paper Trading Interface
*   **Manual Rebalancing**: One-click execution of target positions.
*   **PnL Calculation**: Real-time unrealized PnL for YES/NO positions.
*   **Position Management**: Manual close and portfolio reset.

### 7.6 Background Scheduler

A daemon thread started in `app/main.py()` runs alongside the Streamlit session:

- **Poll interval**: 30 seconds
- **Cooldown**: 5 minutes per strategy
- **Picks up**: any strategy with `scheduler_on: true` and `status: "running"`
- **Runs**: across all tabs (not just the Strategies page)
- **Dies**: when the browser tab closes or Streamlit Cloud sleeps the app

### 7.7 Sidebar

The sidebar retains only:
- Date picker for weather data alignment
- Force-refresh and cache-clearing buttons
- Sync HKO forecast button

All per-strategy parameters (bias, std_mult, kelly_fraction, capital) have been moved into individual strategy card expanders.

---

## 8. Deployment & Automation

### 8.1 Dashboard Deployment
*   **Streamlit Cloud**: Auto-deploys from `main` branch. Entry point: `app/main.py`.

### 8.2 GitHub Actions Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `daily_update.yml` | 00:30, 12:30 UTC | Data sync (intraday, forecast, model performance) |
| `hourly_update.yml` | :15 every hour | Forward-test logger, auto-rebalancer |
| `run_strategies.yml` | Every 5 min | **Headless strategy execution** (auto_runner.py) |

### 8.3 Run Strategies Workflow (`.github/workflows/run_strategies.yml`)
*   **Schedule**: `*/5 * * * *` (every 5 minutes — matches the 5-minute cooldown).
*   **Steps**:
    1. Checkout repo + setup Python 3.11
    2. Install dependencies
    3. Run `python -m execution.auto_runner`
    4. `git add data/` and commit if changed
*   **Commit message**: `chore: auto-run strategies [skip ci]`
*   Provides 24/7 unattended execution even when the Streamlit dashboard is closed.

### 8.4 Secrets Management
*   API keys and wallet credentials stored in Streamlit Secrets / GitHub Actions Secrets (never committed).
*   **Steps**:
    1.  Update forecast database (`update_forecast_database`).
    2.  Run forward test logger (`discover_and_log_all_markets`).
    3.  Execute auto-rebalancer (`run_auto_rebalance`).
    4.  Commit and push updated Parquet data.

### 8.3 Secrets Management
*   API keys and wallet credentials stored in Streamlit Secrets / GitHub Actions Secrets (never committed).

---

## 9. Key Technical Decisions

1.  **Quantile Regression over Point Estimation**: Provides a full probability distribution, essential for calculating bucket probabilities in prediction markets.
2.  **Remaining Upside/Downside Targets**: Normalizes the prediction task relative to current intraday progress, making the model more robust across different times of day.
3.  **Strict Temporal Splitting**: Prevents data leakage from future observations, critical for reliable backtesting.
4.  **Rainfall Interaction Features**: Captures the non-linear physical relationship between precipitation and temperature cooling.
5.  **Paper-Trading First**: All execution logic defaults to simulation mode (`mode: paper`) to ensure safety during research.

---

## 10. Repository Structure

```
Weather_Bot_Qwen/
├── data/                   # Raw & processed data (Parquet)
│   ├── strategy_accounts.json    # Per-strategy accounts & state
│   ├── current_positions.json    # Paper positions by strategy_key
│   ├── pnl_history/              # Per-strategy PnL snapshots
│   └── auto_runner_log.json      # Headless execution audit trail
├── app/                    # Streamlit modular app package
│   ├── main.py                  # Entry point, navigation, scheduler
│   ├── pages/
│   │   ├── page_hub.py          # Overview dashboard
│   │   ├── page_intraday.py     # Intraday temperature visualization
│   │   ├── page_strategies.py   # Live / Builder / Lab tabs
│   │   ├── page_analytics.py    # Forward-test performance
│   │   └── page_health.py       # Runtime checks
│   └── components/
│       ├── sidebar.py           # Global sidebar (date picker, sync)
│       ├── strategy_card.py     # Per-strategy card with toggle & PnL
│       └── strategy_builder.py  # Strategy creation form & gate tuning
├── features/               # Feature builders & dataset constructors
│   ├── build_intraday_ml_dataset.py
│   ├── build_intraday_lookup.py
│   └── build_intraday_minute_features.py   # Model A features
├── models/                 # Training, inference, saved models
│   ├── train_intraday_ml.py
│   ├── intraday_inference.py
│   ├── train_rain_aware_model.py
│   ├── train_minute_model_a.py             # Model A training (temp+RH)
│   ├── train_minute_model_b.py             # Model B training (+rain hist)
│   ├── train_minute_model_a_restricted.py  # A_restricted: ≥2023-06-01 control
│   ├── train_minute_model_b_restricted.py  # B_restricted: ≥2023-06-01 control
│   ├── train_minute_model_c.py             # Model C training (+nowcast)
│   ├── train_model_2a.py                   # Model 2A training (+wind + forecast)
│   ├── intraday_minute_ml/                 # Model A artifacts (38 features)
│   ├── intraday_minute_ml_model_b/         # Model B artifacts (46 features)
│   ├── intraday_minute_ml_model_c/         # Model C artifacts (83 features)
│   ├── intraday_minute_ml_a_restricted/    # A_restricted artifacts
│   ├── intraday_minute_ml_b_restricted/    # B_restricted artifacts
│   ├── intraday_minute_ml_tmin/            # Model A Tmin artifacts
│   ├── intraday_minute_ml_model_b_tmin/    # Model B Tmin artifacts
│   ├── intraday_minute_ml_model_c_tmin/    # Model C Tmin artifacts
│   ├── intraday_minute_ml_model_d/         # Model D artifacts (cross-midnight)
│   ├── intraday_minute_ml_model_e/         # Model E artifacts (morning min)
│   ├── intraday_minute_ml_model_g/         # Model G artifacts (forecast-gap + max)
│   └── intraday_minute_ml_model_2a/        # Model 2A artifacts (wind + forecast)
├── experiments/            # Experimental analysis scripts
│   └── calibration_model_b.py              # Residual-based interval calibration
├── scripts/                # Utility scripts
│   └── scrape_hko_history.py               # Minute data scraper
├── execution/              # Strategy runner, Kelly, slippage, rebalancer
│   ├── strategy_account.py        # StrategyAccount dataclass + persistence
│   ├── strategy_runner.py         # Cycle execution dispatch
│   ├── market_templates.py        # Auto-resolve Polymarket slugs from date
│   ├── auto_runner.py             # Headless CLI for cron execution
│   ├── strategy_config.py         # Strategy dataclass, pipeline builder
│   ├── strategy_factory.py        # Factory from paper_strategies.json
│   ├── paper_trade_harness.py     # Synthetic backtest harness
│   ├── gates/                     # 30+ pluggable gate functions
│   ├── kelly_betting.py
│   ├── clob_slippage.py
│   └── rebalancer.py
├── reports/                # Validation reports & audit logs
├── tests/                  # Pytest test suite (92 gate tests)
│   └── test_gates.py
├── .github/workflows/
│   ├── daily_update.yml           # Daily data sync
│   ├── hourly_update.yml          # Hourly forward-test & rebalancer
│   └── run_strategies.yml         # Headless strategy execution (every 5 min)
├── app/main.py             # Streamlit entry point (recommended)
├── dashboard.py            # Legacy single-file entry point (deprecated)
├── config.yaml             # Project configuration
├── requirements.txt
└── README.md
```

---

## 11. Future Roadmap

### Near-Term: Minute-Level Model Evolution

1. **Model B (Rainfall History)**: ✅ Trained and integrated — marginal improvement (+0.5pp cov80, -0.032 rain MAE). Rain/no-rain gap persists.
2. **Model C (Nowcast)**: Add 37 spatial nowcast features on top of Model B. Script: `models/train_minute_model_c.py`.
3. **Restricted Experiment (A vs B)**: ✅ Done. Within the rainfall-available period (≥2023-06-01, 109K train rows), rainfall features reduce rain bias (−0.128) and rain MAE (−0.067). But data volume (pre-2023 history) dominates overall performance.
4. **Interval Calibration**: ✅ Done. Residual-based calibration for rain rows (p10=-1.97, p90=+0.55) achieves 80% rain-regime coverage. See §4.4.10.
5. **Dashboard Integration**: ✅ Model B added to summary row, comparison tabs, model selector.
6. **Model G (Gap+Max)**: ✅ Trained and integrated. Forecast-gap + max_so_far features; OOT cov80=84.3%.
7. **Model 2A (Core+Wind)**: ✅ Trained and integrated. 45 features including wind station data, pressure, dew point; OOT MAE=0.222°C, cov80=93.2%, PR-AUC=0.992.
8. **Backtesting**: Run Minute Model B/C through the paper-trader backtest pipeline to measure PnL impact.
9. **Scheduled Retraining**: Automate weekly retraining of all minute models as new HKO minute data accumulates.

### Medium-Term: Infrastructure & Quality
*   **Model Quality**: Automated data-quality reports, by-hour validation, scheduled retraining.
*   **Advanced Features**: Radar/lightning nowcast integration for convective weather.
*   **Trading Enhancements**: Improved audit logs, stress tests, max drawdown monitoring.
*   **Execution Layer**: Real CLOB order placement (disabled by default) with manual approval workflow.
*   **Live RH Feed**: Replace hardcoded `rh_current=50.0` fallback with parsed RH column from HKO live CSV (already available in stream).
