# Model 2B Training Notes

## Overview
Model 2B is the observed-rainfall extension of Model 2A v2. It reuses all 45 Model 2A v2 features and adds 9 observed rainfall features.

## Model Architecture
- **Family**: LightGBM
- **Quantiles**: 5 regressors (q10, q25, q50, q75, q90) for remaining_upside
- **Classifier**: 1 binary classifier for is_upside_zero
- **Hyperparameters**: max_depth=6, num_leaves=31, lr=0.03, n_est=1500, min_data=500, reg_lambda=2.0 (identical to Model 2A v2)

## Feature List
- **Base**: 45 Model 2A v2 features (unchanged)
- **Added rainfall features**:
  - `rainfall_60m` — accumulated rainfall over last 60 minutes
  - `rainfall_120m` — accumulated rainfall over last 120 minutes
  - `has_recent_rainfall_obs` — 1 if rainfall_60m > 0 or rainfall_120m > 0
  - `rain_intensity_max_120m` — max 15-min interval rainfall over last 120 minutes
  - `rain_cooling_60m` — rainfall_60m * max(-temp_change_60m, 0)
  - `rain_after_max_flag` — 1 if recent rain AND drop_from_max >= 0.5
  - `post_peak_rain_flag` — 1 if recent rain AND drop >= 0.5 AND 30 <= time_since_max <= 240
  - `rain_data_gap_flag` — 1 if no valid rainfall within 45 min tolerance
  - `rainfall_data_age_minutes` — decision_time - latest_valid_rain_timestamp

## Data Handling
- Observed rainfall from HKO 15-min data (King's Park station), starts 2023-06-01
- Point-in-time merge: rain_available_time <= decision_time
- rain_available_time = ceil_dt_10min(rain_timestamp) + 8min lag
- Pre-2023 rows have rain_data_gap_flag=1, rainfall features = 0
- Rainfall data gap tolerance: 45 minutes

## Variants
### Variant 1 | Model 2B full
- Trained on full Model 2A v2 history (2016-12-08 onwards)
- Pre-rain period retained with rain_data_gap_flag=1
- Preserves long-history strength

### Variant 2 | Model 2B restricted
- Trained on target_date >= 2023-06-01 only
- Isolates incremental value of observed rainfall features

## Temporal Split
- Train: target_date < 2024-06-11
- Validation: 2024-06-11 <= target_date < 2025-06-11
- OOT: target_date >= 2025-06-11
- All rows with same target_date kept in same split

## Prediction Formula
- pred_tmax_qXX = max_so_far + upside_qXX
- Never uses: pred_tmax_qXX = temp_current + upside_qXX

## Results Summary

### OOT Metrics (2025-06-11 onwards, 377 dates, 40,716 rows)

| Metric | 2A v2 | 2B full | 2B restricted |
|---|---|---|---|
| ALL MAE | 0.3086 | 0.3087 | 0.3299 |
| no_rain MAE | 0.3033 | 0.3033 | 0.3241 |
| recent_rain MAE | 0.4078 | 0.4106 | 0.4428 |
| recent_rain bias | -0.0820 | -0.0841 | -0.0762 |
| recent_rain q50_breach | 0.3614 | 0.3598 | 0.3407 |
| post_peak_rain MAE | 0.3945 | 0.3972 | 0.4396 |
| heavy_recent_rain MAE | 0.5077 | 0.5167 | 0.5197 |

### Acceptance Check
| Criterion | Target | Actual | Status |
|---|---|---|---|
| Rain subset MAE improvement | >= +0.05 | -0.0028 | Not met |
| Rain subset bias reduction | >= +0.05 | -0.0021 | Not met |
| Rain subset q50 breach improvement | >= +2pp | +0.16pp | Not met |
| No-rain MAE degradation | <= +0.02 | +0.0000 | Met |

## Classification
**Diagnostic-only / not recommended for live deployment.**

Model 2B does not materially improve rainy-regime performance vs Model 2A v2. The added observed rainfall features do not provide additional signal beyond what the existing temperature, wind, and forecast features already capture. Possible reasons:

1. Rain events are sparse (only 2.3% of rows have `has_recent_rainfall_obs=1`)
2. Model 2A v2 already captures rain-induced cooling through temperature trends and drop_from_max
3. The single-station rainfall source (King's Park) may not represent conditions across all wind groups

Model 2B does not materially harm no-rain or all-data performance.

## Deliverables
- `data/model_2b_feature_store.parquet`
- `models/intraday_minute_ai_model_2b/` (full variant artifacts)
- `models/intraday_minute_ai_model_2b_restricted/` (restricted variant)
- `reports/model_2b_vs_2a_v2_oot_summary.csv`
- `reports/model_2b_rain_regime_breakdown.csv`
- `reports/model_2b_restricted_comparison.csv`
