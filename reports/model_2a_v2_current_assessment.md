# Model 2A v2 Current Assessment

**Status**: Acceptable as the current Core + Wind baseline. Feature change: offshore_highland replaces highland.

## Summary
Model 2A v2 integrates minute observations, forecast features, and wind station data (4 station groups + selected raw stations) with 45 features. v2 replaces `wind_highland_mean/max` with `wind_offshore_highland_mean/max` (merged offshore + highland group). The model achieves:

| Metric | Value |
|--------|-------|
| OOT MAE (remaining upside) | 0.309°C |
| OOT cov80 | 89.0% |
| OOT PIW | 1.008°C |
| OOT q50 bias | +0.005°C |
| Classifier PR-AUC | 0.987 |
| Classifier F1 | 0.934 |

## Key Findings

### 1. 06-12 Regression Performance
The 06-12 bucket is the most challenging decision window:
- MAE up: 0.645-0.824°C
- cov80: 80.6-81.3%
- PIW: 2.2-2.7°C
- This is the key decision-window metric for intraday use.

### 2. 06-15 Interval Calibration
The 06-09, 09-12, and 12-15 buckets are already close to the 80% target:
- cov80 range: 80.6-81.3%
- No narrowing or widening required.

### 3. 15-24 Interval Calibration
Late-day intervals are over-covered (94-98%) and should be narrowed:
- Calibration factors learned on validation set only.
- Factors saved to `model_2a_v2_interval_calibration_factors.json`.
- Applied to OOT for reporting only; not applied to model output.

### 4. Classifier Usage
The `is_upside_zero` classifier is strong overall (PR-AUC 0.987) but should be used mainly after noon (12:00+), not as an early-morning signal (06-12 PR-AUC: 0.484).

### 5. High Drop-from-Max Diagnostic
The original rebound error has been fixed:
- drop>=5°C: mean predicted upside 0.001°C, actual 0.000°C
- is_upside_zero: 100.0% when max reached
- Confirms target anchor correction works.

### 6. Filtered Actual-High Validation
After excluding raw temperature anomalies (>40°C), the feature-store actual_high_today matches the raw daily high to within ±0.1-0.2°C for most days, confirming the target definition is correct.

## Remaining Work
1. Late-day interval calibration factors need to be applied in inference code.
2. Classifier reliable-window flag should be integrated into dashboard/strategy logic.
3. 06-12 MAE improvement remains the largest opportunity for next model iteration.
