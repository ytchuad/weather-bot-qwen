# Model 2A v2 High Drop-from-Max Evidence

## Context
Earlier versions of the model suffered from a "rebound" error: after the daily max was reached and temperature dropped by ≥5°C, the model still predicted a positive remaining upside, implying a rebound back to the day's earlier high.

## Current Diagnostic (OOT 06-23)
| Subset | n | MAE_up | Bias | cov80 | Mean Pred | Mean Actual |
|--------|---|--------|------|-------|-----------|-------------|
| drop>=2 | 10,414 | 0.013 | -0.004 | 0.993 | 0.013 | 0.017 |
| drop>=3 | 5,091 | 0.009 | -0.004 | 0.996 | 0.009 | 0.013 |
| drop>=5 | 357 | 0.001 | +0.001 | 0.997 | 0.001 | 0.000 |
| Max reached, drop>=5 | 357 | — | — | — | 0.001 | 0.000 |

## Confirmation
For the `drop_from_max >= 5°C` subset:
- **Mean predicted upside_q50**: 0.0006°C (near zero)
- **Mean actual remaining_upside**: 0.0000°C (truly zero)
- **is_upside_zero rate**: 100.0%

## Interpretation
The model no longer predicts a temperature rebound after the daily max has been reached and a large drop has occurred. It correctly predicts incremental upside *beyond* `max_so_far` rather than a rebound from the current lower temperature back to the earlier day high.

This confirms that the feature-store fix (recomputing `actual_high_today` and `max_so_far` per `target_date`) resolved the carry-over bug that previously fed stale previous-day highs into current-day decisions.
