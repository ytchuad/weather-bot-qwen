# Model 2A Classifier Usage Note

## Overall Performance
- PR-AUC: 0.987 (06-23 OOT)
- F1: 0.934 (threshold 0.462)
- Positive rate: 53.5% across all OOT rows (06-23)

The classifier reliably detects when the daily max temperature has been reached (remaining_upside ≤ 0.05°C).

## 06-12 Limitation
- 06-12 PR-AUC: 0.498 (barely above random)
- 06-12 positive rate: 3.71%
- 06-12 F1: 0.525

Before noon, very few rows have reached the daily max, so positive cases are rare and the classifier has insufficient signal. Even when it predicts "max reached" in the morning, the prediction is unreliable.

## Recommended Usage
- **Primary signal window**: 12:00 onwards (PR-AUC improves to 0.758 by 12-15)
- **Reliable signal window**: 15:00 onwards (PR-AUC 0.986 for 15-18, 0.9996 for 18-24)
- **Do not use** as a primary early-morning (06-12) decision signal

## Implementation
When using `zero_pred` or `zero_proba` in downstream logic (dashboard display, threshold gates):

```python
df["classifier_reliable_window"] = (df["hour"] >= 12).astype(int)
```

Or for stricter usage:

```python
df["classifier_reliable_window"] = (df["hour"] >= 15).astype(int)
```

Flag rows where classifier output should be treated as actionable.
