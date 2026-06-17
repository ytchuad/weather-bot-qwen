# P0 Fixes Changelog

**Date:** 2026-06-07
**Scope:** Critical data pipeline, model training, and validation infrastructure fixes

---

## P0-1: Complete build_intraday_ml_dataset.py main() pipeline

**File:** `features/build_intraday_ml_dataset.py`

**Changes:**
- Added `add_targets()` function computing `upside`, `remaining_upside`, `remaining_downside`, `is_upside_zero`, `is_downside_zero` with correct threshold (≤0.05)
- Added `validate_required_columns()` function checking 24 required columns and no NaN in targets
- Restructured `main()` into 13 explicit pipeline steps
- Fixed pipeline ordering: `date` column now created before `add_intraday_state_features()`
- Added `dropna(subset=['tmax', 'tmin'])` after daily target merge to handle 5,040 rows with missing daily data
- Added `drop(columns=...)` cleanup in `merge_forecast()` to remove intermediate columns
- Added `extra_cols` to `keep_cols` to preserve reference columns (`tmax`, `tmin`, `min_so_far`, etc.)

**Validation:** 26/26 acceptance checks pass. Output: 227,067 rows × 67 cols.

---

## P0-2: Fix add_intraday_state_features() return and completeness

**File:** `features/build_intraday_ml_dataset.py`

**Changes:**
- Rewrote `add_intraday_state_features()` with correct point-in-time `time_since_extreme` logic
- Old bug: used *final* day's extreme value → negative/NaN times during morning hours
- New: `_compute_minutes_since_extreme()` tracks when the *current* extreme (as of each row) was first reached
- Added `drop_from_max` and `rise_from_min` columns
- Added `rolling_std_30min`, `rolling_std_60min`, `rolling_std_120min` at all lag windows
- Refactored lag/rolling computations to use `lag_steps` dict loop
- Added `reset_index(drop=True)` for clean NumPy indexing

**Validation:** 26/26 acceptance checks pass.

---

## P0-3: Fix rainfall point-in-time merge and freshness calculation

**File:** `features/build_intraday_ml_dataset.py`

**Changes:**
- Renamed `rain_df.datetime` to `rainfall_datetime` before merge (preserves observation timestamp)
- Changed `merge_asof` to use `left_on='datetime'`, `right_on='rainfall_datetime'`
- Increased tolerance from 20min to 120min
- `rainfall_data_age_minutes` now correctly computed as `datetime - rainfall_datetime`
- Added `has_recent_rainfall_obs` flag (1 if age ≤ 20 min)
- `has_rainfall_data` now derived from `rainfall_datetime.notna()` (not `rainfall.notna()`)

**Validation:** 13/13 acceptance checks pass.

---

## P0-4: Do not confuse missing rainfall with confirmed no-rain

**File:** `features/build_intraday_ml_dataset.py`

**Changes:**
- Original rainfall columns (`rainfall_60m`, `rainfall_120m`, `rainfall_30m`) now **kept with NaN** for unmatched rows
- Added `_filled` variants: `rainfall_60m_filled`, `rainfall_120m_filled`, `rainfall_30m_filled` (NaN → 0)
- Added `_missing_flag` columns: `rainfall_60m_missing_flag`, etc. (1 = missing, 0 = observed)
- Added `rain_data_gap_flag` summary column
- `rainfall_data_age_minutes` filled with 999 (sentinel) for unmatched rows instead of 0
- Updated `add_rain_temperature_interactions()` to use `_filled` columns
- Updated `feature_cols` to use `_filled` + `_missing_flag` columns
- Updated `validate_required_columns()` with new columns
- Updated no-rainfall-file fallback to create all new columns

**Validation:** 20/20 acceptance checks pass. 74,006 rows (32.6%) have missing rainfall data.

---

## P0-5: Fix rain cooling feature sign convention

**Files:** `features/build_intraday_ml_dataset.py`, `models/intraday_inference.py`

**Changes:**
- Old bug: `rain_cooling_60m = rainfall_60m * temp_change_60min.clip(upper=0)` → negative values
- New: Compute `cooling_60m = (temp_lag - temp).clip(lower=0)` as positive magnitude
- Then gate: `rain_cooling_60m = (rainfall_60m_filled > 0) * cooling_60m` → always ≥ 0
- Added `rain_cooling_120m` column (120-minute window)
- Re-derived `rain_cooling_30m` with correct sign
- Fixed `intraday_inference.py` to match: `cooling_60m = max(temp_60min_ago - temp_now, 0)`

**Validation:** 12/12 acceptance checks pass. `rain_cooling_60m` range: 0.00–7.30.

---

## P0-6: Tighten post_peak_rain and morning_peak_then_rain flags

**Files:** `features/build_intraday_ml_dataset.py`, `models/intraday_inference.py`

**Changes:**
- `post_peak_rain_flag` now requires ALL of:
  - `rainfall_60m_filled > 5.0` (was `>= 2`)
  - `drop_from_max >= 0.5°C` (new)
  - `time_since_max_so_far` between 30–240 minutes (was just `> 0`)
- Result: 727 rows flagged (was 4,555) — 84% reduction
- `morning_peak_then_rain_flag` replaces `morning_peak_rain_flag`:
  - Same strict conditions + `hour.between(9, 14)`
  - Result: 288 rows
- `morning_peak_rain_flag` kept as backward-compatible alias

**Validation:** 17/17 acceptance checks pass.

---

## P0-7: Complete retrain_full_rain_model.py

**File:** `models/retrain_full_rain_model.py` (complete rewrite)

**Changes:**
- Added `validate_features()` — checks all 34 features exist in dataset
- Added `time_split()` with chronological split and minimum size checks (train ≥ 1000, valid ≥ 100)
- Trains 4 model groups: upside quantile, downside quantile, upside zero classifier, downside zero classifier
- Saves to timestamped candidate directory: `models/intraday_ml_rain_candidate/YYYYMMDD_HHMMSS/`
- Does NOT overwrite production models
- Saves 15 artifacts: 10 quantile models, 2 classifiers, feature list, metrics, config, report, importance plot
- Validation metrics: MAE, coverage_80, coverage_50, interval width, CRPS for quantiles; accuracy, precision, recall, AUC for classifiers
- Generates `candidate_model_report.json` with full metadata

**Validation:** 42/42 acceptance checks pass. Training: 157,237 / Valid: 52,552 / Test: 17,278.

---

## P0-8: Replace copy_rain_model.py with promote_model.py

**Files:**
- `models/copy_rain_model.py` → `models/copy_rain_model_DANGEROUS_DO_NOT_RUN.py` (quarantined)
- `models/promote_model.py` (new)

**Changes:**
- Old script quarantined with `RuntimeError` on execution
- New gated promotion script with 5 validation gates:
  1. Candidate directory exists
  2. All required artifacts present (15 files)
  3. Validation metrics meet thresholds (coverage_80 ≥ 0.70, MAE < 2.0, AUC ≥ 0.60)
  4. Candidate report status is `candidate_ready`
  5. Existing active model backed up to `models/intraday_ml/archive/YYYYMMDD_HHMMSS/`
- Promotion log: `models/intraday_ml/metadata/promotion_log.json`
- Supports `--force` flag to skip metric checks (not recommended)

---

## P0-9: Update leakage audit for rainfall point-in-time logic

**File:** `tools/audit_no_leakage.py` (new)

**Changes:**
- 7 audit checks:
  1. `rainfall_datetime <= datetime` for all matched rows
  2. `rainfall_data_age_minutes >= 0`
  3. `has_recent_rainfall_obs=1` only when `rainfall_datetime <= datetime`
  4. Forbidden rainfall feature names (daily_total_rainfall, rain_end_time, etc.)
  5. Unallowed rainfall features not in whitelist
  6. Target columns not in feature list
  7. Dataset integrity (required columns, no NaN in targets)
- Writes `reports/leakage_audit_report.json`

**Validation:** 6 pass, 1 warn (raw `rainfall` column not in whitelist — expected), 0 fail.

---

## P0-10: Update runtime smoke test for rainfall-aware path

**Files:** `tools/runtime_smoke_test.py` (rewrite), `data/hko_rainfall_sample.parquet` (new)

**Changes:**
- Added rainfall-aware checks:
  1. Rainfall sample data loads
  2. `build_rainfall_features.py` runs successfully
  3. Rainfall point-in-time columns in training data
  4. Rainfall feature existence (10 required columns)
  5. Inference with rainfall at 4 test times (03:00, 10:00, 15:00, 23:00)
  6. Inference result keys match expected schema
  7. Feature list consistency between active model and dataset
- Test times cover early morning, late morning, afternoon, late night
- Writes `reports/runtime_smoke_test_report.json` with all status fields

**Validation:** All smoke tests pass. Overall status: pass.

---

## Summary of All Files Changed

| File | Change |
|------|--------|
| `features/build_intraday_ml_dataset.py` | P0-1 through P0-6: Complete rewrite |
| `models/intraday_inference.py` | P0-5, P0-6: Fix sign convention, fix flag logic, fix drop_from_max ordering |
| `models/retrain_full_rain_model.py` | P0-7: Complete rewrite |
| `models/copy_rain_model.py` | P0-8: Updated feature list (kept for reference) |
| `models/copy_rain_model_DANGEROUS_DO_NOT_RUN.py` | P0-8: Quarantined old script |
| `models/promote_model.py` | P0-8: New gated promotion script |
| `models/train_rain_aware_model.py` | P0-4, P0-6: Updated to use `_filled` columns and new flag names |
| `tools/audit_no_leakage.py` | P0-9: New leakage audit script |
| `tools/runtime_smoke_test.py` | P0-10: Complete rewrite with rainfall checks |
| `data/hko_rainfall_sample.parquet` | P0-10: New sample data file |
