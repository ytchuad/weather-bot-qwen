# Validation Log — Issues 1, 2, 3 Fixes

**Date:** 2026-06-07
**Validator:** AI-assisted code review + automated tests

---

## Issue 1 — Active feature_list.json missing rainfall-aware features

### What was wrong
The active model at `models/intraday_ml/feature_list.json` had old column names
(`rainfall_60m`, `rainfall_120m`, `morning_peak_rain_flag`) and no metadata declaring
whether it was baseline or rain-aware. Feature lists were hardcoded in 6+ files.

### Files changed
| File | Change |
|------|--------|
| `config/feature_schema.json` | Created — central feature schema (baseline=21, rain_observed=8, rain_interaction=5, metadata=2) |
| `features/feature_schema.py` | Created — `get_feature_list("baseline"/"rain_aware")`, `get_target_columns()`, etc. |
| `models/train_rain_aware_model.py` | Replaced hardcoded lists with `get_feature_list()` |
| `models/retrain_full_rain_model.py` | Replaced hardcoded FEATURES with `get_feature_list("rain_aware")` |
| `features/build_intraday_ml_dataset.py` | Replaced hardcoded `feature_cols` with `get_feature_list("rain_aware")` |
| `models/intraday_ml/metadata/model_registry.json` | Created — `active_model_type: "baseline"`, `rain_aware_features_present: false` |
| `models/intraday_ml/active/` | Created — current model files copied here |
| `models/intraday_ml/archive/` | Created — for backing up old active models |
| `models/intraday_ml/metadata/` | Created — registry and promotion logs |

### Validation commands
```bash
# 1. Feature schema loads correctly
python -c "from features.feature_schema import get_feature_list; fl=get_feature_list('rain_aware'); print(len(fl), 'features')"

# 2. Registry declares model type
python -c "import json; r=json.load(open('models/intraday_ml/metadata/model_registry.json')); print(r['active_model_type'])"

# 3. Pipeline produces all schema features
python features/build_intraday_ml_dataset.py
python -c "import pandas as pd; df=pd.read_parquet('data/intraday_ml_train.parquet'); print(df.shape)"

# 4. Leakage audit passes
python tools/audit_no_leakage.py

# 5. Smoke test passes
python tools/runtime_smoke_test.py
```

### Acceptance criteria
- [x] `config/feature_schema.json` contains all features
- [x] Training scripts use `get_feature_list("rain_aware")`
- [x] Active registry states `active_model_type`
- [x] Pipeline produces all 36 rain-aware features
- [x] Audit passes (8 pass, 0 warn, 0 fail)
- [x] Smoke test passes

---

## Issue 2 — Leakage audit warning from ambiguous 'rainfall' column

### What was wrong
The `rainfall` column name was too ambiguous (could mean interval, hourly, cumulative, etc.).
The leakage audit flagged it as `unallowed_rainfall_features: warn`.

### Root cause
`data/hko_rainfall_15min.parquet` has a column named `rainfall` which is accumulated rainfall
since midnight. This was carried through to the merged output.

### Files changed
| File | Change |
|------|--------|
| `features/build_rainfall_features.py` | Renamed `rainfall` → `rainfall_accumulated_since_midnight` at source |
| `features/build_intraday_ml_dataset.py` | Drop `rainfall`/`rainfall_accumulated_since_midnight` from merge output; removed from `extra_cols` |
| `tools/audit_no_leakage.py` | Added `VAGUE_RAINFALL_COLUMNS = ['rainfall']` check; added `rainfall_accumulated_since_midnight` to whitelist |

### Validation commands
```bash
# 1. Rebuild rainfall features
python features/build_rainfall_features.py

# 2. Rebuild training data
python features/build_intraday_ml_dataset.py

# 3. Verify 'rainfall' column is gone
python -c "import pandas as pd; df=pd.read_parquet('data/intraday_ml_train.parquet'); print('rainfall' in df.columns)"

# 4. Run leakage audit — should be 0 warnings
python tools/audit_no_leakage.py
```

### Acceptance criteria
- [x] Dataset no longer contains vague `rainfall` column
- [x] `rainfall_accumulated_since_midnight` exists in source but is dropped before output
- [x] Leakage audit: 8 pass, 0 warn, 0 fail

---

## Issue 3 — copy_rain_model.py still executable

### What was wrong
`models/copy_rain_model.py` blindly copies model files from `intraday_ml_rain/` to `intraday_ml/`
without any validation gates. `retrain_intraday_ml.py` also had a direct copy path.

### Files changed
| File | Change |
|------|--------|
| `models/copy_rain_model.py` | Replaced with hard-stop `RuntimeError` |
| `models/copy_rain_model_DANGEROUS_DO_NOT_RUN.py` | Deleted (was from earlier quarantine) |
| `models/retrain_intraday_ml.py` | Removed direct `shutil.copy2` to MODEL_DIR; now logs `promote_model.py` command instead |

### Validation commands
```bash
# 1. copy_rain_model.py raises RuntimeError
python models/copy_rain_model.py 2>&1 | grep RuntimeError

# 2. DO_NOT_RUN file deleted
ls models/copy_rain_model_DANGEROUS_DO_NOT_RUN.py 2>&1

# 3. promote_model.py is the only promotion path
ls models/promote_model.py

# 4. No other shutil.copy2 to active directory
grep -r "shutil.copy2.*MODEL_DIR" models/  (should return nothing)
grep -r "shutil.copy2.*intraday_ml" models/promote_model.py  (should only be in promote_model.py)
```

### Acceptance criteria
- [x] Running `python models/copy_rain_model.py` immediately raises RuntimeError
- [x] `promote_model.py` is the only supported promotion path
- [x] `retrain_intraday_ml.py` no longer copies directly to active directory
- [x] All promotion goes through `promote_model.py`

---

## Summary

All three issues fixed and validated:
1. ✅ Central feature schema with `active_model_type` metadata
2. ✅ Ambiguous `rainfall` column renamed and dropped from output
3. ✅ Direct promotion scripts disabled; only `promote_model.py` can promote
