# Model 2A v1/v2 lineage review

Reviewed 2026-07-19 after the v1/v2 correction brief. This document separates the two independently trained models. It does not retrain, replace model artifacts, change strategy thresholds, or change the live inference path.

## Executive summary

- Model 2A v1 and Model 2A v2 are separate trained artifacts with separate feature lists. The only feature-list difference is intentional: v1 uses `wind_highland_mean/max`; v2 uses the merged canonical `wind_offshore_highland_mean/max`.
- Both training scripts currently point to the same physical path, `data/model_2a_feature_store.parquet`, but the data-generation pipeline changed between the v1 artifact lineage and the v2 commit. The original store used separate offshore/highland groups; the v2 commit changed the source grouping and feature names. The current store is v2-shaped and no longer contains `wind_highland_mean`.
- The v1 and v2 artifact directories are internally consistent. Each `feature_list.json` exactly matches the feature names and order embedded in its LightGBM `upside_q50.txt` booster. No v2 artifact loaded with the v1 spec was found in the current code.
- The declared realtime module is still defective, but the precise defect is narrower than the previous audit stated: its default is the v1 spec while the shared builder now emits v2 names, so the default path fails its required-feature check. Its `MODEL_DIR` v2 constant is only a fallback because the loader first follows the spec’s `feature_list_path`. With an explicit v2 spec, the artifact, spec, and wind names align, but a later missing-argument call still raises `TypeError`.
- A separate shared canonical-path defect occurs even earlier: `features/model_2a_source_adapters.py` builds quality flags with a pandas `Series << int` expression, which raises `TypeError` in the current environment before the realtime builder runs. The missing-argument defect remains confirmed but is masked by this earlier failure.
- The legacy application path runs both models from the correct artifact directories. Its v1 branch receives zero for the old highland fields because the current live wind service emits only v2 `offshore_highland` names. That is a v1-only operational gap; it is not evidence that v2 wind mapping is wrong. V1 should either get a v1-specific source adapter that recreates its trained group semantics or be isolated/deprecated.
- Current paper accounts use Model A, Model B, and Model G, not Model 2A v1/v2. The app still computes both 2A variants as part of the all-model dashboard result, and the account model key can be changed to `model_2a` or `model_2a_v2`.

## Corrected lineage table

| Lineage field | Model 2A v1 | Model 2A v2 |
|---|---|---|
| Model name | `model_2a` / Model 2A Core+Wind | `model_2a_v2` / Model 2A v2 Offshore+Highland |
| Artifact directory | `models/intraday_minute_ml_model_2a/` | `models/intraday_minute_ml_model_2a_v2/` |
| Training script | `models/train_model_2a.py` | `models/train_model_2a_v2.py` |
| Training data path in script | `data/model_2a_feature_store.parquet` | `data/model_2a_feature_store.parquet` |
| Training-source lineage | Original pre-v2 builder: separate `offshore` and `highland` groups. The v1 artifact was introduced in commit `855c3ae`; the exact final artifact build invocation is not recorded. | v2 builder/store lineage introduced in commit `422bcdd`; station type `離岸及高地` maps to `offshore_highland`, replacing the former separate groups. |
| Feature specification | `config/model_2a_feature_spec.yaml` (`model_version: v1`, `feature_version: v1`) | `config/model_2a_feature_spec_v2.yaml` (`model_version: v2`, `feature_version: v2`) |
| Feature list | `models/intraday_minute_ml_model_2a/feature_list.json` | `models/intraday_minute_ml_model_2a_v2/feature_list.json` |
| Target | `remaining_upside = max(actual_high_today - max_so_far, 0)`; classifier label `is_upside_zero = remaining_upside <= 0.05` | Same target definition, trained separately from the v2-shaped feature store |
| Quantile models | `upside_q10`, `q25`, `q50`, `q75`, `q90` LightGBM regressors | Separate `upside_q10`, `q25`, `q50`, `q75`, `q90` LightGBM regressors |
| Classifier | `upside_zero.txt`; threshold in v1 `best_threshold.json` | `upside_zero.txt`; threshold in v2 `best_threshold.json` |
| Probability conversion | App path: quantile outputs → `combine_with_prior` → `compute_bucket_probs` → `models.inference.predict_bucket_probabilities`; physical max clipping and optional zero-probability mixture are applied there. | Same conversion path when selected by the app; the direct realtime module returns quantiles/`zero_prob` but does not itself convert to market buckets. |
| Offline inference/evaluation | `models/validate_model_2a.py`, `models/fix_model_2a.py`; `models/intraday_inference.py` legacy function | `models/fix_model_2a_v2.py`, `scripts/compare_oot_metrics.py`; `models/intraday_inference.py` v2 function |
| Declared realtime entry | `inference/model_2a_realtime_inference.py` only when explicitly supplied the v1 spec; current shared builder is not a v1 builder, so this is not a valid default v1 route | Same module with explicit `config/model_2a_feature_spec_v2.yaml`; intended v2/HF route, currently blocked first by the shared source-adapter failure and then by the missing-argument defect described below |
| Legacy application entry | `app/services/model_service.py` → `models.intraday_inference.predict_intraday_tmax_all` → `predict_intraday_tmax_model_2a` | Same app orchestration → `predict_intraday_tmax_model_2a_v2` |
| Paper-strategy entry | Configurable through `app/config.py` alias `model_2a_paper`; not selected by current accounts | Configurable through `app/config.py` alias `model_2a_v2_paper`; not selected by current accounts |
| Historical replay entry | `execution/ensemble/backtest_compare*.py` reads stored `context_json.model_probs["model_2a"]`; it does not reload v1 artifacts | Same replay reads stored `model_probs["model_2a_v2"]`; it does not reload v2 artifacts |
| Research/ensemble entry | Stored-probability comparisons include `model_2a`; v1 OOT validator is separate | `scripts/compare_oot_metrics.py` directly loads v2 artifacts; ensemble comparisons include `model_2a_v2` |
| Current status | Legacy comparison/dashboard output; app-loadable; not preferred | Preferred/current 2A implementation; app-loadable and used by v2-specific research/evaluation |

## Exact feature order

Both models have 45 features in the same order except positions 32–33. The difference is part of the trained semantics and must not be translated automatically.

```text
1  temp_current
2  rh_current
3  pressure_current
4  dew_point_current
5  dew_point_spread
6  max_so_far
7  min_so_far
8  range_so_far
9  drop_from_max
10 time_since_max
11 temp_change_30m
12 temp_change_60m
13 temp_slope_30m
14 temp_slope_60m
15 temp_acceleration_60m
16 temp_volatility_60m
17 rh_change_60m
18 dew_point_change_60m
19 dew_point_spread_change_60m
20 pressure_change_60m
21 pressure_change_180m
22 forecast_min_temp
23 forecast_max_temp
24 forecast_range
25 forecast_gap_from_max_so_far
26 forecast_age_minutes
27 forecast_lead_days
28 wind_ref_mean
29 wind_ref_max
30 wind_victoria_harbour_mean
31 wind_victoria_harbour_max
32 <v1: wind_highland_mean | v2: wind_offshore_highland_mean>
33 <v1: wind_highland_max  | v2: wind_offshore_highland_max>
34 wind_all_change_60m
35 wind_kings_park_current
36 minutes_since_midnight
37 month_sin
38 month_cos
39 day_sin
40 day_cos
41 is_morning
42 is_afternoon
43 is_evening
44 obs_data_age_minutes
45 wind_data_age_minutes
```

The installed artifacts were checked directly: each directory’s `feature_list.json` equals the feature-name list embedded in its LightGBM q50 booster. The v1/v2 distinction is therefore not an artifact/spec corruption.

Both YAML specs list `forecast_max_temp` before `forecast_min_temp` inside the reference-only `feature_groups`, while the JSON feature lists and boosters use `forecast_min_temp` before `forecast_max_temp`. No runtime consumer in the repository uses `feature_groups` as model order; this is a documentation-order inconsistency to normalize or explicitly test during strict validation, not a v1/v2 lineage difference.

## Runtime consumer matrix

| Runtime component | Configured/selected version | Artifact and spec actually used | Feature vector and ordering | Missing/unknown/mismatch behavior |
|---|---|---|---|---|
| Legacy app model loader | Loads both `model_2a` and `model_2a_v2` | `models/intraday_inference.py` constants select the matching directory and feature list for each key | v1 manual function emits `wind_highland_*`; v2 manual function emits `wind_offshore_highland_*`; the DataFrame is reindexed to the booster’s embedded feature names | Numeric `None` values are replaced with zeros; default age is 8. Extra feature-dict fields are ignored. There is no explicit model-version/schema assertion. A missing booster field generally causes the per-model call to fail and be logged as `None`. |
| Canonical feature-store build | v2-shaped current store | `data/build_model_2a_feature_store.py` | Current store has `wind_offshore_highland_*` and no `wind_highland_*`; `features/model_2a_feature_builder.py` also emits v2 names | As-of merges have no tolerance; training scripts fill listed feature nulls with zero. No v1 feature builder exists in the current tree. |
| Declared realtime/HF module | Default argument is v1 spec; explicit v2 is possible | Default: v1 spec path resolves to v1 artifact. Explicit v2 spec resolves to v2 artifact. `MODEL_DIR=v2` is fallback only | Shared builder emits v2 field names regardless of the selected spec | Current source-adapter flag construction raises before feature building. After that shared bug is fixed, default v1 would fail the required-feature check for `wind_highland_*`; it does not silently score a v2 artifact with a v1 spec. Explicit v2 has matching names but lacks a version/artifact assertion and then reaches the missing-argument defect. |
| Monitoring parity/quality/shadow utilities | Default to the v1 spec; parity utility uses the shared canonical builder; quality/shadow helpers are spec/log consumers rather than artifact loaders | `monitoring/model_2a_inference_parity_check.py` passes its selected spec into the shared v2-shaped builder; the other two utilities read the selected spec for checks/metrics | Parity has the same v1-default/v2-builder mismatch; all three depend on the shared source adapters when canonicalizing inputs | Current weather/wind adapter flag construction fails before parity; after that, default parity is not a valid v1 route. These utilities do not prove artifact identity unless version metadata is added. |
| Paper scheduler | Current accounts: `model_a`, `model_b`, `model_g`; 2A variants are computed but not selected | `execution/auto_runner.py`/`execution/strategy_runner.py` call `app.services.model_service.run_all_models` | Same legacy app vectors; selected model key picks the stored result | The selected result is required, but source defaults and fixed strategy metadata can hide missingness. If account config selects `model_2a` or `model_2a_v2`, the corresponding app result is used. |
| Strategy context builder | Account model key | `app/services/context_builder.py` uses the all-model app result | Stores model probabilities, feature metadata, and depth in `context_json` | Depth is recorded in `context_json`, but not passed as top-level execution input. `mock_slippage=True` and `model_std=1.5` are hardcoded. |
| Historical replay/backtest | Version is whatever generated the stored snapshot | `execution/ensemble/backtest_compare.py` and model comparison wrappers load CSV `context_json.model_probs` | No feature vector is rebuilt; stored bucket probabilities are consumed | No artifact/spec validation is possible in this replay. It is a strategy replay, not an inference lineage test. |
| OOT/research evaluation | v1 or v2 selected by script | v1 validator loads v1 directory; `scripts/compare_oot_metrics.py` loads v2 directory | Uses the feature list from the selected artifact and `.fillna(0)` | Direct artifact loading is explicit within each script; no cross-version translation is used. |
| Ensemble model path | Model labels from stored snapshots | `execution/ensemble/strategy.py` and `backtest_compare_models.py` | Consumes stored `model_probs`; it does not load v1/v2 weights | This path cannot silently use the wrong artifact because it does not load an artifact at all, but the snapshot producer/version is not independently asserted. |
| Notebook/research notebooks | None found in current repository | No `notebooks/` directory or notebook reference was found | Not applicable | Research is script/report based; lineage must be recorded in the scripts and snapshot metadata. |

## Wind-mapping reassessment

### What v1 trained

The original v1 builder in commit `855c3ae` had separate source groups for offshore and highland. The v1 training script’s 45-feature list retained only the `wind_highland_mean/max` pair in the final model schema, alongside reference, Victoria Harbour, aggregate change, and King's Park fields. The v1 booster and feature list both contain those old names.

### What v2 trained

Commit `422bcdd` intentionally changed the source grouping from separate offshore/highland groups to `offshore_highland`, changed the canonical builder names, changed the v2 training feature list, and wrote a separate v2 artifact directory. The v2 booster and feature list both contain `wind_offshore_highland_mean/max`. This is the intended v2 mapping.

### What the live paths do

- `app/services/weather_service.compute_wind_kwargs()` currently emits `wind_offshore_highland_*` and does not emit `wind_highland_*`.
- `app/services/model_service.py` passes both name families, but reads the v1 family from absent keys with a zero default. Therefore the legacy v1 function receives zero for its highland fields. This affects v1 only.
- The same `model_service.py` call passes the v2 `wind_offshore_highland_*` values to `predict_intraday_tmax_model_2a_v2`. No v2 name-mapping defect was found.
- The canonical `features/model_2a_feature_builder.py` is v2-shaped after the v2 commit. Its module comment still claims exact parity with the v1 feature list; that comment is stale and should be corrected as part of explicit versioning, not by renaming v2 fields.

Changing the v1 artifact’s feature names to v2 names would break v1 training/inference parity. If v1 must remain operational, it needs a v1-specific builder/source adapter that recreates the original highland definition and has its own schema test. If v1 is comparison-only, isolation or deprecation is safer than forcing it to resemble v2.

## Realtime inference reassessment

| Previous claim | Corrected classification | Evidence |
|---|---|---|
| The realtime module loads a v2 artifact with a v1 spec | Withdrawn as stated | `_load_model_2a()` first resolves `Path(spec["feature_list_path"]).parent`; the default v1 spec therefore selects the v1 artifact. `MODEL_DIR=v2` is only the missing-file fallback. |
| The realtime default is version-safe | Confirmed configuration/feature-builder bug after the shared adapter failure is repaired | The default spec is `config/model_2a_feature_spec.yaml` (v1), while the shared builder emits v2 names. Once source-adapter flag construction runs successfully, the default path reaches `missing_feats` and raises for `wind_highland_*`. |
| A missing-argument `TypeError` exists | Confirmed bug in the declared realtime entry point | `run_model_2a_inference()` calls `_update_missing_flags_from_canonical()` with three arguments, while the function signature requires two additional arguments. |
| Realtime source adapters always reach model loading | Revised: shared adapter bug masks later checks | `standardize_weather_obs()` and `standardize_wind_obs()` use pandas `Series << int` flag expressions; the current environment raises before feature construction. |
| Model version and artifact/spec pairing are fail-closed | Not implemented | `load_model_spec()` checks YAML keys, but `_load_model_2a()` does not assert spec version, artifact directory identity, feature-list equality, or booster feature-name equality. |

No direct in-repository call to `run_model_2a_inference()` beyond its definition was found, so current app/paper reachability is not proven. The monitoring parity utility is a separate consumer of the canonical adapters/builder, not a caller of this realtime function. The module is nevertheless the repository-declared realtime/HF entry point and is broken when invoked with its default. The minimal safe direction is an explicit versioned adapter, with v2 as the preferred default and a separate v1 adapter only if v1 is retained. No automatic cross-version feature translation should be introduced.

## Intentional differences versus confirmed defects

### Intentional

- Separate v1/v2 artifact directories.
- Separate feature specs and feature lists.
- `wind_highland_*` in v1 versus `wind_offshore_highland_*` in v2.
- Separate model-loading keys and separate prediction functions.
- Model 2A1 loading the v1 artifact with an i-Lens forecast override is an adjacent legacy variant, not evidence that v1 and v2 should be merged.

### Confirmed defects or shared risks

- Realtime module default selects a v1 spec against the current v2-shaped shared builder.
- Realtime missing-argument failure.
- Canonical realtime source adapters currently fail while constructing data-quality flags due to unsupported pandas Series bit shifting; this is shared and independent of model version.
- Monitoring parity defaults to the v1 spec while using the current v2-shaped shared builder; it has the same version-boundary defect and no artifact identity assertion.
- Missing wind is converted to numeric zero in the live service/app path; synthetic default age values are also used. This is a shared input-contract defect, not a v1/v2 rename defect.
- Current feature-store as-of merges have no stale-data tolerance and training nulls are filled with zero; this needs explicit data-status treatment before changing model semantics.
- Active paper execution uses Gamma/UI prices and `mock_slippage=True`; cached CLOB depth is recorded but not used by the legacy strategy path.
- Strategy context hardcodes `model_std=1.5` instead of exposing the selected model’s derived uncertainty.
- A/B/C morning overprediction remains an independent research problem.

## Operational recommendation

Make version boundaries explicit first. Prefer v2 for new realtime/shadow work. Keep v1 available only behind an explicit v1 adapter and schema validation, or mark it comparison-only. Do not change v2 wind logic, retrain either model, or alter strategy thresholds until each adapter can prove artifact/spec/feature parity and the replay can distinguish stored predictions from newly generated predictions.
