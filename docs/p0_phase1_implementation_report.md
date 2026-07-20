# P0 Pipeline Repairs — Phase 1 Implementation Report

實作日期：2026-07-19

## 1. Implementation summary

本階段只處理 correctness 與 lineage boundary，沒有改動 strategy policy、trained model artifacts、training data 或 legacy application/paper scheduler routing。

已完成：

- 修復 `features/model_2a_source_adapters.py` 的 pandas `Series << integer` quality-flag blocker。
- 建立 explicit `Model2AV1Adapter` / `Model2AV2Adapter` registry。
- 在 scoring 前加入 strict YAML/spec/artifact/feature-order/classifier/threshold validation。
- realtime entry point 改為必須明選 `v1` 或 `v2`；移除 artifact fallback。
- v2 使用現有 v2 canonical builder；v1 不會呼叫 shared v2 builder，並回傳清楚的 deprecated/unsupported result。
- 修復 `_update_missing_flags_from_canonical()` 的 missing-argument call。
- parity、data-quality、shadow utilities 改為 version-aware，並在結果/報告中寫入 version、spec path 與 artifact identity。
- 新增 14 個 Phase 1 focused tests。

輸入限制：要求中提到的 `temperature_pipeline_audit_2026-07-19.md` 不在 repository 或提供的 attachment directory；本次依存在的 `docs/model_2a_lineage.md`、`docs/p0_pipeline_repair_plan.md` 與 pasted request 執行，沒有讀取 `archive_training_data/`。

## 2. Files changed

- `features/model_2a_source_adapters.py`
- `features/model_2a_feature_builder.py`（只更新 v2 builder lineage 說明）
- `inference/model_2a_adapters.py`（新增 registry、adapter contract、strict validator）
- `inference/model_2a_realtime_inference.py`
- `monitoring/inference_parity_check_base.py`
- `monitoring/data_quality_checks_base.py`
- `monitoring/model_2a_inference_parity_check.py`
- `monitoring/model_2a_data_quality_checks.py`
- `monitoring/model_2a_daily_shadow_eval.py`
- `tests/test_model_2a_phase1.py`
- `docs/p0_phase1_implementation_report.md`

未改動使用者原有的 `data/export/2026-07-19.csv`、`data/strategy_accounts.json` 與 SQLite sidecar 狀態。完整測試曾使 `data/current_positions.json` 暫時被 smoke test 清空；已用 HEAD 原始內容精確恢復，最後 `git status` 不再列出該檔案。

## 3. Version-adapter architecture

`inference/model_2a_adapters.py` 是新的 explicit boundary：

| Contract | v1 | v2 |
|---|---|---|
| `model_version` / `feature_version` | `v1` / `v1` | `v2` / `v2` |
| Artifact directory | `models/intraday_minute_ml_model_2a/` | `models/intraday_minute_ml_model_2a_v2/` |
| Spec | `config/model_2a_feature_spec.yaml` | `config/model_2a_feature_spec_v2.yaml` |
| Feature builder | explicit unsupported/deprecated callable | `build_model_2a_features` |
| Wind semantics | `wind_highland_mean/max` | `wind_offshore_highland_mean/max` |
| Missingness policy | preserve existing numeric NaN behavior；quality flags only | same；v2 wind names only |
| Realtime support | no：原始 v1 highland source semantics 尚未安全重建 | yes |

每個 adapter 宣告 `model_version`、`feature_version`、`artifact_directory`、`feature_spec_path`、`feature_list_path`、`feature_builder`、ordered feature names、missingness policy、classifier metadata 與 threshold metadata。`get_model_2a_adapter()` 對 `None`、未知 version 或模糊的 `Model 2A` 請求直接 fail closed；沒有 v1/v2 fallback。

## 4. Strict lineage validation

`validate_model_2a_lineage()` 在 scoring 前檢查：

1. requested version 與 YAML `model_version`。
2. YAML `feature_version`。
3. spec 的 `feature_list_path` 與 adapter artifact directory identity。
4. JSON feature list 的型別、非空、duplicate 與 exact expected ordered lineage。
5. `upside_q10/q25/q50/q75/q90` 檔案存在，且每個 LightGBM booster 的 embedded feature names/order 與 JSON 完全相同。
6. `upside_zero.txt` classifier feature names/order 與 JSON 完全相同。
7. `best_threshold.json` 存在，含 numeric `upside_zero_threshold` 且位於 `[0, 1]`。
8. 不使用 YAML `feature_groups` 作為 scoring order；scoring authority 仍是 JSON feature list 與 booster order。

刻意的 v1-spec/v2-artifact 與 v2-spec/v1-artifact pairings 都在載入/預測前拒絕。

## 5. Confirmed v1 behaviour

- v1 adapter initialization 與 v1 artifact/spec strict validation 通過。
- v1 明確 inference 會先驗證其自身 lineage，然後回傳 `unsupported_model_version` 與 deprecation error。
- v1 不會使用 shared v2 builder，也不會把 `wind_offshore_highland_*` 自動改名成 `wind_highland_*`。
- v1 monitoring data-quality contract 可識別 v1；parity replay 會因沒有可證明保留原始 highland semantics 的 builder 而 fail closed。

## 6. Confirmed v2 behaviour

- v2 adapter initialization 與 v2 artifact/spec/booster/classifier/threshold validation 通過。
- synthetic end-to-end realtime inference 明確指定 `model_version="v2"` 後成功完成。
- result 會記錄 `model_version`、`feature_version`、`spec_path`、`feature_list_path`、`artifact_directory` 與 `artifact_identity`。
- v2 仍只使用 `wind_offshore_highland_mean/max`；本階段沒有修改 weather/wind model numeric values 或 model probability calibration。

## 7. Tests added and run

新增 `tests/test_model_2a_phase1.py`，覆蓋：

- canonical weather/wind quality flags 的 fresh、missing、anomalous/null-safe cases 與 `int64` bit values。
- v1/v2 adapter initialization。
- strict artifact/spec/booster parity。
- cross-version pairing rejection。
- missing/extra/duplicate/reordered feature rejection。
- `feature_groups` 不覆寫 JSON order。
- classifier/threshold metadata 與 quantile artifact availability。
- explicit v2 synthetic inference。
- explicit v1 supported-deprecation behaviour。
- realtime missing-flags call。
- monitoring version selection與 metadata。
- shadow log 的 missing/mixed version rejection。
- artifact manifest unchanged。

結果：

- `python -m pytest -q tests/test_model_2a_phase1.py`：**14 passed**；只有既有 pandas/numexpr warning。
- `ruff check`（本次 touched Python files）：**All checks passed**。
- `python -m pytest -q tests`：**110 passed, 1 failed, 8 errors**。失敗/錯誤集中於既有 `tests/test_paper_trader_connect.py`：Gamma API network 在 sandbox 被拒，以及 `Engine`/`engine` fixtures 未提供；沒有 Phase 1 test failure。
- `python -m pytest -q`：repository 的 `tools/dashboard_strategy_runner_smoke_test.py` 在 collection 時 `SystemExit(1)`；其 embedded smoke 結果為 **169/171**，兩個既有 assertion failure 是 `list_strategies` expected keys 與 missing registry default 5 accounts。

## 8. Artifact-integrity verification

本次沒有修改兩個 artifact directories；Phase 1 test 亦在 validation 前後比較 manifest。以下是 final manifest；before/after hashes 相同。

### v1

```text
best_threshold.json|46|65b0280a0bf9d985ebf15b9384e8a83e3cd9d0e30271609ba8ac6eacd8a76921
feature_list.json|1039|2593d64aab50cbae96cace1115c2efa25f1da4850ddf0ff5de4da860f6e1fa5f
upside_q10.txt|2729698|70d76e004d91625c48cb094e7e907f0e9d6d0242fb5274815a76ae4fe6401358
upside_q25.txt|761333|003ceadb294e6ba6e33bff9acded1cb26512142388232f11db8441bdf01fea6b
upside_q50.txt|744559|da453083a742f52fa0d3a3247960732d84deb32a4824de42348903a638036e08
upside_q75.txt|913977|c43a97f90f16a9982cc662311ec600aed99288bae4ccf260ce2117ce0988ffaa
upside_q90.txt|1325408|4ec4dc8351c450b69912273a122726ab65bb7c700443af870bb84e6e6a70427b
upside_zero.txt|1555736|29ce5b87e7989ebac808844bc3b086a485c9c39f7cc29daccc9989ff22f38b18
```

### v2

```text
best_threshold.json|45|d9513f47dc6d3aa8f9ff1ad75963b1e98a14a56367ab289a4cf753651b8775e5
feature_list.json|1057|3647c6e1539ad5afbc53ec2d077562fc0c1e3e489ae168084d14f08feb3ff831
upside_q10.txt|2759005|326766d755840f67a732ea97fee466acd31c6366059a443b01494a50045db0c8
upside_q25.txt|534668|fd6b44d81b58a40e02ab0616915db0595ce459d8db136b1730e7309d480ed076
upside_q50.txt|563105|3446a3355f54eb0c9554ebb53c5643323fbf1901747fc62c322d6f2bae0fb962
upside_q75.txt|970623|0c6ae187856ffb250845bc05806ce7398c6f9aba772f4d63cf6db69a83864bcf
upside_q90.txt|1386680|facf94eef48e1a3d5d9c6d3847bf7ff47ffde91aadcac95f0aed889728bd2283
upside_zero.txt|1498978|02848bd2ea0b009fe2247267f4914843f22b59245d85a501935c747306091070
```

另外：`git diff -- models` 無輸出；`data/model_2a_feature_store.parquet` 無本次 status；沒有 training/export runtime data 被本次 implementation 覆寫。

## 9. Remaining blockers for Phase 2

- Truthful weather-input status propagation：fresh/stale/missing/fallback/source timestamps 與 age 尚未改動 model numeric semantics。
- Paper CLOB execution：仍未改動 Gamma reference、depth walking、fees 或 partial-fill policy。
- Uncertainty-gate、strategy threshold、entry window、Kelly、max-lock 與任何 strategy policy：按 request 明確 deferred。
- v1 realtime：若要恢復，必須先重建並測試原始 `wind_highland_*` trained semantics；否則維持 explicit deprecation。
- Repository existing test blockers：paper trader network/fixtures 與 dashboard smoke 的兩個 pre-existing assertions 需要獨立處理。

## 10. Concise diff risk review

- Source adapter：使用乘法組合 bit values，避免 unsupported pandas Series shift；valid numeric inputs 的 quality bits 保持 1/2/4（weather）及 1/2（wind）。非 numeric temperature anomaly 會安全成為 `NaN`，這是 crash prevention，不是 model feature rename。
- Adapter/validator：增加 hard fail boundary，可能讓錯誤 pairing 在 startup/realtime 暴露；這是預期的 fail-closed 行為。validation 會載入全部 quantile/classifier boosters，增加新 realtime path 的 startup cost，但不改權重。
- Realtime：未提供 explicit `model_version`、錯 spec 或錯 artifact 會拒絕；legacy app 沒有被自動切換到新 path。
- Monitoring：報告檔名按 v1/v2 分離並加入 lineage metadata；generic base API 只增加 optional metadata，不改其他 models 的預設行為。
