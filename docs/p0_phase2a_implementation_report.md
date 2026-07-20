# P0 Pipeline Repairs — Phase 2A Implementation Report

日期：2026-07-19  
狀態：完成本階段 scope；未修改 model artifact、training data、strategy threshold、Kelly、entry window 或 CLOB execution。

## 1. 執行範圍與判定依據

本階段以以下文件作為 source of truth：

- `reports/temperature_pipeline_audit_2026-07-19.md`
- `docs/model_2a_lineage.md`
- `docs/p0_pipeline_repair_plan.md`
- `docs/p0_phase1_implementation_report.md`

Audit 的關鍵觀察是：11:03 context 仍記錄 `wind_data_age_minutes=8`，legacy wind features 全為 zero；forecast 亦出現 `30.2 → 31.6 → 30.8` 的 revision sequence。這些數值不能再被單一的 freshness/default 欄位掩蓋。

本次採用兩條並行資料線：

1. `legacy_compatible` numeric path：保持 LightGBM vector 與既有 prediction input 不變。
2. `truthful` status path：獨立記錄 source、timestamps、age、missing/stale/fallback/error 與 forecast continuity diagnostics。

## 2. Status contract

新增 `features/input_status.py`，contract version 為 `phase2a.v1`。每個 input status 具備：

`value`、`source_timestamp`、`decision_timestamp`、`age_seconds`、`age_minutes`、`is_missing`、`is_stale`、`is_fallback`、`fallback_method`、`source_name`、`quality_flags`、`raw_status`。

Age 只由 source/issue timestamp 與 decision timestamp 計算；缺少 source timestamp 時 age 保持 `null`，不再補入 `8` 或其他猜測值。`jsonable()` 與 `serialize_status()` 確保 nested status 可安全寫入 context、parquet、CSV/SQLite payload。

已區分以下 fallback method：

- `previous_observation`
- `cached_api_result`
- `model_compat_zero`
- `climatological_default`
- `unavailable`

source fetch exception 另以 `raw_status=source_error` 和 `quality_flags=["source_error"]` 表示，不與正常 empty/unavailable 混為一談。

## 3. Wind data path

`app/services/weather_service.py` 的 wind path 現在從 raw fetch 到 status metadata 都保留 group-level provenance：

- `reference`
- `victoria_harbour`
- `offshore_highland`
- `kings_park`
- `aggregate_change`

每個 group 的 mean/max/current/change 都有 flat status 與 nested group status。真實觀測值 `0.0` 會是 `is_missing=false`；無 source 時的 compatibility zero 會是 `is_missing=true`、`is_fallback=true`、`fallback_method=model_compat_zero`。

當 API 回傳空值但有 cache，numeric value 保持 cache value，status 標示 `cached_api_result` 並保留原 source timestamp。沒有 cache 時才使用 compatibility zero。舊 wrapper 使用的 `wind_data_age_minutes=8` 已從 weather status path 移除。

Model 2A v1 的 `wind_highland_mean/max` 只記錄為 `model_compat_zero` 加 `v1_semantics_unavailable`；沒有把 v2 `wind_offshore_highland_*` 翻譯成 v1 欄位。v2 欄位名稱維持 `wind_offshore_highland_mean/max`。

## 4. Forecast diagnostics

`standardize_forecast()` 現在保留 `target_date` 與 `forecast_source`，使 revision provenance 不會在 canonical adapter 被丟失。

`build_forecast_input_status()` 會：

- 只選取 requested target date 且 as-of decision time 可用的 row；
- 記錄 `forecast_source`、issue time、target date、age、previous value、revision size；
- 保留完整 `revision_history`，因此 `30.2 → 31.6 → 30.8` 可被重建；
- 標示 `target_date_mismatch`、`unexpected_target_date_change`、`issue_time_regression`、`large_revision`、`source_switching`、`stale_forecast_reuse`、`missing_issue_timestamp`；
- 不對 forecast 做 smoothing、cap、reject 或 numeric correction。

HKO forecast fetch 也保留 `ModelTime` 作為 issue timestamp；若 request 失敗，status 會記錄 source error，而不是製造 forecast age。

## 5. Observation 與 buffer status

`get_intraday_state()` 與 `build_observation_buffer_status()` 現在覆蓋：

- current temperature、RH、pressure、dew point；
- `max_so_far`、`min_so_far`；
- 30/60/120-minute lag；
- rolling/trend/derived change；
- `obs_data_age_minutes`。

buffer status 會排除 decision timestamp 之後的 row。缺少 lag/history 會標示 `insufficient_history`/`unavailable`，而不是把 fallback 數值誤報成 observation。RH 的既有 numeric default `50.0` 保留，但明確標示 `climatological_default`。

Rain、pressure、HKO RHRREAD 也使用相同 status contract；cache、default、compatibility zero 與 source error 各自可辨識。

## 6. Model inference separation

`models/intraday_inference.py` 的 Model 2A v1/v2 function 新增 status-aware diagnostic logging，但不把 status nested dict 放入 LightGBM `X`。

每次帶入 status 時，raw prediction 同時保存：

- `_features`：diagnostic feature log；freshness 欄位使用 truthful age 或 `null`；
- `_numeric_features`：實際 legacy-compatible numeric feature log，保留既有 vector semantics。

因此在 compatibility path 中，`_numeric_features.wind_data_age_minutes` 仍可為既有的 `8`，但 `_features.wind_data_age_minutes` 不再把它冒充為實際觀測 age。兩者會在 context 中使用不同欄位保存。

## 7. Realtime inference 與 lineage

`inference/model_2a_realtime_inference.py` 會在 v2 canonical numeric builder 後建立獨立 `input_status`，prediction payload 追加：

- status contract/policy；
- weather/wind/forecast/observation status；
- decision timestamp；
- `feature_values` 與 `numeric_features`；
- model version、feature version、artifact identity、feature spec path。

`_write_inference_log()` 將 nested status 與 feature maps 序列化為 clean JSON string supplemental columns，不改變既有 model feature schema，也不把 diagnostics 混入 numeric vector。

## 8. Context、snapshot 與 legacy app path

以下 execution/context path 已接入 status metadata：

- `app/services/context_builder.py`
- `app/api/strategies.py` 的 legacy context builder
- `execution/auto_runner.py` snapshot context

context 會保留：

- `feature_metadata` 與 `input_status` envelope；
- `weather_input_status`、`wind_input_status`、`pressure_input_status`、`forecast_input_status`、`observation_buffer_status`、rain/nowcast status；
- `model_2a_features` / `model_2a_v2_features` diagnostic maps；
- `model_2a_numeric_features` / `model_2a_v2_numeric_features` numeric maps；
- v1/v2 model lineage 與 feature spec identity。

既有 snapshot logger 的 context JSON serialization 保持不變；本次新增的 status maps 在進入 logger 前已轉為 JSON-safe 結構。

## 9. Compatibility policy

目前 active policy 明確記錄為：

```text
numeric_policy = legacy_compatible
status_policy  = truthful
```

本階段沒有啟用任何 strategy policy、threshold、Kelly、entry-window、execution 或 model retraining 變更。status metadata 只增加可觀測性與 provenance，不會自動平滑、拒絕或修正 input。

## 10. Tests 與 verification

新增 `tests/test_model_2a_phase2a.py`，覆蓋：

- age 計算與缺 source timestamp；
- real zero、stale、cache、compatibility zero、unavailable；
- JSON-safe serialization；
- forecast revisions、target mismatch、issue regression、source switching、large revision、missing issue time；
- future observation 排除與 insufficient history；
- v2 wind naming 與 v1/v2 status separation；
- legacy numeric prediction invariance；
- wind cache/fallback/source provenance。

驗證結果：

- `python -m pytest -q tests/test_model_2a_phase1.py tests/test_model_2a_phase2a.py`：`27 passed`；
- `python -m pytest -q tests`：`124 passed`、`1 failed`、`8 errors`。剩餘失敗全部來自既有 `tests/test_paper_trader_connect.py`：Gamma API network 被 sandbox 拒絕，以及該測試檔缺少既有 `Engine`/`engine` fixtures；沒有 Phase 2A failure；
- `python -m pytest -q`：除上述測試外，repository collection 也會載入既有 `tools/dashboard_strategy_runner_smoke_test.py`，其內含 2 個既有 smoke assertion failures；
- `python -m py_compile`：本次變更的 Python modules 通過；
- `ruff check`：本次 Phase 2A 核心檔案通過。

Phase 1 的 artifact manifest test 仍通過，確認 v1/v2 model artifact 未被修改。Phase 2A 測試本身沒有新增 runtime parquet/data generation；repository 既有 smoke test 曾暫時改寫 `data/current_positions.json`，其內容已恢復。

## 11. 明確未做的事情

- 沒有讀取或修改 `archive_training_data/`；
- 沒有修改 `models/intraday_minute_ml_model_2a*/` artifact weights/spec outputs；
- 沒有修改 strategy config、threshold、Kelly、entry window；
- 沒有修改 CLOB order placement、paper-trading execution semantics 或 real-money path；
- 沒有 retrain、promote 或 push Hugging Face remote；
- 沒有刪除 runtime data。

## 12. 後續邊界

本階段只完成 status propagation 與 truthful diagnostics。下一階段若要改變 fallback numeric policy、策略決策、threshold、model retraining 或 execution 行為，必須另立 change scope，不能把 status contract 的新增欄位視為 numeric policy authorization。
