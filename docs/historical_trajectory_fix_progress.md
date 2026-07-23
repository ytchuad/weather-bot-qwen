# Historical Trajectory Timestamp Fix — Progress

## Session 1 — timestamp semantics and weather capture/schema

已完成：

- HKO naive wall-clock timestamps 一律按 `Asia/Hong_Kong` 解析，再以 UTC 寫入。
- `snapshot_timestamp` 保留作 backward-compatible alias，且固定等同 `observation_timestamp`。
- 新 weather records 額外寫入 `observation_timestamp`、`source_release_timestamp`（可空）、`first_seen_timestamp` 及 `capture_timestamp`。
- `weather_observation_id` 識別同一 observation minute；`weather_snapshot_id` 識別不可變的 observation version。相同 version 的重複 polling 不覆寫最早的 `first_seen_timestamp`；數值或 source release time 改變的 correction 會產生獨立 version，保留自己的 availability time。
- `canonical_cycle_link` 和獨立 weather collector 均把 HKO observation time 與 collector/model time 分離。

驗證：

```text
python -m pytest -q tests/test_weather_timestamp_semantics.py
4 passed in 1.21s
```

未做（留待後續 Session）：

- chart projection 不再 forward-fill raw actual observations。
- history/minute view 與 replay 依 `first_seen_timestamp <= decision_timestamp` 做 point-in-time availability selection。
- corrupt historical records 的 read-time rejection、quality metadata、跨 HKT 日期保護及 frontend HKT formatting。

## Session 2 — Historical Prediction Trajectory backend projection

已完成：

- `layer_a.minute_view` 將 raw actual-temperature 與 replay as-of weather 分開：`actual_temperature` 只在 `observation_timestamp` 對應的 row 輸出，絕不以 `capture_timestamp`、request time 或 model `decision_timestamp` 作 x-axis，亦不對每分鐘 forward-fill。
- replay／lineage weather selection 現要求 `first_seen_timestamp <= row timestamp`；同一 observation 的 correction 只會在其自身 availability time 後取代原值。
- trajectory model values 只在真實 `decision_timestamp` 的 row 輸出；沒有 synthetic per-minute model cycles。
- projection 一律用 `Asia/Hong_Kong` 日曆日期篩選：today 只到 current HKT time，future selected date 回傳空 rows。
- read-time 排除 `observation_timestamp > capture_timestamp + 5 minutes` 的 corrupt weather record；不改寫 Layer A 原始檔。
- `/api/history/minute` 與 `/api/charts/models-comparison` 回傳 diagnostics：`excluded_future_weather_records`、`duplicate_observation_versions`、`latest_weather_observation_timestamp`、`latest_weather_first_seen_timestamp`。
- chart row 額外保留 as-of weather lineage：`weather_data_through`、`weather_first_seen_timestamp`、`weather_age_seconds`、`weather_snapshot_id`；raw actual 另有 `actual_observation_timestamp` 與 `actual_first_seen_timestamp`。

驗證：

```text
python -m pytest -q tests/test_historical_trajectory_projection.py
6 passed in 1.11s

python -m pytest -q tests/test_layer_a.py tests/test_layer_a_market.py tests/test_layer_a_quality_contract.py tests/test_layer_a_weather_history.py tests/test_layer_a_runtime_debug.py tests/test_weather_timestamp_semantics.py tests/test_historical_trajectory_projection.py -k "not test_export_archive_and_date_filter and not test_replay_smoke_and_threshold_kelly_variants and not test_export_archive_contains_closed_market_partitions and not test_export_contains_weather_stream_and_admin_requires_token"
66 passed, 4 deselected, 1 warning in 2.93s
```

已知環境限制：完整上述 Layer A suite 的 4 個 export tests 需要 Python package `zstandard` 解讀既有 `.zstd` snapshot；目前 runtime 未安裝該 package，4 個測試均在 export read path 以 `ModuleNotFoundError: zstandard` 失敗，與 Session 2 projection diff 無關。

後續（不屬於 Session 2）：

- frontend 明確指定 `Asia/Hong_Kong` formatting 及 tooltip lineage 顯示。
