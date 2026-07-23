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

## Session 3 — point-in-time weather availability and replay

已完成：

- `select_weather_as_of` 成為共用 selector：weather version 必須同時滿足 `observation_timestamp <= decision_timestamp` 與 `first_seen_timestamp <= decision_timestamp`；同一 observation 的較晚 correction 只會在其本身 first-seen time 起可被選取。
- 每個新 canonical model-cycle record 現保存 `weather_snapshot_id`、`weather_data_through`、`weather_first_seen_timestamp`、`weather_age_seconds`，並以相同內容保存於 `weather_lineage`。timestamp 一律 canonical UTC。
- 若 canonical capture 遇到相同 immutable weather version，weather storage 會把已儲存版本（含原始 `first_seen_timestamp`）回傳給 lineage writer；不會以當次 polling 時間覆寫 lineage。
- `replay_model_cycle_minute_view` 會在 model `decision_timestamp` 先做完整 version-stream as-of selection，然後要求 stored `weather_snapshot_id` 與該 selection 完全一致；lineage metadata 也會交叉驗證。失配、尚未 first-seen 或不存在的 anchor 會標記 linkage failure，不會以較晚 correction 補值。
- replay minute rows 固定使用該 model cycle 的 immutable selected version；下一個 real model cycle 之前不會因新 correction 改變已儲存 prediction 的 weather lineage。
- 未變更 model cadence、market replay interval 或任何 trading strategy rule。

驗證：

```text
python -m pytest -q tests/test_weather_point_in_time_replay.py tests/test_weather_timestamp_semantics.py tests/test_historical_trajectory_projection.py tests/test_layer_a_weather_history.py -k "not test_export_contains_weather_stream_and_admin_requires_token"
29 passed, 1 deselected in 1.55s

python -m pytest -q tests/test_layer_a.py tests/test_layer_a_market.py tests/test_layer_a_quality_contract.py tests/test_layer_a_weather_history.py tests/test_layer_a_runtime_debug.py tests/test_weather_timestamp_semantics.py tests/test_historical_trajectory_projection.py tests/test_weather_point_in_time_replay.py -k "not test_export_archive_and_date_filter and not test_replay_smoke_and_threshold_kelly_variants and not test_export_archive_contains_closed_market_partitions and not test_export_contains_weather_stream_and_admin_requires_token"
69 passed, 4 deselected, 1 warning in 3.04s
```

新增 focused cases：

- `09:40` observation 在 `09:48` 才 first seen：`09:47` replay 不可用，`09:48` 可用。
- `09:40` original（`09:48`）與 correction（`10:05`）：`09:50` replay 保持 original，`10:05` cycle 才使用 correction。
- stored model-cycle lineage 必須 dereference 到 exact as-of version；把 `10:05` correction 錨定到 `09:50` cycle 會被拒絕。

已知環境限制：完整上述 Layer A suite 的 4 個 export tests 需要 Python package `zstandard` 解讀既有 `.zstd` snapshot；目前 runtime 未安裝該 package，4 個測試均在 export read path 以 `ModuleNotFoundError: zstandard` 失敗，與 Session 2 projection diff 無關。

後續（不屬於 Session 3）：

- frontend 明確指定 `Asia/Hong_Kong` formatting 及 tooltip lineage 顯示。
