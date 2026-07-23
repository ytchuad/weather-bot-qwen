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
