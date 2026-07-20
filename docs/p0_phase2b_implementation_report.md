# P0 Pipeline Repairs — Phase 2B Implementation Report

日期：2026-07-19  
狀態：Phase 2B code、targeted tests、strict replay audit 完成；歷史 export 因 snapshot metadata 不完整而無法進行 executable counterfactual PnL replay。

## 1. Scope 與不變更邊界

本階段只處理 paper execution integrity：把 active strategy path 的 executable quote、depth walk、fee、partial fill、position mutation 與 audit logging 對齊到同一個 decision-time CLOB snapshot。

沒有修改：

- model weights、model specs、training data、calibration artifacts；
- strategy assignment、entry threshold、Kelly policy、active account 狀態；
- `data/current_positions.json`、`data/strategy_accounts.json` 與既有 runtime data；
- live exchange order path。`_paper_guard()` 仍要求 `allow_live_orders=False`。

## 2. Active execution path

目前 active scheduler 的資料流是：

`context_builder / app.api.strategies._build_strategy_context` → `strategy_runner.run_single_strategy_cycle` → `strategy_engine.run_*_rebalance_cycle` → explicit execution mode router → legacy `PaperAdapter` 或 `ClobDepthPaperAdapter`。

`execution/auto_runner.py` 也接入相同的 snapshot builder。`execution.rebalancer.generate_orders_from_probs` 是 inactive legacy path；在 `clob_depth` mode 會直接 blocked，不會偷偷退回 Gamma。

## 3. Explicit rollout modes

新增 `execution/paper_execution_config.py`：

- `legacy_gamma_mock`：既有 Gamma reference + mock slippage 行為；
- `clob_depth_shadow`：預設 rollout mode。legacy paper fill 繼續執行；CLOB path 同時計算但只寫 shadow JSONL，不會再 mutation 一次相同 position；
- `clob_depth`：明確 opt-in 的 CLOB paper execution；沒有 valid snapshot 就 fail closed。

環境變數：

- `PAPER_EXECUTION_MODE`；
- `PAPER_PARTIAL_FILL_POLICY`：`fail_closed`、`accept_partial`、`reduce_to_available`；
- `PAPER_CLOB_MAX_BOOK_AGE_SECONDS`，預設 60 秒；
- `PAPER_CLOB_SHADOW_LOG`，預設 `data/clob_execution_shadow.jsonl`。

預設仍是 shadow，沒有自動把 active paper account 升級成 CLOB mutation。

## 4. Canonical CLOB snapshot contract

`execution/clob_execution.py` 新增 `CLOBExecutionSnapshot`，每個 bucket 的 YES/NO token 都是獨立 snapshot，包含：

`market_id`、`condition_id`、`bucket`、`token_side`、`token_id`、`decision_timestamp`、`book_timestamp`、`book_age_seconds`、`tick_size`、`minimum_order_size`、完整 bids/asks、`fetch_cycle_id`、`schema_version`、`source_name`。

`build_execution_snapshots()` 會拒絕：

- Hong Kong temperature event slug、日期或 highest/lowest kind 不匹配；
- market/model/depth bucket schema 不一致；
- 缺少或重複 market/condition/token identity；
- outcomes 不是明確 `Yes/No`；
- YES/NO token ID 互換、重複或 book `asset_id` 不匹配；
- 缺少 book timestamp、fetch cycle、source、tick/min size；
- YES/NO 不在同一 fetch cycle；
- stale/future book 或 depth parser validation errors；
- 非嚴格排序的 bid/ask levels。

不會以 `time.time()` 補 book timestamp，也不會把 Gamma price 當作缺失 CLOB 的替代品。

`market_service` 現在保留 `outcomes`、YES/NO token IDs、market/condition ID、tick/min-size metadata；`market_depth_service` 保留完整 normalized depth、asset ID、timestamp、source 與 cycle ID。

## 5. Depth walking、fee 與 price semantics

`walk_depth()`：

- BUY 由 asks low-to-high walk；
- SELL 由 bids high-to-low walk；
- 每一個 fill level 分別計算 fee；
- 未成交 shares 不收費。

每 level 使用指定 formula：

`fee = filled_shares × 0.05 × price × (1 - price)`

BUY 使用 all-in VWAP：`(gross_notional + fee) / filled_shares`。  
SELL 使用 net sell VWAP：`(gross_notional - fee) / filled_shares`。

NO token 只在映射到 YES-probability space 時取 complement；實際 adapter 仍以 NO token 自己的 CLOB book 與 fee 計算。

舊 `compute_execution_estimate()` 已改成 CLOB-only fail-closed compatibility helper；Gamma 參數保留為 API compatibility，但不再參與 executable estimate。

## 6. Depth-aware Kelly sizing

`compute_depth_adjusted_bets()` 只把 CLOB top quote 或 NO top-ask complement 傳入 Kelly。Gamma 價格只回傳為 `diagnostic_edge`，不參與：

- candidate side selection；
- Kelly size；
- execution price；
- entry/exit gate；
- fill simulation；
- mark-to-market 或 PnL。

採 bounded fixed-point iteration，最多 6 rounds；每輪用上一輪 all-in executable price 把 Kelly cash allocation 換算成 shares，再 walk depth，直到 action 與 amount 同時符合 `< $1` 且 `<1%`，否則整體 fail closed。這也避免 depth slippage 導致 actual cash exposure 高於 Kelly allocation。

`kelly_betting.compute_multi_kelly_bets()` 新增 `executable_sides`，缺 quote 的 bucket 不會再用 0.5 placeholder；legacy callers 的 default behavior 保留。

## 7. Strategy gates 與 order construction

`strategy_engine.py` 的 enhanced/config paths 現在：

- CLOB mode 不呼叫 `apply_slippage_to_bets()`；
- entry edge 使用 final size-specific all-in executable price；
- exit gate 使用 CLOB bid walk 的 net sell price；
- target quantity 使用實際 depth-filled shares；
- 缺 snapshot、缺 bid depth、partial fail-closed 都產生 explicit `BLOCKED` reason；
- partial policy 為 `accept_partial` / `reduce_to_available` 時，entry target 降為 available filled shares；
- target position 保存 depth VWAP、fee、fill ratio、diagnostic/executable edge 等 decision diagnostics。

`strategy_gate.evaluate_refined_entry()` 對 all-in execution price 不再重複扣一次 slippage percentage。

## 8. Direct CLOB paper adapter

新增 `execution/clob_paper_adapter.py`：

- 只接受已驗證的 `CLOBExecutionSnapshot`；不在 adapter 重新 fetch Gamma/CLOB；
- 使用現有 `pm_trader.Engine` 的 paper SQLite schema/DB interface，但不使用 `pm_trader.Engine.buy/sell` 的 alternate quote fetch 或 alternate fee formula；
- 直接以同一 snapshot walk depth，更新 paper cash、position、trade ledger；
- trade ledger 保存 gross price、fee、levels filled、partial flag 與 slippage diagnostics；
- position average entry cost 使用 all-in cash outflow；
- sell 更新 net proceeds、remaining shares、remaining cost、realized PnL；
- `mark_positions()` 回傳 midpoint mark 與 immediate bid-depth liquidation value/fill ratio；
- legacy JSON mirror 先保留未變更 HOLD positions，再套用 actual fills，避免 partial/增倉時誤刪未列出的 bucket；
- audit events 增加 execution mode、partial policy、VWAP、fee、fill ratio、residual 與 depth-level fields。

## 9. Shadow logging

新增 `execution/shadow_logger.py`。每個 bucket 產生一筆 JSONL record，包含：

model probability、Gamma reference price、legacy simulated price、CLOB best bid/ask、requested size、depth-adjusted VWAP、fee、fill ratio、legacy/clob executable edge、legacy/clob would-trade、action 與 difference reason。

shadow writer 不寫 positions、balances、SQLite 或 legacy audit；`clob_depth_shadow` 只讓既有 legacy paper adapter 完成原本的 paper mutation，不會再執行一個 CLOB mutation。

## 10. Historical replay

新增 `scripts/replay_clob_execution.py`，對 CSV export 執行 read-only strict eligibility audit；不會修改 runtime files，也不會補 metadata。

2026-07-14 至 2026-07-19 的結果詳見 [`reports/clob_execution_replay_2026-07-14_2026-07-19.md`](../reports/clob_execution_replay_2026-07-14_2026-07-19.md)：

- 4,220 rows；4,214 rows 有雙側 depth；
- 0 canonical eligible rows；4,220 rejected；
- 歷史 export 缺 `no_token_id`、explicit outcomes、tick/min size、asset ID、source、fetch cycle；
- 1,800 rows 有至少一本 stale、53 rows 有 future book、1,950 rows 有 YES/NO timestamp mismatch；這些是獨立 book diagnostics；
- export 沒有 persisted target orders，因此 trade decision change、VWAP、fill ratio、fee、turnover 與 PnL 都是 `N/A`；
- Strategy A (`enhanced_v1_paper` / `model_a`) 六日共 1,410 rows，0 eligible。

這個 replay 結果是「資料 contract 不足」的明確證據，不是 counterfactual execution 結果，也沒有用它調整 threshold、Kelly 或 model。

## 11. Verification

已執行：

- `python -m pytest -q tests/test_clob_execution_phase2b.py tests/test_market_depth_service.py tests/test_model_2a_phase1.py tests/test_model_2a_phase2a.py`：**59 passed**；
- `python -m py_compile`：所有本階段 touched Python modules 通過；
- Phase 2B/new helper files 的 `ruff check`：通過；
- isolated temporary paper DB smoke test：BUY 10 shares at 0.20 產生 all-in 0.208、fee 0.08、cash outflow 2.08，position average entry cost 為 all-in cost；
- direct CLOB strategy probe with empty snapshots：completed with zero target mutation，沒有 Gamma fallback。

測試 warning 是既有 `numexpr` version warning、`strategy_gate` deprecation warning 與既有 `datetime.utcnow()` deprecation warning；沒有把它們當作 Phase 2B trade result。

## 12. Remaining operational prerequisite

要把 rollout 從 shadow 提升到 `clob_depth`，先讓每個新 cycle 持久化完整 snapshot contract 與 target order/ledger，再重跑 strict replay，確認有 eligible rows、可重建 fill/fee/position/PnL，並觀察 shadow difference。此步驟不需要更改 model artifact、strategy assignment 或 account policy。
