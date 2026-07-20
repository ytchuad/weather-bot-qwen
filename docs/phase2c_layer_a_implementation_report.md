# Phase 2C-LA：Layer A canonical capture implementation report

## Boundary confirmation

Layer A 現在以 canonical sampling cycle 為唯一 capture 單位，不以 paper account 或 strategy account 為單位。以下三條執行路徑都只建立 account adapter，不再各自 fetch weather、run models 或 re-fetch CLOB：

| entry point | 原本的觸發頻率 | 同一 cycle 的重疊可能性 | 現行行為 |
| --- | --- | --- | --- |
| `POST /api/strategies/run-all` → `execution.strategy_runner.run_enabled_strategies_once()` → `app.services.context_builder.build_strategy_context()` | 手動呼叫；逐一處理 due 的 running accounts，預設 interval 為 registry 的 5 分鐘 | 可與 background scheduler 或 headless runner 同時處理同一 5 分鐘 slot | adapter → shared `get_canonical_cycle()` |
| `app.api.strategies._scheduler_loop()` → `_build_strategy_context()` | thread 每 30 秒巡檢；每 account 的 `last_run` cooldown 為 300 秒 | 可與手動 `/run-all` 重疊 | adapter → shared `get_canonical_cycle()` |
| `execution.auto_runner.run_strategy()` | 每次 CLI/cron invocation 逐一處理 due accounts；`is_due()` 使用 300 秒 cooldown | 可與 Space scheduler 或手動呼叫重疊 | 已改為同一 shared cycle adapter |

兩個 entry point 同時執行時，deterministic `decision_cycle_id` 加上 process-local cache 會讓它們取得同一個 `CanonicalCycle` object；Layer A storage 再以相同 ID 做 persistent deduplication。第二個 account 不會觸發第二組 CLOB fetch。

## Canonical cycle freeze location

`app/services/canonical_cycle.py::_build_uncached_cycle()` 是唯一 upstream builder，順序是：

```text
fetch event/market metadata
→ fetch HKO forecast and observation state
→ compute rain/nowcast inputs; run all model outputs once
→ update token maps and read one YES/NO DepthCache bundle
→ validate market/token identity and exact CLOB snapshots
→ build normalized Layer A record and completeness flags
→ construct frozen CanonicalCycle
→ persist Layer A once (capture/dedup)
→ distribute the same cycle to account adapters
```

`CanonicalCycle` 是 `frozen=True` dataclass。account adapter 只讀取它，選擇 account 指定的 model，並添加 execution 所需的 account-local `capital`、`kelly_fraction` 等欄位；這些欄位不會傳入 Layer A record。

Canonical model generation 使用 neutral `bias=0.0`、`std_mult=1.0`，因此 capture 不依賴 account。為保留舊 runtime 對 `9d`/`aws` 的行為，這兩個 profile 的 `bias`/`std_mult` 只在 adapter 對已生成的 canonical prediction 做 deterministic derived view；不會重新 fetch 或重新 run models。現有 `model_a`/`model_b`/`model_g` intraday outputs 不受這兩個舊參數改寫。

## `layer_a.v1` record

每個 record 包含：

- cycle envelope：`decision_cycle_id`、schema、decision/capture timestamp、event identity；
- weather/forecast/wind/rain/nowcast state，以及 Phase 2A input-status fields；
- cycle 內所有 generated models，而非 account-selected model：model name/version、artifact identity、feature spec、numeric/diagnostic features、point/quantile predictions、full bucket probabilities、classifier probability、input-status summary；
- market identity：market/condition ID、explicit YES/NO outcomes、YES/NO token/asset IDs、tick size、minimum order size、market schema version；
- 每個 bucket 的 exact normalized YES 與 NO full-depth books：完整 bids/asks、timestamp、age、source、fetch-cycle ID、tick/min size、validation status/errors、normalized source book；
- `completeness`：weather、model、market、token、YES/NO depth pair、book timestamp、fetch-cycle coherence，以及 model/CLOB replay eligibility 和 rejection reasons。

`validate_layer_a_record()` 會遞迴拒絕 account/trading/PnL 欄位，包括 account ID、strategy ID、capital/cash、paper position、target order、fill、realized/unrealized PnL 等。缺少資料不會被填假值；該 cycle 會保留為 incomplete 並列出原因。

## Local persistence and recovery

runtime root 預設為 `data/layer_a/`，並由 `.gitignore` 排除：

```text
data/layer_a/
  date=YYYY-MM-DD/
    hour=HH/
      cycles-<partition-id>.parquet
      books-<partition-id>.jsonl.zst
      manifest-<partition-id>.json
```

每個 closed partition 目前一個 canonical cycle。Parquet、books JSONL 與 manifest 先寫 sibling `.tmp`；manifest 最後 atomic rename，startup scan 可辨認 crash 留下的 temporary/incomplete partition，不會自動刪除或覆寫。manifest 保存 schema、cycle IDs、timestamps、sizes、SHA-256 與 compression mode；export bundle 另行保存 manifest/file checksums。

`LayerAStore.capture()` 在 close 前以 manifest、Parquet 及 temporary Parquet 掃描 `decision_cycle_id`。相同 ID 回傳 `duplicate`，不建立第二 partition。capture/storage failure 不會阻斷既有 paper execution；health 會暴露 incomplete/pending state。

## Export, HF Dataset upload and health

CLI：

```powershell
python scripts/export_layer_a.py --date 2026-07-20 --verify-checksums
python scripts/export_layer_a.py --start 2026-07-20 --end 2026-07-20 --only-unuploaded --output .\layer_a.zip
```

支援 `--date`、`--start`、`--end`、`--only-unuploaded`、`--verify-checksums`。archive 包含 selected closed files、partition manifests、`export_manifest.json`、`checksums.sha256`、schema document，並列出 pending/incomplete partitions。`scripts/replay_layer_a.py` 可從 directory 或 zip read-only 載入，重建 full CLOB depth walk、fee/VWAP、threshold candidate 與 Kelly probe；不修改 archive 或 stored data。

Space protected endpoints：

- `GET /admin/export-layer-a`：由 `LAYER_A_ADMIN_TOKEN` 加 `X-Layer-A-Admin-Token` 或 Bearer token 保護；
- `GET /admin/layer-a-health`：同一 token 保護；
- `GET /api/health`：提供非 secret Layer A health summary。

Optional private HF Dataset upload 使用：

```text
HF_LAYER_A_REPO_ID=<private-dataset-repo>
HF_LAYER_A_TOKEN=<secret>
HF_LAYER_A_AUTO_UPLOAD=false
HF_LAYER_A_UPLOAD_INTERVAL_MINUTES=30
```

上傳固定使用 `repo_type="dataset"`，只上傳 closed immutable partition，逐檔 bounded retry/backoff，remote existence verification 通過後才寫 `.upload_receipts/<partition-id>.json`。已上傳檔案不覆寫；失敗寫 `.upload_failures/`，token 不進 receipt、health 或 log。Dataset repository 與 Space code repository 分離，不會因 data upload 觸發 Space code rebuild。

API startup lifespan 會先 `startup_scan()`；開啟 `HF_LAYER_A_AUTO_UPLOAD` 時會 retry pending closed partitions。這是必要的，因為 Space local filesystem 是 ephemeral；跨 restart 的 durable copy 必須透過 private Dataset upload 或 operator download export archive 保存。

## Verification

- Layer A + Phase 1/2A/market-depth targeted suite：`50 passed`；加入 market schema、hourly storage、collector、export 與 replay tests 後，Layer A market suite：`25 passed`。
- `tests/` suite 排除兩個既有 external/dependency-only test files：`149 passed`。repository-wide pytest discovery 另外會執行既有 `tools/dashboard_strategy_runner_smoke_test.py`，其現有兩個 assertion failures 與 `SystemExit` 不屬於本次 Layer A change。
- 未排除的 full suite 不能完成：`tests/test_clob_execution_phase2b.py` 與 `tests/test_paper_trader_connect.py` 需要目前 environment 未安裝的 `pm_trader`；paper-trader connectivity tests 另會直接連 Gamma API，而 sandbox 以 `WinError 10013` 阻擋 socket。這不是 Layer A capture failure。
- touched Layer A/canonical/API/adapter/script/test files 的 `ruff check` 通過；`py_compile` 通過；FastAPI OpenAPI 確認兩個 admin paths 已註冊。

目前 `data/layer_a/` 與 `data/layer_a_market/` 尚未產生 runtime data，因為測試使用 temporary roots；runtime Layer A data 不會提交到 Space code repository。既有 dirty worktree、runtime CSV/Parquet/JSON、model source changes 均保留，沒有使用 reset/checkout 或刪除操作；model artifact directories 與 strategy config files 未被本階段修改。

本階段只建立 Layer A；沒有新增 Layer B/C persistence，也沒有修補歷史上缺乏 exact CLOB evidence 的舊 export。

## Phase 2C-LA extension：one-minute market capture

為保留原本 one-minute Polymarket market-data frequency，新增
`layer_a/market_capture.py` 與 `layer_a/market_storage.py`。collector 是
strategy-independent daemon，與 paper accounts 的 scheduler 分離：

```text
Gamma market metadata (reference only)
→ one combined YES/NO CLOB batch
→ exact normalized market snapshot
→ latest completed five-minute model-cycle link
→ hourly append-and-close market partition
```

每分鐘不重跑 weather 或 models。`market_snapshot_id` 是 deterministic，
storage 以該 ID deduplicate；每一小時只產生一個 compressed JSONL payload
與一個 manifest。snapshot 保留 market/condition/bucket/token/asset identity、
full bids/asks、timestamps、source、`fetch_cycle_id`、tick/minimum size、
validation status/errors。`latest_model_cycle_id` 與 `model_age_seconds` 只指向
該時間以前的 completed model cycle。Gamma data 僅在 reference 欄位保存。

`export_layer_a()` 現在可將 closed `layer_a_market/` partitions 與五分鐘 model
partitions 放入同一 downloadable archive；replay 可將一個 model record 對應到
下一個 model cycle 前的每一本 one-minute CLOB book。新增測試覆蓋四本
one-minute books 的 replay、hourly append/dedup、model link、exact identity/full
depth、single-batch collector 與 export archive loading。

## Boundary and failure isolation

five-minute full model capture 的 freeze hook 仍是
`app/services/canonical_cycle.py::_build_uncached_cycle()`，不受 one-minute
collector 改動。market collector 不進入任何 account context entry point；它只
讀取已完成 model store 以建立 link。market fetch/storage failure 被 collector
捕捉並寫入 health/source status，不會阻止 model cycle；model capture failure
同樣不會停止 market thread。runtime data root `data/layer_a_market/` 已加入
`.gitignore`，不會提交到 Space code repository。
