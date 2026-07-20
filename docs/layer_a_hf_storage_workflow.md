# Layer A local／Hugging Face Dataset workflow

## Canonical flow

`app/services/canonical_cycle.py` 是唯一 upstream cycle builder。背景 scheduler 與 manual strategy runner 都透過 deterministic five-minute cycle key 取得同一個 `CanonicalCycle`；第二個 account 或第二個 entry point 只讀 process-local frozen payload。

builder 的順序是：

```text
weather／forecast／wind
→ all model outputs
→ market metadata
→ one YES／NO DepthCache bundle
→ CLOB snapshot validation
→ freeze cycle object
→ Layer A capture once
→ account context adapters
```

Layer A storage 失敗會記錄 error，但不會阻止既有 paper strategy cycle；下一次 startup／health 可見 incomplete 或 pending state。

## Local partitions

```text
data/layer_a/
  date=YYYY-MM-DD/
    hour=HH/
      cycles-<uuid>.parquet
      books-<uuid>.jsonl.zst
      manifest-<uuid>.json
```

每個 closed partition 目前包含一個 canonical cycle，避免 open partition rewrite。Parquet 與 JSONL 先寫 sibling `.tmp`，manifest 最後以 atomic rename close。manifest 包含 schema version、first／last timestamp、cycle count、cycle ID、檔案大小與 SHA-256。manifest 自身的 hash 由 export bundle 的 `checksums.sha256` 計算，避免 self-hash circularity。

若 process 在 close 中斷，startup scan 會列出 `.tmp` 或缺檔 partition；不會自動刪除、覆寫或假裝 complete。

`zstandard` 是 production dependency。minimal local environment 若沒有此 package，manifest 會明確標記 `books_compression=plain_fallback`；HF Space deployment 應使用 real zstd。

## Export

```powershell
python scripts/export_layer_a.py --date 2026-07-20 --verify-checksums
python scripts/export_layer_a.py --start 2026-07-20 --end 2026-07-20 --only-unuploaded --output .\layer_a.zip
```

archive 內容包括 selected closed Layer A files、partition manifests、`docs/layer_a_schema.md`、`export_manifest.json`、`checksums.sha256`、unuploaded partition list 與 incomplete partition list。Export 只讀 local state，不修改 partition。

Space endpoint 是 `GET /admin/export-layer-a`，使用 `LAYER_A_ADMIN_TOKEN` 的 `X-Layer-A-Admin-Token` 或 Bearer token；未設定 token 時 endpoint disabled。endpoint 沒有任意 path 參數，也不會回傳 secrets。

## Optional Dataset upload

```text
HF_LAYER_A_REPO_ID=<private-dataset-repo>
HF_LAYER_A_TOKEN=<secret>
HF_LAYER_A_AUTO_UPLOAD=false
HF_LAYER_A_UPLOAD_INTERVAL_MINUTES=30
```

upload client 固定使用 `repo_type="dataset"`，remote path 為 `layer_a/date=.../hour=.../<file>`。只上傳 closed immutable partition；每個 file upload 後以 Hub API（若 client 支援）verify remote existence，全部成功才寫 `.upload_receipts/<partition-id>.json`。已存在的 remote file 會跳過，不覆寫。

failed upload 會寫 `.upload_failures/`，bounded exponential backoff 不會使 local capture 失敗。token 不會寫入 log、receipt 或 health output。HF Dataset repo 與 Space code repo 是不同 repository，因此 Dataset upload 不觸發 Space code rebuild。

## Startup and ephemeral Space storage

API lifespan startup 會 scan complete／incomplete／uploaded partitions，並在 auto-upload 開啟時 retry pending closed partitions。`/api/health` 與 protected `/admin/layer-a-health` 顯示 last cycle、today count、CLOB replay eligibility、pending partitions、last remote upload、upload failures、disk usage、oldest unuploaded partition。

HF Space local filesystem 是 ephemeral。只有已 upload 到 private Dataset repo 或 operator 手動下載的 export archive 能跨 restart、stop 或 rebuild 存活。

## Independent one-minute market loop

`layer_a.market.v1` 使用獨立的 `layer_a.market_capture.MarketSnapshotCollector`。
API lifespan 啟動後，它每 60 秒只做 Gamma market reference refresh 與一次合併的
YES/NO CLOB batch fetch；不呼叫 weather ingestion、forecast、model inference、
paper account 或 strategy execution。既有 `canonical_cycle` 仍維持約五分鐘的
weather → all models → metadata → DepthCache → frozen `CanonicalCycle` →
one-time `layer_a.v1` capture。

market-only local layout 為：

```text
data/layer_a_market/
  date=YYYY-MM-DD/
    hour=HH/
      snapshots-<partition-id>.jsonl.zst
      manifest-<partition-id>.json
```

同一 local hour 的 snapshots 先 append 到一個 `.jsonl.tmp`，跨小時或 startup
recovery 才 close 成 compressed immutable payload；因此不會每分鐘建立三個檔案。
manifest last rename、SHA-256、upload receipt、pending upload 與 incomplete
temporary files 的 recovery 規則與 model Layer A 相同。market partition 以
`layer_a_market/...` prefix 上傳到 optional private HF Dataset repository，且
不會寫入 Space code repository。

`/api/health` 與 `/admin/layer-a-health` 另外回報 market store 及 collector 的
last success、failed runs、pending partitions、replay eligibility、upload failures
與 temporary files。CLOB fetch/storage exception 只影響該 market run；model cycle
的 capture exception 也不會阻止 market collector。

## Minute streams and remote read cache

The minute market and weather streams use the same close interval:

```text
LAYER_A_MINUTE_PARTITION_MINUTES=10   # constrained to 5..15
```

Each stream appends JSONL to one temporary chunk per local
`date/hour/minute` slot. Closing writes the compressed payload to a sibling
temporary file, computes SHA-256, writes the manifest, then atomically renames
the payload and manifest. Temporary files are never silently removed. A
closed chunk is the only upload-eligible state. Model-cycle `layer_a.v1`
storage remains one cycle per immutable partition.

`layer_a.historical_store.HistoricalStore` keeps the two storage domains
separate:

```text
data/layer_a/                 writable local model cycles
data/layer_a_market/          writable local market chunks
data/layer_a_weather/         writable local weather chunks
/tmp/layer_a_remote_cache/    downloaded remote files; read-only to readers
```

Remote files are downloaded from the private Dataset repository under
`layer_a/`, `layer_a_market/` and `layer_a_weather/` prefixes. The cache index
reuses unchanged paths, and a reader never calls upload APIs or marks a remote
file as a local pending partition. If refresh fails, local records remain
available and health reports `degraded` or `unavailable` without exposing the
repository token.

The optional upload worker is controlled by `HF_LAYER_A_AUTO_UPLOAD` and its
own `HF_LAYER_A_UPLOAD_INTERVAL_MINUTES`; capture cadence and upload cadence
are therefore independent.

Server startup also starts the account-independent `CanonicalCycleCollector`
at 300 seconds. Its process-local cache is shared with strategy adapters, so
an adapter in the same five-minute slot reuses the frozen cycle instead of
rerunning inference.

## Runtime debug checklist

Local development and the HF Docker Space both start the same application:

```text
uvicorn app.api.server:app --host 0.0.0.0 --port 7860
```

`Dockerfile` builds `app/frontend` with `npm run build` and copies the result
to `app/frontend/dist`; `app/api/server.py` serves that directory with
`StaticFiles`. There is no alternate Procfile or Gradio entrypoint.

The `hf` Git remote in this project points to the Space code repository. It is
not the private Dataset repository used by `DatasetUploader`. Dynamic
`data/layer_a*` directories are intentionally gitignored and cannot be
transferred by `git push hf`. To persist closed Layer A chunks across a Space
rebuild, configure a separate private Dataset and enable the upload worker:

```text
HF_LAYER_A_REPO_ID=<owner>/<private-dataset>
HF_LAYER_A_TOKEN=<secret>
HF_LAYER_A_AUTO_UPLOAD=true
```

The uploader and reader both use `repo_type="dataset"` and these exact paths:

```text
layer_a/date=YYYY-MM-DD/hour=HH/<model files>
layer_a_market/date=YYYY-MM-DD/hour=HH/minute=MM/<market files>
layer_a_weather/date=YYYY-MM-DD/hour=HH/minute=MM/<weather files>
```

The repository-synced legacy CSV is a separate compatibility path. Since
`data/export/YYYY-MM-DD.csv` is tracked but runtime updates are uncommitted,
`git push hf` alone does not transfer the latest local rows. Commit the CSV
before pushing the Space; after rebuild the minute API reads it only when no
Layer A minute records exist and labels each returned row
`source="legacy_csv"`.

Use the deployment-safe checklist after startup:

```powershell
python scripts/layer_a_runtime_diagnostics.py --base-url http://127.0.0.1:7860 --date 2026-07-20 --json
```
