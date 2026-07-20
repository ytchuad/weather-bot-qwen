# Layer A Runtime Integration Debug Report

日期：2026-07-20

## 1. Root cause

問題不是 schema 或 deterministic ID，而是 production path 的 runtime integration：

1. `HistoricalStore.records()` 原本只在 remote refresh 時增加 generation。local
   market/weather open chunk append 後，API 仍讀取第一次 query 的 in-memory cache，
   所以 UI 長時間只看到舊 row。
2. open chunk 雖然有讀取分支，但未能可靠處理最後一行尚未寫完的 JSONL；現在只忽略
   partial final line，保留前面完整 records。
3. minute API 沒有 legacy CSV fallback。`data/export/YYYY-MM-DD.csv` 即使已在 Space
   repository，歷史 UI 也不會讀取它。
4. `git push hf` 只更新 HF Space code repository。`data/layer_a*` 被 `.gitignore`
   排除，不能靠 Git push 傳遞；目前 `data/export/2026-07-20.csv` 仍是工作樹的
   uncommitted modification，也不會被 `git push` 傳遞。
5. `HF_LAYER_A_REPO_ID` reader/uploader 使用 `repo_type="dataset"`，但目前 `hf`
   remote 是 `https://huggingface.co/spaces/Shea-Hilton/weather-prediction`，即 Space
   repo，不是 private Dataset repo。兩者不能互相替代。
6. frontend request 原本沒有 `cache: "no-store"`、AbortSignal 或明確 refresh timestamp，
   error message 亦被簡化。

## 2. 實際 entrypoint

Local README、HF Space YAML frontmatter 與 Dockerfile 指向同一個 entrypoint：

```text
uvicorn app.api.server:app --host 0.0.0.0 --port 7860
```

HF Space 設定為 `sdk: docker`、`app_port: 7860`。Dockerfile 先在
`node:22-slim` build `app/frontend/dist`，再將它 copy 到 Python image，最後以
`app.api.server:app` 啟動 Uvicorn。沒有 Procfile、Gradio 或 alternate Python entrypoint。

`app/api/server.py` 的 FastAPI `lifespan` 確實會執行，並啟動 market/weather/canonical、
remote refresh 與 optional upload worker；現在每個 service 都有不含 token 的 startup log。

## 3. Runtime evidence

### 既有 local server baseline

直接查詢正在運作的 `127.0.0.1:7860`：

- market collector：65 runs、128 snapshots、last success `2026-07-20T07:50:59Z`
- weather collector：65 runs、64 snapshots、last success `2026-07-20T07:50:57Z`
- canonical collector：300 秒 cadence、13 runs
- local minute chunks：2 open、15 closed
- remote history：`disabled`
- 舊 `/api/history/minute` 只有 1 row，沒有 `retrieved_at` / `sources`

這個結果直接證明 collector 有運作，但 historical reader cache 使 UI 不更新。

### 修正版 actual server probe

使用相同 repository code 與 Anaconda runtime，在隔離的 temporary Layer A roots 啟動
`app.api.server:app`，只停用 probe 內既有 strategy scheduler 以避免 paper state 變動，
觀測 150 秒：

- market tick：`07:53:34`、`07:54:34`、`07:55:34` UTC
- weather tick：`07:53:34`、`07:54:34`、`07:55:34` UTC
- `market_collector_running=true`、`weather_collector_running=true`
- `/api/health` 與 `/api/history/minute` 均返回 HTTP 200
- `retrieved_at` 每次 query 更新
- 無 Layer A source 時回傳 58 個 `source="legacy_csv"` rows

該隔離 probe 的 external source 回報 market `no_markets/no_token_ids`、weather
`state_unavailable`；這是該 runtime/network probe 的資料源狀態，不是 collector thread
沒有 tick。既有 port 7860 的 live health 已證明在原有執行環境中 market/weather
snapshots 能成功寫入。

## 4. Storage/open chunk 修正

`HistoricalStore` 現在每次 query 重新掃描 local stores，並合併：

```text
remote closed history
+ local closed partitions
+ local current open market chunk
+ local current open weather chunk
```

open chunk 只讀、不具 upload eligibility；partial final line 會被安全忽略；close
後以 snapshot ID deduplicate，因此 close transition 不會增加 duplicate UI row。

## 5. API 與 timezone

以下 routes 已在 actual app OpenAPI 及 running TestClient/server probe 驗證：

```text
GET  /api/history/minute
GET  /api/history/model-cycles
GET  /api/history/market-snapshots
GET  /api/history/weather-snapshots
POST /admin/layer-a-history-refresh
```

Layer A timestamps 以 aware ISO 8601 儲存；naive compatibility timestamps 使用 HKT
wall-clock，不會對已是 HKT 的 legacy CSV 再加八小時。minute projection 的 model
join 仍是 backward as-of，沒有 future model leakage。

## 6. Remote history

Dataset reader/uploader 都驗證 `repo_type="dataset"`，並使用：

```text
layer_a/date=YYYY-MM-DD/hour=HH/<model files>
layer_a_market/date=YYYY-MM-DD/hour=HH/minute=MM/<market files>
layer_a_weather/date=YYYY-MM-DD/hour=HH/minute=MM/<weather files>
```

remote files 只下載到 `LAYER_A_REMOTE_CACHE_DIR`，不會被 remote reader upload；
refresh health/log 只報 configured/status/file counts/latest timestamp/error class，
不報 token。

## 7. Frontend wiring

`MinuteHistoryPanel.tsx` 已由 `Hub.tsx` import/render，mount 立即載入，60 秒 polling，
使用 `cache: "no-store"`、React Query AbortSignal、loading/empty/error states 與
refresh timestamp。UI 保留 minute weather/model history，但移除「Actual observations
and execution books」標題與 execution-book column。

Production build path：

```text
app/frontend/package.json
  npm run build
      ↓
app/frontend/dist/
      ↓ Docker COPY --from=frontend-build
app/frontend/dist/
      ↓ FastAPI StaticFiles mount "/"
```

實際 build 產物：`app/frontend/dist/assets/index-rWZzmYOz.js`，build timestamp
`2026-07-20T07:39:55Z`，SHA-256
`fa2d99d4949aa3dbba24655ca82e1e67f95f5c5d86e76401848bd804728d961f`；直接請求
`http://127.0.0.1:7860/` 可見相同 hashed bundle。

## 8. Legacy CSV policy

`features/strategy_snapshot_logger.py` 的 strategy scheduler cooldown 是 300 秒，
因此 `data/export/YYYY-MM-DD.csv` 是 legacy five-minute canonical export，不是假裝
model 每分鐘 inference。現有資料的最近間隔實測包含約 `5.17`、`16.80`、`22.52`、
`50.99` 分鐘，慢於五分鐘是 scheduler/network/run duration 造成的。

minute API 只在該日期沒有任何 Layer A minute records 時讀取 CSV fallback；rows 標記
`source="legacy_csv"`，不建立 CLOB identity。只要 Layer A 存在，就優先 Layer A。

## 9. Files changed

- `layer_a/historical_store.py`
- `layer_a/market_capture.py`
- `layer_a/market_storage.py`
- `layer_a/weather_capture.py`
- `layer_a/weather_storage.py`
- `layer_a/minute_view.py`
- `app/api/server.py`
- `app/api/health.py`
- `app/api/layer_a.py`
- `app/api/history.py`
- `app/config.py`
- `app/frontend/src/api/client.ts`
- `app/frontend/src/components/MinuteHistoryPanel.tsx`
- `app/frontend/src/types/index.ts`
- `scripts/layer_a_runtime_diagnostics.py`
- `tests/test_layer_a_runtime_debug.py`
- `docs/layer_a_historical_ui.md`
- `docs/layer_a_hf_storage_workflow.md`

沒有修改 model artifacts、inference、model cadence、strategy parameters、Kelly、
paper accounts 或 CLOB execution semantics。

## 10. Tests and verification

隔離 local Layer A roots 後執行：

```text
49 passed  tests/test_layer_a.py tests/test_layer_a_market.py
         tests/test_layer_a_weather_history.py tests/test_layer_a_runtime_debug.py
173 passed broader relevant Layer A/model/depth suite
compileall: passed
ruff: passed
npm.cmd run build: passed
actual OpenAPI/server route probe: passed
actual 150-second collector/server probe: completed
```

完整 repository suite 仍有既有 smoke/external blockers：
`tools/dashboard_strategy_runner_smoke_test.py` collection 會 `sys.exit(1)`，另有
測試依賴未提供的 `pm_trader`、external Gamma network 與 fixtures；沒有為此改動交易
或模型程式。

## 11. Post-deployment verification

Local：

```powershell
uvicorn app.api.server:app --host 0.0.0.0 --port 7860
python scripts/layer_a_runtime_diagnostics.py --base-url http://127.0.0.1:7860 --date 2026-07-20 --json
Invoke-RestMethod http://127.0.0.1:7860/api/health
Invoke-RestMethod "http://127.0.0.1:7860/api/history/minute?date=2026-07-20&limit=10000"
```

HF Space rebuild 後：

```powershell
python scripts/layer_a_runtime_diagnostics.py --base-url https://shea-hilton-weather-prediction.hf.space --date 2026-07-20 --json
curl.exe -H "Cache-Control: no-cache" "https://shea-hilton-weather-prediction.hf.space/api/history/minute?date=2026-07-20&limit=10000"
```

若依賴 legacy CSV，必須先 commit tracked CSV，再 `git push hf main`；若要跨 Space
rebuild 保存 Layer A chunks，必須另設 private Dataset 並開啟 `HF_LAYER_A_AUTO_UPLOAD`。

## 12. Remaining limitations

- HF Space local filesystem 是 ephemeral；未 upload 到 Dataset 的 open/closed Layer A
  chunks 不會跨 rebuild 保存。
- `git push hf` 不會包含 uncommitted CSV。
- external HKO/Gamma/CLOB availability 仍可能令單次 collector run 失敗；health 會
  顯示 last tick、last success、last error，下一個 tick 會繼續嘗試。
- legacy CSV 只提供 compatibility display，不可重建 Layer A snapshot ID 或 CLOB book。
