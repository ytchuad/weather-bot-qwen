# Project Claude Guidance

## Language
- 一律用 **繁體中文** 回覆使用者。
- 複雜術語（如 Kelly、slippage、Gamma mid、backtest、ensemble 等）可保留英文並以括號補充，方便對照程式碼與文獻。

## HF 推送工作流（重要）
每次要把變更推到 Hugging Face Space（`hf` remote）前，**必須先**把 Space 上累積的
snapshot record 下載回本地、與程式碼一起提交後，再推 HF：

1. 下載 record：`python scripts/download_snapshots.py https://shea-hilton-weather-prediction.hf.space`
   （合併進 `data/export/*.csv`，僅 append 新列，dedup by timestamp+strategy_key）
2. 若有新資料：`git add data/export/ && git commit -m "sync snapshot exports before push"`
3. `git push origin main`（GitHub）
4. `git push hf main`（Hugging Face Space）

> 原因：HF Space 的檔案系統是**暫時的**，`git push hf` 會重置它。
> 若不同步 record，Space 上已累積的 snapshot 會遺失。完整腳本見 `scripts/sync_and_push.ps1`。
>
> 規則：使用者未明確要求時，**不要**推 HF（只推 `origin`）。一旦要求推 HF，就要走上述完整流程。

## HF 同步防遺失（A+B 雙保險）
2026-07-12 發生過一次慘痛教訓：17:24 之後的 snapshot 只存在 HF 暫存、
還沒抓回本地就因 push 重置而**永久遺失**。根因是「下載」只在 push 前觸發、
平時 HF 一直在產生新資料卻無人備份。防範機制：

- **A. 定時下載**：註冊 Windows 工作排程（每小時 :05 跑）
  `scripts/hf_sync_hourly.ps1`（先 `download_snapshots.py` 抓回、再 `check_hf_sync.py`
  確認；有落差只寫 log 告警，**不**自動 commit/push）。一鍵註冊：
  `powershell -File scripts/register_hourly_sync.ps1`。log 在 `output/hf_sync_cron.log`。
- **B. pre-check 守門員**：`scripts/check_hf_sync.py` 比對「本地最後 timestamp」vs
  「HF export 端點最後 timestamp」，有落差（HF 較新 / 整日遺失）就印紅字 + 回傳 exit 1。
  已被接進 `sync_and_push.ps1` 的 Step 1b（download 之後、push 之前）：
  **若有落差，直接 abort 推 HF**，逼人工先確認資料已抓回。
- 跑分析 / 回測**之前**也建議先跑一次 `check_hf_sync.py`，避免基於落後資料下結論。
