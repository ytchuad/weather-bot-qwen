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
