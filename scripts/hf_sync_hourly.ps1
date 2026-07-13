<#
.SYNOPSIS
  A — 定時下載 HF snapshot 的 wrapper (每小時由 Windows 工作排程器呼叫)。

  流程：
    1. 跑 download_snapshots.py 把 HF 上新增的 snapshot append 回本地 data/export/
       (只加新列, 不刪本地資料)
    2. 跑 check_hf_sync.py 確認抓回後本地與 HF 一致;
       若仍有落差 (例如 HF 暫存被重置導致永久遺失), 把遺失時窗寫進 log 告警,
       但不自動 commit / push —— 避免定時任務意外推 HF。

  設計原則：
    - 靜默成功：一切正常時只寫一行 timestamp 到 log。
    - 出錯顯眼：有落差 / 連不上 HF 時, 在 log 用 [WARN]/[ERR] 標記。
    - 不碰 git：自動 commit/push 風險太高, 交給人工或 sync_and_push.ps1。

  註冊排程請用同一目錄下的 register_hourly_sync.ps1。
#>
param(
    [string]$HfUrl = "https://shea-hilton-weather-prediction.hf.space",
    [string]$RepoDir = "C:\Users\cyt\OneDrive\Documents\Weather_Bot_Qwen"
)

$ErrorActionPreference = "Stop"
$log = Join-Path $RepoDir "output\hf_sync_cron.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Log($msg) { "$stamp  $msg" | Add-Content -Path $log -Encoding UTF8 }

Push-Location $RepoDir
$env:PYTHONIOENCODING = "utf-8"  # 避免 cp1252 印 emoji/中文時 UnicodeEncodeError
try {
    Log "[RUN] download_snapshots.py"
    & python scripts/download_snapshots.py $HfUrl 2>&1 | ForEach-Object { Log $_ }

    Log "[RUN] check_hf_sync.py"
    & python scripts/check_hf_sync.py $HfUrl 2>&1 | ForEach-Object { Log $_ }
    # check_hf_sync 回傳 1 = 有落差(告警但不中止), 2 = 連不上
}
catch {
    Log "[ERR] $_"
}
finally {
    Pop-Location
    Log "[DONE]"
}
