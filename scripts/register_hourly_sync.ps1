<#
.SYNOPSIS
  A — 一鍵註冊「每小時下載 HF snapshot」的 Windows 工作排程任務。

  註冊後, 系統每小時的第 5 分鐘會跑 scripts/hf_sync_hourly.ps1：
    - 把 HF 上的新增 snapshot 抓回本地 data/export/
    - 確認本地與 HF 一致, 有落差就寫進 output/hf_sync_cron.log 告警

  注意：
    - 需以「目前使用者」身分、且勾選「只有在使用者登入時才執行」(預設)，
      因為 OneDrive 路徑在使用者登入後才掛載。
    - 本任務「不」自動 commit/push git —— 避免定時任務意外推 HF。
    - 若要改頻率/時間, 改下面 $trigger 那行即可。

  移除任務：Unregister-ScheduledTask -TaskName "HF_SnapshotSync_Hourly" -Confirm:$false
#>

$RepoDir = "C:\Users\cyt\OneDrive\Documents\Weather_Bot_Qwen"
$TaskName = "HF_SnapshotSync_Hourly"
$PsExe = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
$Script  = Join-Path $RepoDir "scripts\hf_sync_hourly.ps1"

$action = New-ScheduledTaskAction `
    -Execute $PsExe `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`""

# 每小時的第 5 分鐘執行
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date "00:05:00") `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# 以目前登入使用者執行 (OneDrive 路徑需使用者環境)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force

Write-Host "已註冊工作排程: $TaskName (每小時 :05 執行)"
Write-Host "檢視: 工作排程器 -> 工作排程程式庫 -> $TaskName"
Write-Host "移除: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
