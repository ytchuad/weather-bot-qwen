param(
    [string]$HfUrl = "https://shea-hilton-weather-prediction.hf.space"
)

Write-Host "=== Step 1: Download snapshots from HF Space ==="
python scripts/download_snapshots.py $HfUrl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Step 1b: Sync pre-check (本地 vs HF 一致性) ==="
python scripts/check_hf_sync.py $HfUrl
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ABORT] 本地落後於 HF，有資料尚未抓回。" -ForegroundColor Red
    Write-Host "請先確認資料已抓回（若真的是永久遺失，請手動確認後再忽略）。" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 2: Check if CSV changed ==="
$diff = git diff --stat data/export/
if (-not $diff) {
    Write-Host "No new data. Pushing current state..."
} else {
    git add data/export/
    git commit -m "sync snapshot exports before push"
}

Write-Host "`n=== Step 3: Push to GitHub ==="
git push origin main
if ($LASTEXITCODE -ne 0) { Write-Host "GitHub push failed!"; exit $LASTEXITCODE }

Write-Host "`n=== Step 4: Push to HF Spaces ==="
git push hf main
if ($LASTEXITCODE -ne 0) { Write-Host "HF push failed!"; exit $LASTEXITCODE }

Write-Host "`n=== Done ==="
