Write-Host "Clinverity release validation starting..." -ForegroundColor Cyan
Write-Host "Phases: framework, full, visual, cache, accessibility, soak" -ForegroundColor DarkCyan

$runOutput = & .\.venv\Scripts\python.exe -u scripts\run_release_validation.py 2>&1
$exitCode = $LASTEXITCODE
$runOutput | ForEach-Object { $_ }

$markdownLine = $runOutput | Where-Object { $_ -match '^Markdown summary:\s+' } | Select-Object -Last 1
if ($markdownLine) {
    $reportPath = ($markdownLine -replace '^Markdown summary:\s+', '').Trim()
    if (Test-Path $reportPath) {
        Write-Host "Opening release validation summary..." -ForegroundColor Green
        Start-Process $reportPath | Out-Null
    }
}

exit $exitCode
