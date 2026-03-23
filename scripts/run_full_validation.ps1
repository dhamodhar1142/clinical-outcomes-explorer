param(
    [Parameter(Mandatory=$false)]
    [string]$Dataset = ".\tests\fixtures\datasets\STG_EHP__VIST.csv"
)

Write-Host "Clinverity full validation starting..." -ForegroundColor Cyan
Write-Host "Dataset: $Dataset" -ForegroundColor DarkCyan
Write-Host "Mode: full uploaded-dataset validation" -ForegroundColor DarkCyan

$runOutput = & .\.venv\Scripts\python.exe -u scripts\run_dataset_validation.py --dataset $Dataset --fixture default --mode full 2>&1
$exitCode = $LASTEXITCODE
$runOutput | ForEach-Object { $_ }

$markdownLine = $runOutput | Where-Object { $_ -match '^Markdown report:\s+' } | Select-Object -Last 1
if ($markdownLine) {
    $reportPath = ($markdownLine -replace '^Markdown report:\s+', '').Trim()
    if (Test-Path $reportPath) {
        Write-Host "Opening markdown report..." -ForegroundColor Green
        Start-Process $reportPath | Out-Null
    }
}

exit $exitCode
