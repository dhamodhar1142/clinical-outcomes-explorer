param(
    [Parameter(Mandatory=$false)]
    [string]$Dataset = ".\tests\fixtures\datasets\SMALL_HEALTHCARE_VISITS.csv"
)

& .\.venv\Scripts\python.exe scripts\run_dataset_validation.py --dataset $Dataset --fixture small-healthcare --mode quick
