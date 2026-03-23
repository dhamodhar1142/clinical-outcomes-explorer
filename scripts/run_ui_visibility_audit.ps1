param(
    [Parameter(Mandatory=$false)]
    [string]$Dataset = ".\tests\fixtures\datasets\STG_EHP__VIST.csv"
)

& .\.venv\Scripts\python.exe scripts\run_dataset_validation.py --dataset $Dataset --fixture default --mode ui
