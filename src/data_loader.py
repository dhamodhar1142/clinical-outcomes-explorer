from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/synthetic_hospital_data.csv")


def load_hospital_data(csv_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the hospital dataset and normalize column names."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'.")

    data = pd.read_csv(path)
    data.columns = [column.strip().lower().replace(" ", "_") for column in data.columns]
    return data
