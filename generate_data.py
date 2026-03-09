import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "patient_id": np.random.randint(10000, 99999, n),
    "admission_id": np.arange(1, n + 1),
    "age": np.random.randint(18, 90, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "department": np.random.choice(
        ["Cardiology", "ICU", "Orthopedics", "Neurology", "General Medicine"], n
    ),
    "diagnosis": np.random.choice(
        ["Infection", "Heart Failure", "Stroke", "Fracture", "Diabetes"], n
    ),
    "comorbidity_score": np.random.randint(0, 6, n),
    "prior_admissions_12m": np.random.randint(0, 4, n),
    "length_of_stay": np.random.randint(1, 15, n)
})

data["cost"] = data["length_of_stay"] * np.random.randint(1000, 4000, n)

risk_score = (
    data["comorbidity_score"]
    + data["prior_admissions_12m"]
    + (data["length_of_stay"] > 7).astype(int)
)

data["readmitted"] = np.where(risk_score > 4, "Yes", np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]))

data.to_csv("data/synthetic_hospital_data.csv", index=False)

print("Synthetic dataset created successfully.")