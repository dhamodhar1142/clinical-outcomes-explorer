# Clinical Outcomes Explorer

### Dual-Mode Streamlit Analytics Dashboard for CSV Profiling and Healthcare Operations Insights

## Overview
Clinical Outcomes Explorer is a Streamlit dashboard designed to do two things well:

1. Profile and summarize any uploaded CSV through a generic analytics workflow.
2. Activate advanced healthcare analytics when the uploaded dataset contains enough mapped clinical and operational fields.

That dual-mode design makes the project more realistic and more versatile than a fixed demo dashboard. A recruiter or hiring manager can see both broad analytics product thinking and healthcare-specific domain application in one portfolio project.

## Business Problem
Analytics teams often receive raw CSV extracts with inconsistent schemas, mixed data quality, and varying levels of business readiness. In healthcare settings, that challenge is even more important because leaders need fast answers to questions like:

- Which departments are driving cost pressure?
- Which diagnoses show elevated readmission risk?
- Which patient cohorts require closer intervention?
- Where are there opportunities to reduce length of stay or avoid preventable readmissions?
- Is the uploaded data complete enough to support advanced operational analysis?

Clinical Outcomes Explorer addresses that problem by combining automatic dataset profiling with a healthcare-specific analytics layer that only activates when the detected schema supports it.

## Two Analysis Modes
### 1. Generic Auto-Analysis
This mode runs for any uploaded CSV, regardless of industry or schema.

It automatically profiles the dataset and surfaces:
- row count and column count
- column names and inferred data types
- missing values by column
- duplicate row count
- unique values by column
- numeric summary statistics
- top categories for categorical columns
- correlation matrix for numeric columns
- basic outlier detection for numeric columns
- automatic date-column detection
- trend charts when a date field is detected
- automatic charts chosen from available numeric and categorical columns

This means the app remains useful even when the uploaded file has no healthcare fields at all.

### 2. Healthcare Analytics Mode
This mode activates only when schema matching detects enough healthcare-relevant fields, such as combinations of:
- diagnosis
- department
- cost
- length of stay
- readmission
- age
- gender
- comorbidity score
- prior admissions
- date

When that coverage is strong enough, the app unlocks advanced clinical and hospital operations views such as KPIs, cohort analysis, risk modeling, cost drivers, benchmarking, scorecards, and intervention scenarios.

If the uploaded CSV does not contain enough mapped healthcare fields, the app stays in Generic Auto-Analysis mode and presents one clear message rather than showing a page full of unavailable sections.

## How Uploaded CSVs Are Profiled
When a user uploads a CSV, the app:

1. Normalizes column names.
2. Attempts schema matching against common healthcare aliases.
3. Builds a generic dataset profile based on rows, columns, inferred data types, and completeness checks.
4. Detects likely date columns and generates trend views when possible.
5. Determines whether the dataset is suitable for advanced healthcare analysis.
6. Enables healthcare tabs only when schema coverage supports them.

This keeps the experience stable, professional, and adaptable to real-world file variability.

## Key Features
- Sidebar CSV upload for uploaded-data workflows
- Generic Auto-Analysis for any CSV file
- Automatic schema detection and healthcare field mapping
- Clean split between generic profiling and healthcare analytics
- Executive summary and KPI monitoring for healthcare-ready datasets
- Cohort analysis by age group, diagnosis, department, gender, and comorbidity bucket
- Logistic regression model for readmission risk prediction
- Model explainability using logistic regression coefficients
- Risk segmentation and high-risk patient review
- Cost driver analysis and department scorecards
- Benchmarking against the overall uploaded dataset
- Operational scenario comparison for LOS and readmission improvement ideas
- Data quality review and export-ready outputs
- Rule-based analytics assistant for common business questions
- Graceful fallback behavior for sparse filters and partial schemas

## Tech Stack
- Python
- Streamlit
- Pandas
- Plotly
- scikit-learn

## Local Setup
1. Clone or download the repository.
2. Create and activate a virtual environment.
3. Install project dependencies.

```bash
pip install -r requirements.txt
```

4. Make sure the sample dataset exists at `data/synthetic_hospital_data.csv` if you want to use the built-in healthcare demo.

## Run the App
```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

You can either:
- use the included synthetic hospital dataset
- upload your own CSV for generic auto-analysis
- upload a healthcare-oriented CSV to unlock advanced clinical analytics when the schema is compatible

## Example Healthcare CSV Schema
A healthcare upload does not need to match these names exactly, but fields like the following help activate the advanced healthcare mode:

```text
admission_id
admission_date
age
gender
diagnosis
department
length_of_stay
cost
readmission
comorbidity_score
prior_admissions_12m
```

The schema matcher also supports common aliases such as:
- `patient_age`
- `sex`
- `dx`
- `diagnosis_group`
- `unit`
- `service_line`
- `los`
- `total_cost`
- `charges`
- `readmitted`
- `readmission_flag`
- `admit_date`
- `encounter_date`

## Project Structure
```text
clinical-outcomes-explorer/
+-- app.py
+-- data/
|   +-- synthetic_hospital_data.csv
+-- src/
|   +-- analytics.py
|   +-- analytics_assistant.py
|   +-- charts.py
|   +-- data_loader.py
|   +-- generic_profile.py
|   +-- metrics.py
|   +-- schema_detection.py
+-- README.md
+-- requirements.txt
```

## Screenshots
Add screenshots here after running the app locally.

Suggested screenshots:
- Generic Auto-Analysis overview for a non-healthcare CSV
- Detected schema and healthcare-mode activation on a compatible file
- Executive Overview and KPI section
- Readmission Risk Modeling and explainability section
- Cost and department performance views

## Synthetic Data Note
The included sample healthcare dataset is synthetic and is used for demonstration purposes only. It does not contain real patient records, protected health information, or production clinical data.

## Why This Project Matters for Healthcare Analytics Roles
This project is especially relevant for healthcare analytics, hospital operations, and clinical strategy roles because it demonstrates:
- practical KPI design for utilization, cost, and readmissions
- strong attention to schema variability and data readiness
- cohort-based and department-based operational analysis
- interpretable predictive modeling for readmission risk
- dashboard thinking for both analyst and executive audiences
- modular Python design with reusable analytics helpers
- the ability to translate messy data inputs into a polished analytics experience

For recruiters and hiring managers, Clinical Outcomes Explorer shows both technical implementation and business framing in a way that mirrors real analytics product work.
