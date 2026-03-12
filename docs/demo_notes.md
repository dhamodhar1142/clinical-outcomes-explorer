# Demo Dataset Notes

## Best demo dataset characteristics

The strongest walkthroughs usually include:

- an identifier such as `patient_id` or `encounter_id`
- a date or year field
- age
- BMI or another risk signal
- diagnosis, service, or department grouping
- at least one outcome-like field
- cost or utilization context if available

## What unlocks the most modules

- `event_date` or year -> trend analysis, monitoring, freshness, temporal views
- `patient_id` + dates -> readmission and encounter-style analytics
- diagnosis or service group -> clinical segmentation and pathway review
- cost field -> financial analytics and spend benchmarking
- outcome field -> survival, risk, fairness, and decision-support modules

## Synthetic helper behavior

When native fields are incomplete, the platform may add deterministic helper fields such as:

- synthetic `event_date`
- remediated BMI
- synthetic `estimated_cost`
- derived diagnosis labels
- synthetic readmission support

These are clearly disclosed in lineage, exports, and readiness summaries.

## Good demo paths

- `Healthcare Operations Demo`
  - best for healthcare intelligence, governance, and readmission-style walkthroughs
- `Hospital Reporting Demo`
  - best for standards, reporting, and structured operational summaries
- `Generic Business Demo`
  - best for schema-flexible profiling and governance readiness without strong healthcare assumptions

## Stronger source data would improve

- readmission and longitudinal analysis with encounter-level dates
- cost analytics with native financial fields
- diagnosis/procedure analysis with real structured clinical groupings
- interoperability review with coded terminology and encounter/patient structures
