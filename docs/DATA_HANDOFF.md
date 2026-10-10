# DS04 Data Contract (D12 draft — NOT YET APPROVED)

Dataset: https://huggingface.co/datasets/imbalet/used_cars
Repository revision pinned in `configs/project.json`. Reference column snapshot `configs/ds04_schema.json`.

## Source fields (21)

Numeric candidates (8): `owners`, `year`, `price`, `mileage`, `power`, `displacement`, `super_gen_year_from`, `super_gen_year_to`.

Categorical/text candidates (13): `region`, `mark`, `model`, `complectation`, `steering_wheel`, `gear_type`, `engine`, `transmission`, `characteristics`, `color`, `body_type_type`, `body_type_name`, `super_gen_name`.

- Target: `price`. Unit/currency: **pending audit**. Prices look RUB-denominated; verify upstream provenance before final report.
- Russian text is valid Unicode and must not be stripped during ingestion.
- No VIN/listing ID column is visible in the source schema; duplicates and near-duplicates require a documented assessment.
- Displacement unit requires verification. No automatic conversion is made in this patch.
- Source viewer exposes one split named `train`: it is **not** the project's actual Train/Dev/Test split.

## TV1 → TV2 → TV3 handoff

`vehicles_cleaned_v1.parquet`: typed, row-preserving cleaned data. Keep categorical strings and numeric fields **unencoded**. Track data version and source revision. Do not automatically remove extreme values solely for being extreme.

`schema.json`: explicit target name, currency, columns, feature roles, types, units, null policy and data version.

`cleaning_report.json`: source rows, duplicates, invalid rows, missingness, decisions, dropped rows and reasons.

`preprocessing.py`: fitting preprocessing logic. Fit imputers/encoders/learned thresholds ONLY on Train after Train/Dev/Test split. Dev/Test use transform only; target never enters X. If deduplication uses a record key, agree it before splitting.

This first patch supplies only a **draft raw schema**, not a fully cleaned Parquet file and not an approved preprocessing pipeline.
