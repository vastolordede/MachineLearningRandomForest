# Model workflow (TV3) — Kaggle & local

## Assumptions
- Canonical code: GitHub, patch ZIPs applied to local repo, then Git review/commit/PR.
- Dataset candidate DS04 `imbalet/used_cars`, **provisionally chosen**, actual quality audit D10/D12 pending.
- `price` is regression target; currency still **unverified**.
- Input: cleaned Parquet from Thảo/Sang. Do not commit raw/cleaned files.
- Full-preprocessing belongs to Sang; TV3 provides only contract/checks and a clearly labeled notebook adapter, not final cleaned data.
- Random forest is **custom ensemble** wrapping optimized `sklearn.tree.DecisionTreeRegressor` (not reimplementing splits). `sklearn.ensemble.RandomForestRegressor` serves only as optional reference.

## Tests

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src
```

These tests run on synthetic fixtures; they DO NOT validate production data, Kaggle runtime or actual monetary predictions.

## Kaggle source/data attachment
1. Generate a **versioned source package** of the repo's `src/` and `configs/` (or use a pinned Git commit when internet available).
2. Attach that package to Kaggle Notebook through `Add Input`, and set `SOURCE_ROOT` (folder containing `mlrf/`).
3. Attach an approved cleaned Parquet dataset; set `DATA_PATH` to actual attached path.
4. Review `schema.json` and run D12 schema audit before using data; do not infer currency or fit imputer/encoder on full data.
5. Notebook `01_training_smoke.ipynb` runs synthetic smoke by default. Set `USE_APPROVED_DATA=True` only after TV1/TV2 handoff and D13 split approval.
6. Keep all artifact files under `/kaggle/working/mlrf_runs/`, save Kaggle version with outputs; sync selected model/run files to persistent store after successful integration.

## Why session files are not backups
Kaggle's runtime filesystem is ephemeral. Save a notebook version and keep an externally recoverable copy of selected outputs. Experiment manifest includes model config, dataset source/version and Git commit. W&B was discussed but **is not integrated in this patch**. No credentials should ever go in source or notebook output.

## Dataset conversion and split
`split_indices` uses reproducible train/dev/test positions, default 70/15/15 and seed 42 as **provisional** only. If the same vehicle can have repeat listings, use group IDs and `groups=`. Test is not used to tune/select feature/model. Preprocessing must be fitted only on train. If an outlier rule learns from distribution, learn from train as well.

## Custom forest
`CustomRandomForestRegressor.fit(X, y)` draws a bootstrap sample for each tree and sets `DecisionTreeRegressor(max_features=...)` so random feature selection happens at each split. `predict` averages trees. OOB is a NaN-masked mean of trees that never trained on each row; `oob_score_` is R² over covered rows, NaN if too few covered. OOB is valid only with bootstrap=True. It is **not identical to sklearn RandomForestRegressor** in random number generation or all options.

## Model interface

```python
from mlrf.training import build_model, fit_predict, model_search_space
m = build_model('custom_random_forest', {'n_estimators': 30, 'max_features': 0.7, 'random_state': 42})
fitted, predictions = fit_predict(m, X_train, y_train, X_dev)
search = model_search_space('custom_random_forest')
```

`build_model('sklearn_reference_rf', params)` is allowed for algorithm sanity checks, not the student's main implementation.

## Safe model persistence
Save a *fully fitted preprocessing+regressor pipeline* after dataset contract is frozen (O05/O09). `joblib.load` must be used only on trusted local artifacts because pickle-based formats can run code on load. Keep compatible library versions; record version in manifest.
