# MachineLearningRandomForest

Final-term ML project: used-car price prediction with Decision Tree Regression and a **custom Random Forest orchestration** built on `sklearn.tree.DecisionTreeRegressor` (the component-level tree builder). No model implementation or real training is included in the initial scaffold patch.

## Responsibilities / Phân công

- TV1 Thảo — dataset acquisition, numeric cleaning, quality and numeric EDA.
- TV2 Sang — categorical cleaning, feature engineering and preprocessing.
- TV3 Đăng — model implementation, tests, ML Ops and Git/Kaggle integration.
- TV4 Danh — experiments, tuning, evaluation and result analysis.

## Initial dataset (provisional DS04)

- Source: https://huggingface.co/datasets/imbalet/used_cars
- Target: `price`
- Source includes 21 columns and about 432k rows.
- Source CSV, Kaggle handoff expected as versioned **Parquet**.
- Schema snapshot: `configs/ds04_schema.json`.
- **Currency has not been verified.** Source contains Russian locations; do not label metrics in USD.
- The dataset has no authoritative train/dev/test separation, despite its Hugging Face viewer calling its sole split `train`.
- Dataset selection/audit (D10/D12) and split policy (D13) still require group approval; no checklist item for data/model is complete merely because this project compiles.

## Setup on Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src
```

Smoke tests above use only Python's standard library and can also run without installing dependencies.

## Kaggle

Start with `notebooks/00_environment_check.ipynb`. The notebook validates the runtime and prints available Kaggle input mounts; it **does not download/upload data or train models**. Later, attach a pinned-version Parquet dataset and matching source-code snapshot to Kaggle. Keep outputs in `/kaggle/working/` then Save & Run All before the session expires. Persist selected artifacts to the chosen experiment store only after integration is tested.

## Workflow

- Keep `work/apply_patch.py` and patch ZIPs in local-only `work/`.
- One logical feature cluster → one branch → one main commit → one PR to `dev`.
- Create `main`/`dev` bootstrap for an initially empty GitHub repo; then branch from latest `dev`.
- Never commit secrets, raw data, trained artifacts, notebooks' private tokens, or `work/`.
- Checklist items tick only after code, tests and merged PR with commit hash.

See `docs/DATA_HANDOFF.md` and `docs/WORKFLOW.md`.
