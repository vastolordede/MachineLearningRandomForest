"""Write portable experiment manifests before any optional external sync."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
from typing import Any
import numpy as np


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def environment_info() -> dict:
    import sklearn, numpy, pandas
    return {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
            'numpy': numpy.__version__, 'pandas': pandas.__version__}


def save_run(root: str | Path, run_id: str, *, config: dict, metrics: dict,
             dataset: dict, git_commit: str | None = None,
             predictions=None, importance=None) -> Path:
    """Create one run folder, write metadata; never overwrite existing runs."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Unsafe run ID')
    dest = Path(root) / run_id
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {'run_id': run_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': dataset, 'git_commit': git_commit, 'environment': environment_info(),
                'files': ['config.json', 'metrics.json']}
    def write(name, value):
        (dest / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                            default=_json_default) + '\n', encoding='utf-8')
    write('config.json', config)
    write('metrics.json', metrics)
    if predictions is not None:
        import pandas as pd
        pd.DataFrame(predictions).to_csv(dest / 'predictions.csv', index=False)
        manifest['files'].append('predictions.csv')
    if importance is not None:
        import pandas as pd
        pd.DataFrame(importance).to_csv(dest / 'feature_importance.csv', index=False)
        manifest['files'].append('feature_importance.csv')
    write('manifest.json', manifest)
    return dest


def save_trusted_model(model: Any, path: str | Path) -> Path:
    """Persist a fitted model/pipeline from trusted project code only."""
    import joblib
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_trusted_model(path: str | Path):
    """WARNING: joblib/pickle can execute code. Only load trusted artifacts."""
    import joblib
    return joblib.load(path)
