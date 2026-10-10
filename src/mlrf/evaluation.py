"""TV3 metric utility; experiment comparisons/interpretation belong to TV4."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError('Prediction length mismatch or empty arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite target or prediction')
    return {'mae': float(mean_absolute_error(a, b)),
            'rmse': float(np.sqrt(mean_squared_error(a, b))),
            'r2': float(r2_score(a, b)) if len(a) >= 2 else float('nan')}
