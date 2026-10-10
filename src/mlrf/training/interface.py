"""Simple shared model interface for notebook & experiment runner."""
from __future__ import annotations
import json
from pathlib import Path
from mlrf.config import repository_root
from mlrf.models import CustomRandomForestRegressor, build_decision_tree, build_dummy


def build_model(model_name: str, params: dict | None = None):
    params = dict(params or {})
    if model_name == 'dummy':
        return build_dummy(**params)
    if model_name == 'decision_tree':
        return build_decision_tree(**params)
    if model_name == 'custom_random_forest':
        return CustomRandomForestRegressor(**params)
    if model_name == 'sklearn_reference_rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    raise ValueError(f'Unsupported model name: {model_name!r}')


def fit_predict(model, X_train, y_train, X_eval):
    model.fit(X_train, y_train)
    return model, model.predict(X_eval)


def model_search_space(name: str, path: str | Path | None = None) -> dict:
    file = Path(path) if path is not None else repository_root() / 'configs' / 'model_search.json'
    config = json.loads(file.read_text(encoding='utf-8'))
    if name not in config:
        raise ValueError(f'Unknown search space: {name}')
    return config[name]
