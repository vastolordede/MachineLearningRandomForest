"""Model training interfaces; experiment selection remains TV4's task."""
from .interface import build_model, fit_predict, model_search_space
__all__ = ['build_model', 'fit_predict', 'model_search_space']
