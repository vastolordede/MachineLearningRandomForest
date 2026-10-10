"""Custom bagged random-feature forest assembled from sklearn CART trees.

This implements bootstrap sampling, per-tree random seeds, aggregation and OOB
predictions. CART splitting itself is delegated to DecisionTreeRegressor.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CustomRandomForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, *, n_estimators: int = 100, criterion: str = 'squared_error',
                 max_depth: int | None = 8, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int | float | str | None = 1.0,
                 bootstrap: bool = True, max_samples: int | float | None = None,
                 oob_score: bool = False, random_state: int | None = 42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.random_state = random_state

    def _sample_size(self, n: int) -> int:
        if not self.bootstrap:
            if self.max_samples is not None:
                raise ValueError('max_samples requires bootstrap=True')
            return n
        value = self.max_samples
        if value is None:
            return n
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            if not 1 <= value <= n:
                raise ValueError('integer max_samples must be in [1, n_samples]')
            return int(value)
        if isinstance(value, (float, np.floating)) and 0 < value <= 1:
            return max(1, round(n * value))
        raise ValueError('max_samples must be an integer count or float in (0,1]')

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csc', y_numeric=True)
        if not isinstance(self.n_estimators, (int, np.integer)) or isinstance(self.n_estimators, bool) or self.n_estimators < 1:
            raise ValueError('n_estimators must be a positive integer')
        if self.oob_score and not self.bootstrap:
            raise ValueError('OOB prediction requires bootstrap=True')
        n_samples, self.n_features_in_ = X.shape
        sample_size = self._sample_size(n_samples)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.bootstrap_indices_ = []
        oob_sum = np.zeros(n_samples, dtype=float)
        oob_counts = np.zeros(n_samples, dtype=int)
        for _ in range(self.n_estimators):
            sampled = (rng.integers(0, n_samples, sample_size)
                       if self.bootstrap else np.arange(n_samples))
            tree = DecisionTreeRegressor(
                criterion=self.criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[sampled], y[sampled])
            self.estimators_.append(tree)
            self.bootstrap_indices_.append(sampled)
            if self.oob_score:
                present = np.zeros(n_samples, dtype=bool)
                present[sampled] = True
                oob = np.flatnonzero(~present)
                if len(oob):
                    oob_sum[oob] += tree.predict(X[oob])
                    oob_counts[oob] += 1
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0)
        if self.oob_score:
            mask = oob_counts > 0
            self.oob_counts_ = oob_counts
            self.oob_prediction_ = np.full(n_samples, np.nan)
            self.oob_prediction_[mask] = oob_sum[mask] / oob_counts[mask]
            if mask.sum() < 2:
                self.oob_score_ = float('nan')
                warnings.warn('Fewer than two observations received OOB predictions', RuntimeWarning)
            else:
                self.oob_score_ = float(r2_score(y[mask], self.oob_prediction_[mask]))
                if (~mask).any():
                    warnings.warn('Some training rows have no OOB predictions', RuntimeWarning)
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse='csr')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        # An online mean avoids storing n_trees x n_rows dense predictions.
        mean = np.zeros(X.shape[0], dtype=float)
        for index, tree in enumerate(self.estimators_, start=1):
            mean += (tree.predict(X) - mean) / index
        return mean
