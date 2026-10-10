"""One optimized CART tree (our elementary brick, not a hand-coded splitter)."""
from __future__ import annotations
from sklearn.tree import DecisionTreeRegressor


def build_decision_tree(*, random_state: int = 42, criterion: str = 'squared_error',
                        max_depth: int | None = 8, min_samples_split: int = 2,
                        min_samples_leaf: int = 1, max_features: int | float | str | None = None
                        ) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        random_state=random_state, criterion=criterion,
        max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
    )
