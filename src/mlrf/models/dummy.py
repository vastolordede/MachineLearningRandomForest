"""Baseline regression brick. TV3 owns code, TV4 owns experimental scores."""
from __future__ import annotations
from sklearn.dummy import DummyRegressor


def build_dummy(*, strategy: str = 'median') -> DummyRegressor:
    if strategy not in {'mean', 'median'}:
        raise ValueError('strategy must be mean or median')
    return DummyRegressor(strategy=strategy)
