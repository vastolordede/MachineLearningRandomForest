"""Safe split boundaries. Do not fit preprocessing on dev/test."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split


def assert_target_absent(X, target: str = 'price') -> None:
    if hasattr(X, 'columns') and target in X.columns:
        raise ValueError(f'Target leakage: {target} found in X')


def assert_disjoint_ids(train_ids, dev_ids, test_ids) -> None:
    ids = [list(x) for x in (train_ids, dev_ids, test_ids)]
    sets = [set(x) for x in ids]
    if any(len(s) != len(x) for s, x in zip(sets, ids)):
        raise ValueError('Duplicate row IDs within a split')
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Split leakage: IDs overlap across train/dev/test')


def split_indices(n_rows: int, *, train: float = .70, dev: float = .15,
                  test: float = .15, random_state: int = 42,
                  groups=None):
    """Return positional row IDs. Group-aware splits when repeat vehicle IDs exist.

    Group splitting requires stronger sampling constraints; no stratification is
    implied. TV4 owns choice of split policy, TV3 enforces boundaries.
    """
    if not n_rows >= 6:
        raise ValueError('At least six rows required for three-way split')
    if min(train, dev, test) <= 0 or not np.isclose(train + dev + test, 1.0):
        raise ValueError('Split proportions must be > 0 and sum to 1')
    idx = np.arange(n_rows)
    if groups is None:
        tr, remaining = train_test_split(idx, train_size=train, random_state=random_state)
        de, te = train_test_split(remaining, train_size=dev / (dev + test), random_state=random_state)
    else:
        from sklearn.model_selection import GroupShuffleSplit
        groups = np.asarray(groups)
        if len(groups) != n_rows:
            raise ValueError('groups length mismatch')
        s1 = GroupShuffleSplit(n_splits=1, train_size=train, random_state=random_state)
        tr_pos, rem_pos = next(s1.split(idx, groups=groups))
        tr, remaining = idx[tr_pos], idx[rem_pos]
        s2 = GroupShuffleSplit(n_splits=1, train_size=dev / (dev + test), random_state=random_state)
        de_pos, te_pos = next(s2.split(remaining, groups=groups[remaining]))
        de, te = remaining[de_pos], remaining[te_pos]
        assert not (set(groups[tr]) & set(groups[de]) | set(groups[tr]) & set(groups[te]) | set(groups[de]) & set(groups[te]))
    assert_disjoint_ids(tr, de, te)
    return np.sort(tr), np.sort(de), np.sort(te)


def inspect_split(train_ids, dev_ids, test_ids) -> dict:
    assert_disjoint_ids(train_ids, dev_ids, test_ids)
    return {'train_rows': len(train_ids), 'dev_rows': len(dev_ids), 'test_rows': len(test_ids)}
