"""Data contract, split and leakage safeguards (TV3 checks, TV1/TV2 clean)."""
from .guards import assert_target_absent, assert_disjoint_ids, split_indices, inspect_split
__all__ = ['assert_target_absent', 'assert_disjoint_ids', 'split_indices', 'inspect_split']
