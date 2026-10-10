"""Validate DS04 *raw* column names; does not clean or fit any data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from mlrf.config import repository_root


def load_ds04_contract(path: str | Path | None = None) -> dict:
    source = Path(path) if path is not None else repository_root() / "configs" / "ds04_schema.json"
    contract = json.loads(source.read_text(encoding="utf-8"))
    numeric = contract["numeric_columns"]
    categorical = contract["categorical_columns"]
    all_cols = numeric + categorical
    if len(all_cols) != len(set(all_cols)):
        raise ValueError("Schema has duplicate column names")
    if contract["target"] not in numeric:
        raise ValueError("Target must be in numeric columns")
    return contract


def validate_raw_columns(columns: Iterable[str], *, strict: bool = False) -> None:
    expected = set(load_ds04_contract()["numeric_columns"] + load_ds04_contract()["categorical_columns"])
    actual = list(columns)
    if len(actual) != len(set(actual)):
        raise ValueError("Dataset has duplicated column names")
    missing = sorted(expected - set(actual))
    unexpected = sorted(set(actual) - expected)
    if missing or (strict and unexpected):
        raise ValueError(f"DS04 columns mismatch: missing={missing}, unexpected={unexpected}")
