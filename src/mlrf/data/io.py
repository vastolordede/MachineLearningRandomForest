"""Parquet handoff reader. No preprocessing learned from the dataset."""
from __future__ import annotations

from pathlib import Path
from mlrf.data.contract import validate_raw_columns


def load_cleaned_parquet(path: str | Path):
    """Read a cleaned DS04 table and check names (TV1/TV2 own its content)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before reading Parquet") from exc
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Parquet input does not exist: {source}")
    frame = pd.read_parquet(source)
    validate_raw_columns(frame.columns)
    if frame.empty:
        raise ValueError("Input dataset has no rows")
    return frame
