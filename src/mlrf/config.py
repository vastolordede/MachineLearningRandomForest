"""Central project configuration loader (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else repository_root() / "configs" / "project.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = config["split"]
    fractions = [float(split[key]) for key in ("train", "dev", "test")]
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("Each split fraction must be in (0,1)")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Train/dev/test split fractions must sum to 1")
    return config
