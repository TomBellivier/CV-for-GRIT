"""Reading / writing of the artefacts. Atomic writes, optional validation.

Every write goes through here: it guarantees that a partially written file is never
visible and that the contracts are validated in a single place (§10).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file atomically. Side effect: creates `path` and its parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)  # noqa: PTH105  # atomic replacement
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file. Fails if missing (never a silent fallback value)."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def write_parquet(path: Path, df: pd.DataFrame, artifact: str | None = None,
                  validate: bool = True) -> Path:
    """Write a parquet file atomically, after validating the contract if asked.

    Side effect: creates `path` and its parents.
    """
    if artifact is not None and validate:
        from insectpose.data.schema import validate_frame

        validate_frame(df, artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # noqa: PTH105  # atomic replacement
    return path


def read_parquet(path: Path, artifact: str | None = None, validate: bool = False) -> pd.DataFrame:
    """Read a parquet file, with an optional contract validation."""
    if not path.exists():
        raise FileNotFoundError(f"Expected artefact not found: {path}")
    df = pd.read_parquet(path)
    if artifact is not None and validate:
        from insectpose.data.schema import validate_frame

        validate_frame(df, artifact)
    return df


def purge_incomplete_runs(runs_dir: Path, dry_run: bool = True) -> list[str]:
    """List (and delete if dry_run=False) the runs without a manifest (§8.2)."""
    import shutil

    victims: list[str] = []
    if not runs_dir.exists():
        return victims
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name == "optuna":
            continue
        if not (d / "manifest.json").exists():
            victims.append(d.name)
            if not dry_run:
                shutil.rmtree(d)
    return victims
