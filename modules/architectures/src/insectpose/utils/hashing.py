"""Stable hashing for the identity of the runs and the invalidation of the splits (§6.4, §3.3)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(obj: Any) -> str:
    """blake2b hash of a JSON-serialisable object, insensitive to the order of the keys."""
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def short_hash(value: str, length: int = 8) -> str:
    """Short prefix of a hash, for readable identifiers."""
    return value[:length]


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hash of the content of a file."""
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def content_hash_annotations(df: pd.DataFrame) -> str:
    """Fingerprint of the annotations used by a split or a run.

    Any change of the data (addition, removal, re-annotation) changes this value and so
    invalidates the splits referring to it (§3.3).
    """
    cols = [c for c in ("dataset", "image_id", "instance_id", "group_id") if c in df.columns]
    key = df[cols].sort_values(cols).astype(str).agg("|".join, axis=1)
    digest = hashlib.blake2b(digest_size=16)
    for row in key:
        digest.update(row.encode("utf-8"))
    digest.update(str(len(df)).encode("utf-8"))
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> str:
    """Hash of the (name, size) set of a list of source files."""
    items = sorted(
        {"name": p.name, "size": p.stat().st_size} for p in paths if p.exists()
    )  # type: ignore[type-var]
    return stable_hash(items)
