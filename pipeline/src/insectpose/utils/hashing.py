"""Hachage stable pour l'identite des runs et l'invalidation des splits (§6.4, §3.3)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(obj: Any) -> str:
    """Hash blake2b d'un objet JSON-serialisable, insensible a l'ordre des cles."""
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def short_hash(value: str, length: int = 8) -> str:
    """Prefixe court d'un hash, pour les identifiants lisibles."""
    return value[:length]


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hash du contenu d'un fichier."""
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def content_hash_annotations(df: pd.DataFrame) -> str:
    """Empreinte des annotations utilisees par un decoupage ou un run.

    Toute modification des donnees (ajout, retrait, re-annotation) change cette valeur
    et invalide donc les splits qui s'y referent (§3.3).
    """
    cols = [c for c in ("dataset", "image_id", "instance_id", "group_id") if c in df.columns]
    key = df[cols].sort_values(cols).astype(str).agg("|".join, axis=1)
    digest = hashlib.blake2b(digest_size=16)
    for row in key:
        digest.update(row.encode("utf-8"))
    digest.update(str(len(df)).encode("utf-8"))
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> str:
    """Hash de l'ensemble (nom, taille, mtime) d'une liste de fichiers sources."""
    items = sorted(
        {"name": p.name, "size": p.stat().st_size} for p in paths if p.exists()
    )  # type: ignore[type-var]
    return stable_hash(items)
