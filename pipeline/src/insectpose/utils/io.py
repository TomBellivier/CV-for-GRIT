"""Lecture / ecriture des artefacts. Ecritures atomiques, validation optionnelle.

Toute ecriture passe ici : cela garantit qu'un fichier partiellement ecrit n'est
jamais visible et que les contrats sont valides au meme endroit (§10).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Ecrit un JSON de facon atomique. Effet de bord : cree `path` et ses parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)  # noqa: PTH105  # remplacement atomique
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Lit un JSON. Echoue si absent (jamais de valeur de repli silencieuse)."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier attendu introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def write_parquet(path: Path, df: pd.DataFrame, artifact: str | None = None,
                  validate: bool = True) -> Path:
    """Ecrit un parquet de facon atomique, apres validation du contrat si demande.

    Effet de bord : cree `path` et ses parents.
    """
    if artifact is not None and validate:
        from insectpose.data.schema import validate_frame

        validate_frame(df, artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # noqa: PTH105  # remplacement atomique
    return path


def read_parquet(path: Path, artifact: str | None = None, validate: bool = False) -> pd.DataFrame:
    """Lit un parquet, avec validation de contrat optionnelle."""
    if not path.exists():
        raise FileNotFoundError(f"Artefact attendu introuvable : {path}")
    df = pd.read_parquet(path)
    if artifact is not None and validate:
        from insectpose.data.schema import validate_frame

        validate_frame(df, artifact)
    return df


def purge_incomplete_runs(runs_dir: Path, dry_run: bool = True) -> list[str]:
    """Liste (et supprime si dry_run=False) les runs sans manifeste (§8.2)."""
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
