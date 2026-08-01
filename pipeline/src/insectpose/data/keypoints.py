"""Schemas de keypoints et espace union (CONVENTIONS.md §3.1).

L'ordre des points d'un schema est FIGE : il est encode dans tous les artefacts
existants. Ajouter un point => l'ajouter en fin de liste et bumper schema_version.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from insectpose.contracts import ContractError


@dataclass(frozen=True)
class KeypointSchema:
    """Definition ordonnee des points d'un dataset (ou de l'espace union)."""

    name: str
    schema_version: int
    kind: str
    status: str
    names: tuple[str, ...]
    sigmas: np.ndarray
    difficulty: np.ndarray
    flip_index: tuple[int, ...]
    union_names: tuple[str | None, ...]
    union_space: str | None
    skeleton: tuple[tuple[int, int], ...]
    _sigma_source: str = "explicit"

    @property
    def n_keypoints(self) -> int:
        return len(self.names)

    @property
    def sigma_source(self) -> str:
        """'difficulty' si les sigmas sont derives, 'explicit' s'ils sont ecrits en dur."""
        return self._sigma_source

    @property
    def is_placeholder(self) -> bool:
        """True tant que le schema n'a pas ete valide par un expert (DECISION OPEN-01)."""
        return self.status.upper() == "PLACEHOLDER"

    def index(self, name: str) -> int:
        """Index d'un point par son nom."""
        return self.names.index(name)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Schema de keypoints introuvable : {path}. "
            "Chaque dataset doit declarer son schema dans configs/keypoints/."
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=32)
def load_schema(name: str, configs_dir: Path) -> KeypointSchema:
    """Charge un schema de keypoints depuis configs/keypoints/<name>.yaml."""
    raw = _load_yaml(Path(configs_dir) / "keypoints" / f"{name}.yaml")
    kpts = raw["keypoints"]
    names = tuple(k["name"] for k in kpts)
    if len(set(names)) != len(names):
        raise ContractError(f"[{name}] noms de keypoints dupliques : {names}")

    scale = float((raw.get("sigma_from_difficulty") or {}).get("scale", 0.0))
    sigmas, difficulty, sources = [], [], set()
    for k in kpts:
        if "sigma" in k:
            sigmas.append(float(k["sigma"]))
            sources.add("explicit")
        elif "difficulty" in k and scale > 0:
            sigmas.append(float(k["difficulty"]) * scale)
            sources.add("difficulty")
        else:
            raise ContractError(
                f"[{name}] le point '{k['name']}' n'a ni 'sigma' ni ('difficulty' + "
                "sigma_from_difficulty.scale). Un OKS sans tolerance definie n'a pas de sens."
            )
        difficulty.append(float(k.get("difficulty", np.nan)))
    if len(sources) > 1:
        raise ContractError(
            f"[{name}] sigmas mixtes (explicites et derives de la difficulte) : choisir une "
            "seule source, sinon la tolerance OKS n'est plus interpretable."
        )

    name_to_idx = {n: i for i, n in enumerate(names)}
    flip = []
    for k in kpts:
        partner = k.get("flip")
        if partner is None:
            flip.append(name_to_idx[k["name"]])
        elif partner not in name_to_idx:
            raise ContractError(f"[{name}] flip '{partner}' inconnu (point '{k['name']}').")
        else:
            flip.append(name_to_idx[partner])

    return KeypointSchema(
        name=raw["name"],
        schema_version=int(raw["schema_version"]),
        kind=raw.get("kind", "dataset_schema"),
        status=str(raw.get("status", "VALIDATED")),
        names=names,
        sigmas=np.asarray(sigmas, dtype=float),
        difficulty=np.asarray(difficulty, dtype=float),
        flip_index=tuple(flip),
        union_names=tuple(k.get("union") for k in kpts),
        union_space=raw.get("union_space"),
        skeleton=tuple(tuple(e) for e in raw.get("skeleton", [])),
        _sigma_source=next(iter(sources)),
    )


def load_schemas(names: list[str], configs_dir: Path, strict: bool = False
                 ) -> dict[str, KeypointSchema]:
    """Charge plusieurs schemas. `strict=True` refuse les PLACEHOLDER (§13.1)."""
    out = {n: load_schema(n, Path(configs_dir)) for n in names}
    if strict:
        placeholders = [n for n, s in out.items() if s.is_placeholder]
        if placeholders:
            raise ContractError(
                f"Schemas de keypoints non valides (status=PLACEHOLDER) : {placeholders}. "
                "DECISION OPEN-01 doit etre tranchee, ou passer strict.require_validated_"
                "keypoints=false pour du developpement."
            )
    return out


@dataclass(frozen=True)
class UnionMapping:
    """Correspondance schema local <-> espace union, pour les modeles multi-datasets."""

    local: KeypointSchema
    union: KeypointSchema
    local_to_union: np.ndarray   # (K_local,) index union ou -1 si aucun equivalent
    union_to_local: np.ndarray   # (K_union,) index local ou -1

    @property
    def masked_local(self) -> list[str]:
        """Points locaux sans equivalent union : masques dans la loss, jamais mis a zero."""
        pairs = zip(self.local.names, self.local.union_names, strict=True)
        return [name for name, union in pairs if union is None]


def build_union_mapping(local: KeypointSchema, union: KeypointSchema) -> UnionMapping:
    """Construit la correspondance locale <-> union et verifie sa coherence.

    Cas nominal du projet (ADR-0006) : les 4 datasets partagent `insect42_v1`, donc
    local is union et le mapping est l'identite. Le mecanisme reste en place pour
    absorber sans refonte une divergence future entre ordres d'insectes.
    """
    if local.union_space is not None and local.union_space != union.name:
        raise ContractError(
            f"[{local.name}] declare union_space='{local.union_space}' mais recoit "
            f"'{union.name}'."
        )
    l2u = np.full(local.n_keypoints, -1, dtype=int)
    u2l = np.full(union.n_keypoints, -1, dtype=int)
    for i, uname in enumerate(local.union_names):
        if uname is None:
            continue
        if uname not in union.names:
            raise ContractError(
                f"[{local.name}] le point '{local.names[i]}' pointe vers '{uname}', absent de "
                f"l'espace union '{union.name}'."
            )
        j = union.index(uname)
        if u2l[j] != -1:
            raise ContractError(
                f"[{local.name}] deux points locaux pointent vers '{uname}' : correspondance "
                "ambigue, l'espace union doit etre desambiguise."
            )
        l2u[i] = j
        u2l[j] = i
    return UnionMapping(local=local, union=union, local_to_union=l2u, union_to_local=u2l)


def local_to_union(values: np.ndarray, mapping: UnionMapping, fill: float = np.nan) -> np.ndarray:
    """Projette (..., K_local, C) vers (..., K_union, C). Les trous valent `fill`."""
    arr = np.asarray(values, dtype=float)
    out = np.full((*arr.shape[:-2], mapping.union.n_keypoints, arr.shape[-1]), fill, dtype=float)
    sel = mapping.local_to_union >= 0
    out[..., mapping.local_to_union[sel], :] = arr[..., sel, :]
    return out


def union_to_local(values: np.ndarray, mapping: UnionMapping, fill: float = 0.0) -> np.ndarray:
    """Projette (..., K_union, C) vers (..., K_local, C).

    A appliquer AVANT l'ecriture des predictions d'un modele multi-datasets (§3.1) :
    le contrat 3 impose le schema local du dataset.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full((*arr.shape[:-2], mapping.local.n_keypoints, arr.shape[-1]), fill, dtype=float)
    sel = mapping.union_to_local >= 0
    out[..., mapping.union_to_local[sel], :] = arr[..., sel, :]
    return out


def union_mask(mapping: UnionMapping) -> np.ndarray:
    """Masque booleen (K_union,) des points reellement supervises par ce dataset."""
    return mapping.union_to_local >= 0
