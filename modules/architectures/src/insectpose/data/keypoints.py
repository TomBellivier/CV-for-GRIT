"""Keypoint schemas and union space (CONVENTIONS.md §3.1).

The order of the points of a schema is FROZEN: it is encoded in every existing
artefact. Adding a point => append it at the end of the list and bump schema_version.

The schema of the project (`insect42_v1`) is not declared in this module: it is read
from kp_infos.yaml, at the repository root, the single definition shared with the
annotation, the measurement classifiers and the pipeline. `configs/keypoints/` is still
read first, for a study schema that would have nothing to do anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from insectpose.contracts import ContractError
from insectpose.paths import KP_INFOS_PATH


@dataclass(frozen=True)
class KeypointSchema:
    """Ordered definition of the points of a dataset (or of the union space)."""

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
        """'difficulty' if the sigmas are derived, 'explicit' if they are hard-coded."""
        return self._sigma_source

    @property
    def is_placeholder(self) -> bool:
        """True as long as the schema has not been validated by an expert (DECISION OPEN-01)."""
        return self.status.upper() == "PLACEHOLDER"

    def index(self, name: str) -> int:
        """Index of a point by its name."""
        return self.names.index(name)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Keypoint schema not found: {path}. The schema of the project is in "
            f"{KP_INFOS_PATH}; a study schema is declared in configs/keypoints/."
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def schema_file(name: str, configs_dir: Path) -> Path:
    """File declaring the schema `name`: configs/keypoints/<name>.yaml if it exists,
    else kp_infos.yaml (whose `name` is the schema of the project)."""
    local = Path(configs_dir) / "keypoints" / f"{name}.yaml"
    if local.exists() or not KP_INFOS_PATH.exists():
        return local
    with KP_INFOS_PATH.open(encoding="utf-8") as f:
        declared = (yaml.safe_load(f) or {}).get("name")
    return KP_INFOS_PATH if declared == name else local


@lru_cache(maxsize=32)
def load_schema(name: str, configs_dir: Path) -> KeypointSchema:
    """Load a keypoint schema (kp_infos.yaml, or configs/keypoints/<name>.yaml)."""
    raw = _load_yaml(schema_file(name, Path(configs_dir)))
    kpts = raw["keypoints"]
    names = tuple(k["name"] for k in kpts)
    if len(set(names)) != len(names):
        raise ContractError(f"[{name}] duplicated keypoint names: {names}")

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
                f"[{name}] point '{k['name']}' has neither 'sigma' nor ('difficulty' + "
                "sigma_from_difficulty.scale). An OKS without a defined tolerance makes no sense."
            )
        difficulty.append(float(k.get("difficulty", np.nan)))
    if len(sources) > 1:
        raise ContractError(
            f"[{name}] mixed sigmas (explicit and derived from the difficulty): choose a "
            "single source, otherwise the OKS tolerance can no longer be interpreted."
        )

    name_to_idx = {n: i for i, n in enumerate(names)}
    flip = []
    for k in kpts:
        partner = k.get("flip")
        if partner is None:
            flip.append(name_to_idx[k["name"]])
        elif partner not in name_to_idx:
            raise ContractError(f"[{name}] unknown flip '{partner}' (point '{k['name']}').")
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
        # Without a `union` key (kp_infos.yaml), a point is its own union equivalent.
        union_names=tuple(k["union"] if "union" in k else k["name"] for k in kpts),
        union_space=raw.get("union_space"),
        skeleton=tuple(tuple(e) for e in raw.get("skeleton", [])),
        _sigma_source=next(iter(sources)),
    )


def load_schemas(names: list[str], configs_dir: Path, strict: bool = False
                 ) -> dict[str, KeypointSchema]:
    """Load several schemas. `strict=True` refuses the PLACEHOLDER ones (§13)."""
    out = {n: load_schema(n, Path(configs_dir)) for n in names}
    if strict:
        placeholders = [n for n, s in out.items() if s.is_placeholder]
        if placeholders:
            raise ContractError(
                f"Keypoint schemas not validated (status=PLACEHOLDER): {placeholders}. "
                "DECISION OPEN-01 must be settled, or set strict.require_validated_"
                "keypoints=false for development."
            )
    return out


@dataclass(frozen=True)
class UnionMapping:
    """Local schema <-> union space correspondence, for the multi-dataset models."""

    local: KeypointSchema
    union: KeypointSchema
    local_to_union: np.ndarray   # (K_local,) union index or -1 if no equivalent
    union_to_local: np.ndarray   # (K_union,) local index or -1

    @property
    def masked_local(self) -> list[str]:
        """Local points without a union equivalent: masked in the loss, never set to zero."""
        pairs = zip(self.local.names, self.local.union_names, strict=True)
        return [name for name, union in pairs if union is None]


def build_union_mapping(local: KeypointSchema, union: KeypointSchema) -> UnionMapping:
    """Build the local <-> union correspondence and check its consistency.

    Nominal case of the project (ADR-0006): the 4 datasets share `insect42_v1`, so local
    is union and the mapping is the identity. The mechanism stays in place to absorb a
    future divergence between insect orders without a redesign.
    """
    if local.union_space is not None and local.union_space != union.name:
        raise ContractError(
            f"[{local.name}] declares union_space='{local.union_space}' but receives "
            f"'{union.name}'."
        )
    l2u = np.full(local.n_keypoints, -1, dtype=int)
    u2l = np.full(union.n_keypoints, -1, dtype=int)
    for i, uname in enumerate(local.union_names):
        if uname is None:
            continue
        if uname not in union.names:
            raise ContractError(
                f"[{local.name}] point '{local.names[i]}' points to '{uname}', missing from "
                f"the union space '{union.name}'."
            )
        j = union.index(uname)
        if u2l[j] != -1:
            raise ContractError(
                f"[{local.name}] two local points point to '{uname}': ambiguous "
                "correspondence, the union space must be disambiguated."
            )
        l2u[i] = j
        u2l[j] = i
    return UnionMapping(local=local, union=union, local_to_union=l2u, union_to_local=u2l)


def local_to_union(values: np.ndarray, mapping: UnionMapping, fill: float = np.nan) -> np.ndarray:
    """Project (..., K_local, C) to (..., K_union, C). The holes are `fill`."""
    arr = np.asarray(values, dtype=float)
    out = np.full((*arr.shape[:-2], mapping.union.n_keypoints, arr.shape[-1]), fill, dtype=float)
    sel = mapping.local_to_union >= 0
    out[..., mapping.local_to_union[sel], :] = arr[..., sel, :]
    return out


def union_to_local(values: np.ndarray, mapping: UnionMapping, fill: float = 0.0) -> np.ndarray:
    """Project (..., K_union, C) to (..., K_local, C).

    To be applied BEFORE writing the predictions of a multi-dataset model (§3.1):
    contract 3 imposes the local schema of the dataset.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full((*arr.shape[:-2], mapping.local.n_keypoints, arr.shape[-1]), fill, dtype=float)
    sel = mapping.union_to_local >= 0
    out[..., mapping.union_to_local[sel], :] = arr[..., sel, :]
    return out


def union_mask(mapping: UnionMapping) -> np.ndarray:
    """Boolean mask (K_union,) of the points actually supervised by this dataset."""
    return mapping.union_to_local >= 0
