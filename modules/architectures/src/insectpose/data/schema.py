"""Validation of the data contracts (CONVENTIONS.md §3, §10).

Fails early, loudly, with an actionable message. No filtering, no automatic
correction: validating is not cleaning.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from insectpose.contracts import (
    BBOX_SOURCES,
    DATASETS,
    ROLES,
    SCHEMA_VERSIONS,
    SCHEMAS,
    ContractError,
    required_columns,
)

_LIST_KINDS = {"list_float", "list_int"}


def validate_frame(df: pd.DataFrame, artifact: str) -> None:
    """Validate a DataFrame against a contract ('annotations', 'splits', ...).

    Raises ContractError at the first structural problem.
    """
    if artifact not in SCHEMAS:
        raise ContractError(f"Unknown artefact: {artifact}. Known: {sorted(SCHEMAS)}")

    missing = [c for c in required_columns(artifact) if c not in df.columns]
    if missing:
        raise ContractError(
            f"[{artifact}] missing mandatory columns: {missing}. "
            f"Expected: {required_columns(artifact)}"
        )
    if df.empty:
        return

    expected_version = SCHEMA_VERSIONS[artifact]
    versions = set(pd.unique(df["schema_version"]))
    if versions != {expected_version}:
        raise ContractError(
            f"[{artifact}] schema_version={versions}, expected {expected_version}. "
            "An artefact of another version must go through a dedicated reader."
        )

    for spec in SCHEMAS[artifact]:
        if spec.name not in df.columns:
            continue
        col = df[spec.name]
        if col.isna().any() and spec.required:
            raise ContractError(f"[{artifact}] null values forbidden in '{spec.name}'.")
        if spec.kind in _LIST_KINDS:
            _check_list_column(artifact, spec.name, col)

    _validate_vocabulary(df, artifact)
    _validate_geometry(df, artifact)


def _check_list_column(artifact: str, name: str, col: pd.Series) -> None:
    """Check that a list column does contain numeric sequences."""
    sample = col.iloc[0]
    if not isinstance(sample, (list, tuple, np.ndarray)):
        raise ContractError(
            f"[{artifact}] '{name}' must contain lists, found {type(sample).__name__}."
        )


def _validate_vocabulary(df: pd.DataFrame, artifact: str) -> None:
    """Check the columns with a closed vocabulary."""
    if "dataset" in df.columns:
        unknown = set(df["dataset"].unique()) - set(DATASETS)
        if unknown:
            raise ContractError(
                f"[{artifact}] unknown datasets: {sorted(unknown)}. "
                f"Closed vocabulary: {list(DATASETS)} (see contracts.DATASETS)."
            )
    if artifact == "splits":
        unknown_roles = set(df["role"].unique()) - set(ROLES)
        if unknown_roles:
            raise ContractError(f"[splits] unknown roles: {sorted(unknown_roles)}")
    if artifact == "predictions":
        unknown_src = set(df["bbox_source"].unique()) - set(BBOX_SOURCES)
        if unknown_src:
            raise ContractError(f"[predictions] unknown bbox_source: {sorted(unknown_src)}")
        if df["pred_id"].duplicated().any():
            dup = df.loc[df["pred_id"].duplicated(), "pred_id"].head(3).tolist()
            raise ContractError(f"[predictions] pred_id not unique, e.g. {dup}")
    if artifact == "annotations" and df["instance_id"].duplicated().any():
        dup = df.loc[df["instance_id"].duplicated(), "instance_id"].head(3).tolist()
        raise ContractError(f"[annotations] instance_id not unique, e.g. {dup}")


def _validate_geometry(df: pd.DataFrame, artifact: str) -> None:
    """Check the bbox / keypoints / visibility consistency."""
    if artifact not in ("annotations", "predictions"):
        return
    bad_bbox = df["bbox_xywh"].map(lambda b: len(b) != 4)
    if bad_bbox.any():
        raise ContractError(f"[{artifact}] bbox_xywh must contain 4 values (xywh).")

    # This check comes FIRST: it names the faulty files, where the next ones only report
    # sizes. On real data, this difference is what saves a manual investigation.
    _validate_schema_consistency(df, artifact)

    n_kpts = df["kpts_xy"].map(len)
    if (n_kpts % 2 != 0).any():
        raise ContractError(f"[{artifact}] kpts_xy must contain 2K values.")
    k = (n_kpts // 2).astype(int)

    second = "kpts_vis" if artifact == "annotations" else "kpts_score"
    n_second = df[second].map(len).astype(int)
    if not (n_second == k).all():
        raise ContractError(
            f"[{artifact}] length of '{second}' inconsistent with kpts_xy "
            f"(K derived={sorted(set(k))[:3]}, found={sorted(set(n_second))[:3]})."
        )


def _validate_schema_consistency(df: pd.DataFrame, artifact: str) -> None:
    """Refuse a keypoint schema showing several sizes, naming the faulty ones.

    The order and the number of points are frozen for life (ADR-0006): a divergence
    signals labels of another schema, or a malformed file.
    """
    sizes = df["kpts_xy"].map(len)
    for schema_name, group in df.groupby("keypoint_schema"):
        counts = sizes[group.index].value_counts()
        if len(counts) <= 1:
            continue
        # The majority size is presumed correct; the ones deviating from it are named.
        expected = int(counts.idxmax())
        outliers = group[sizes[group.index] != expected]
        column = "image_path" if "image_path" in outliers.columns else "instance_id"
        examples = "\n  ".join(
            f"{row[column]}: {len(row['kpts_xy']) // 2} keypoints"
            for _, row in outliers.head(5).iterrows()
        )
        raise ContractError(
            f"[{artifact}] the schema '{schema_name}' appears with several keypoint "
            f"sizes: {sorted(int(v) // 2 for v in counts.index)} points. The order and "
            f"the number of points are frozen.\n"
            f"Expected {expected // 2} points ({int(counts.max())} instances). "
            f"{len(outliers)} diverging instance(s), for example:\n  {examples}"
        )


def validate_single_instance(df: pd.DataFrame) -> None:
    """Check the "one image = one insect" hypothesis (ADR-0017).

    If it is violated, the top-1 detection of the approaches silently becomes wrong: the
    failure must therefore be blocking, at data preparation time.
    """
    counts = df.groupby("image_id").size()
    offenders = counts[counts > 1]
    if len(offenders):
        raise ContractError(
            f"{len(offenders)} image(s) contain several instances "
            f"(e.g. {offenders.index[0]}: {int(offenders.iloc[0])}), whereas "
            "data.single_instance_per_image=true (ADR-0017). Fix the annotations or set "
            "this flag to false and review the top-1 detection approaches."
        )


def validate_coordinates_in_image(df: pd.DataFrame, tolerance: float = 0.05) -> pd.Series:
    """Flag (without deleting) the instances whose coordinates leave the image.

    Returns a Series of flags; filtering remains a config decision (§3.2).
    """
    flags = []
    for row in df.itertuples(index=False):
        w, h = float(row.image_width), float(row.image_height)
        pts = np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2)
        vis = np.asarray(row.kpts_vis) > 0
        issues: list[str] = []
        if vis.any():
            p = pts[vis]
            if (p[:, 0] < -tolerance * w).any() or (p[:, 0] > (1 + tolerance) * w).any():
                issues.append("kpt_x_out_of_image")
            if (p[:, 1] < -tolerance * h).any() or (p[:, 1] > (1 + tolerance) * h).any():
                issues.append("kpt_y_out_of_image")
        else:
            issues.append("no_visible_keypoint")
        bw, bh = float(row.bbox_xywh[2]), float(row.bbox_xywh[3])
        if bw <= 0 or bh <= 0:
            issues.append("degenerate_bbox")
        flags.append(";".join(issues))
    return pd.Series(flags, index=df.index, dtype="object")


def ensure_columns(df: pd.DataFrame, artifact: str, extra: dict[str, Any] | None = None
                   ) -> pd.DataFrame:
    """Add the missing optional columns and order the columns as in the contract."""
    out = df.copy()
    if extra:
        for key, value in extra.items():
            out[key] = value
    out["schema_version"] = SCHEMA_VERSIONS[artifact]
    for spec in SCHEMAS[artifact]:
        if spec.name not in out.columns and not spec.required:
            out[spec.name] = "" if spec.kind == "str" else np.nan
    ordered: Sequence[str] = [c.name for c in SCHEMAS[artifact] if c.name in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[[*ordered, *rest]]