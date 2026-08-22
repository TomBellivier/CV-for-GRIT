"""Loading and assembly of the measurement-validity dataset.

The pipeline is:

1. read the human annotations (one CSV per annotation batch in ``data/``),
2. attach the taxonomic group by looking up the image stem in the image
   database,
3. read the pose-estimation outputs (confidences, keypoint positions,
   measurements in pixels),
4. merge both on the image name.

No scaling, no imputation and no dimensionality reduction happen here: every
transformation that learns something from the data belongs inside the
cross-validation pipeline, otherwise it leaks test information into training.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from insect_anatomy import INSECT_GROUPS, MEASUREMENTS, POINTS

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".tiff")

STATUS_SUFFIX = "_status"
MEASURABLE_LABEL = "measurable"

#: Substrings identifying each family of pose-model outputs.
CONF_TOKEN = "kp_conf"
X_TOKEN = "kp_x"
Y_TOKEN = "kp_y"
MEAS_TOKEN = "px"


@dataclass
class ColumnIndex:
    """Resolved mapping from anatomical names to actual dataframe columns.

    Column naming conventions differ between exports (``kp_conf[head-top]``,
    ``head-top_kp_conf``, ...), so columns are resolved by searching for both
    the family token and the entity name rather than by hard-coded patterns.
    """

    conf: Dict[str, str] = field(default_factory=dict)
    x: Dict[str, str] = field(default_factory=dict)
    y: Dict[str, str] = field(default_factory=dict)
    measure: Dict[str, str] = field(default_factory=dict)
    status: Dict[str, str] = field(default_factory=dict)

    @property
    def complete_points(self) -> List[str]:
        """Keypoints for which position *and* confidence were both found."""
        return [p for p in POINTS if p in self.x and p in self.y and p in self.conf]

    def summary(self) -> str:
        return (
            f"{len(self.conf)}/{len(POINTS)} conf, "
            f"{len(self.x)}/{len(POINTS)} x, "
            f"{len(self.y)}/{len(POINTS)} y, "
            f"{len(self.measure)}/{len(MEASUREMENTS)} measurements, "
            f"{len(self.status)}/{len(MEASUREMENTS)} statuses"
        )


def resolve_columns(columns: Iterable[str]) -> ColumnIndex:
    """Locate the pose-output column matching each keypoint and measurement.

    Raises a warning (not an error) for every entity that could not be resolved
    or that resolved ambiguously, so that a partial export still runs.
    """
    columns = list(columns)
    index = ColumnIndex()

    def _find(candidates: Sequence[str], entity: str, family: str) -> Optional[str]:
        matches = [c for c in candidates if entity in c]
        if not matches:
            return None
        if len(matches) > 1:
            # Prefer the shortest match: extra characters usually mean a
            # different, longer entity name happened to contain this one.
            matches.sort(key=len)
            LOGGER.warning(
                "Ambiguous %s column for %r: %s -- using %r",
                family,
                entity,
                matches,
                matches[0],
            )
        return matches[0]

    conf_cols = [c for c in columns if CONF_TOKEN in c]
    x_cols = [c for c in columns if X_TOKEN in c]
    y_cols = [c for c in columns if Y_TOKEN in c]
    meas_cols = [
        c
        for c in columns
        if re.search(rf"\b{MEAS_TOKEN}\b", c) and STATUS_SUFFIX not in c
    ]
    status_cols = [c for c in columns if c.endswith(STATUS_SUFFIX)]

    for point in POINTS:
        for family, pool, target in (
            ("conf", conf_cols, index.conf),
            ("x", x_cols, index.x),
            ("y", y_cols, index.y),
        ):
            found = _find(pool, point, family)
            if found is not None:
                target[point] = found

    for measure in MEASUREMENTS:
        found = _find(meas_cols, measure, "measurement")
        if found is not None:
            index.measure[measure] = found
        status = f"{measure}{STATUS_SUFFIX}"
        if status in status_cols:
            index.status[measure] = status

    missing_points = [p for p in POINTS if p not in index.conf]
    if missing_points:
        LOGGER.warning("No confidence column found for: %s", missing_points)
    LOGGER.info("Resolved columns: %s", index.summary())
    return index


def index_image_database(database_dir: Path) -> Dict[str, str]:
    """Map every image stem in ``database_dir`` to its taxonomic group.

    Built once by walking the tree, instead of re-scanning the directory for
    every row of the annotation table.
    """
    stem_to_group: Dict[str, str] = {}
    if not database_dir.exists():
        raise FileNotFoundError(f"Image database not found: {database_dir}")

    for folder in sorted(database_dir.iterdir()):
        if not folder.is_dir():
            continue
        group = folder.name.strip().lower()
        if group not in INSECT_GROUPS:
            LOGGER.warning("Ignoring folder %r: not a known insect group", folder.name)
            continue
        for path in folder.iterdir():
            if path.suffix not in IMAGE_EXTENSIONS:
                continue
            previous = stem_to_group.get(path.stem)
            if previous is not None and previous != group:
                LOGGER.warning(
                    "Image stem %r appears in both %r and %r", path.stem, previous, group
                )
            stem_to_group[path.stem] = group

    LOGGER.info("Indexed %d images across %d groups", len(stem_to_group), len(INSECT_GROUPS))
    return stem_to_group


def load_annotations(data_dir: Path, stem_to_group: Dict[str, str]) -> pd.DataFrame:
    """Read annotation CSVs, binarise the statuses and attach the group.

    Rows whose image cannot be located in the database are dropped, with a
    warning: silently keeping them would misalign the group column.
    """
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No annotation CSV found in {data_dir}")
    frames = [pd.read_csv(path) for path in files]
    annotations = pd.concat(frames, ignore_index=True)
    LOGGER.info("Loaded %d annotation rows from %d files", len(annotations), len(files))

    annotations = annotations.rename(columns={"image": "image_name"})
    stems = annotations["image_name"].map(lambda p: Path(str(p)).stem)
    annotations["image_name"] = stems
    annotations["group"] = stems.map(stem_to_group)

    unresolved = annotations["group"].isna()
    if unresolved.any():
        LOGGER.warning(
            "Dropping %d annotation rows whose image is absent from the database",
            int(unresolved.sum()),
        )
        annotations = annotations.loc[~unresolved].copy()

    duplicated = annotations["image_name"].duplicated().sum()
    if duplicated:
        LOGGER.warning(
            "%d duplicated image names in the annotations; keeping the first occurrence",
            int(duplicated),
        )
        annotations = annotations.drop_duplicates(subset="image_name", keep="first")

    for column in annotations.columns:
        if column.endswith(STATUS_SUFFIX):
            annotations[column] = (
                annotations[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(MEASURABLE_LABEL)
                .astype(int)
            )

    return annotations.reset_index(drop=True)


def load_pose_results(results_path: Path) -> pd.DataFrame:
    """Read the pose-estimation outputs.

    Neither normalisation nor NaN filling is applied: XGBoost handles missing
    values natively and treats them as an informative branch, which is exactly
    what a failed keypoint detection is. The previous min-max rescaling was both
    a leak (fitted on the whole dataset) and a loss of information, since it
    destroyed the geometric interpretability of the x/y columns.
    """
    results = pd.read_csv(results_path)
    results = results.rename(columns={c: c.replace("[", "").replace("]", "") for c in results.columns})
    if "image_name" not in results.columns:
        raise KeyError(f"'image_name' column missing from {results_path}")
    results["image_name"] = results["image_name"].map(lambda p: Path(str(p)).stem)
    LOGGER.info("Loaded %d pose-result rows", len(results))
    return results


def build_dataset(
    data_dir: Path,
    database_dir: Path,
    results_path: Path,
) -> tuple:
    """Assemble the full modelling table and its resolved column index."""
    stem_to_group = index_image_database(database_dir)
    annotations = load_annotations(data_dir, stem_to_group)
    results = load_pose_results(results_path)

    merged = annotations.merge(results, how="inner", on="image_name", suffixes=("", "_pose"))
    LOGGER.info(
        "Merged dataset: %d rows (%d annotations, %d pose results)",
        len(merged),
        len(annotations),
        len(results),
    )
    if merged.empty:
        raise ValueError("The merge produced an empty dataset; check image_name formats")

    merged["group"] = pd.Categorical(merged["group"], categories=INSECT_GROUPS)
    for group in INSECT_GROUPS:
        merged[f"{group}_one_hot"] = (merged["group"] == group).astype(int)

    column_index = resolve_columns(merged.columns)
    return merged, column_index


def label_report(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Per-group prevalence table for one measurement.

    ``baseline_accuracy`` is the accuracy of the constant classifier predicting
    the majority class. Any model accuracy must be read against it.
    """
    status = f"{measure}{STATUS_SUFFIX}"
    rows = []
    for group, subset in frame.groupby("group", observed=True):
        positives = int(subset[status].sum())
        total = len(subset)
        if total == 0:
            continue
        prevalence = positives / total
        rows.append(
            {
                "group": group,
                "n": total,
                "n_measurable": positives,
                "n_unmeasurable": total - positives,
                "prevalence_measurable": prevalence,
                "baseline_accuracy": max(prevalence, 1 - prevalence),
            }
        )
    positives = int(frame[status].sum())
    total = len(frame)
    prevalence = positives / total
    rows.append(
        {
            "group": "ALL",
            "n": total,
            "n_measurable": positives,
            "n_unmeasurable": total - positives,
            "prevalence_measurable": prevalence,
            "baseline_accuracy": max(prevalence, 1 - prevalence),
        }
    )
    return pd.DataFrame(rows)