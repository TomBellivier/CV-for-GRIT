#!/usr/bin/env python3
"""
check_annotations.py
====================

Check every annotation source of the repository, on its own and against the others,
before anything is trained on it. Above all, it reports the images that are present in
some annotation files but missing from others.

Sources (defaults: the standard locations, see the README):

    label_studio   annotation_data/label_studio_annotations/<group>/*.json
    pose           annotation_data/pose/pose_annotations.csv
    measurements   annotation_data/meas_classifier/*.csv
    scale          annotation_data/scale/scale_annotations.csv
    table          annotation_data/annotation_data.csv   (the merged table)
    image files    annotated_images/full databases/<group>/...          (--images-dir)
                   modules/architectures/data/raw/<group>/images/ (--training-images)

What is checked
---------------
- every file: readable, expected columns, valid values (insect group, image size,
  keypoint visibility, keypoints inside the image, measurement statuses, scale fields),
  images annotated twice;
- across files: an image present in one source and missing from another (Label Studio
  exports vs pose CSV, pose vs measurement labels, pose vs scale), and group conflicts;
- the merged table: rebuilt in memory from its three sources and compared cell by cell
  with annotation_data.csv, so an out-of-date table is caught;
- the image files: every annotated image found in the image database (under the folder
  of its group) and in the training folder of the pose module;
- the measurement-validity classifiers: which measurements have enough labels of each
  class to get a classifier.

Severity: ERROR = the data is wrong or a step will fail; WARNING = data is silently
lost or inconsistent; INFO = expected in a partially annotated project, reported for
completeness.

Outputs
-------
    console                    one line per problem, with its count and a few examples
    <report-dir>/issues.csv    every problem, one row per image
    <report-dir>/presence.csv  one row per image, one column per source (1 = present)

Exit code: 1 if an ERROR was found (or any WARNING with --strict), 0 otherwise.

Usage
-----
    python annotation_tools/check_annotations.py
    python annotation_tools/check_annotations.py --no-image-files --strict
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for _path in (REPO_ROOT, TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_annotation_data as build  # noqa: E402
from kp_infos import INSECT_GROUPS, KEYPOINT_NAMES, MEAS_TO_KP, MEASUREMENTS  # noqa: E402
from labelstudio_to_csv import image_name_of, parse_annotation  # noqa: E402

DEFAULT_LABEL_STUDIO = REPO_ROOT / "annotation_data" / "label_studio_annotations"
DEFAULT_IMAGES_DIR = REPO_ROOT / "annotated_images" / "full databases"
DEFAULT_TRAINING_IMAGES = REPO_ROOT / "modules" / "architectures" / "data" / "raw"
DEFAULT_REPORT_DIR = REPO_ROOT / "results" / "annotation_check"

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"
SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}

KEY = build.KEY                                  # "image_name"
STATUS_SUFFIX = "_status"
STATUS_VALUES = {"measurable", "non_measurable"}
SCALE_TYPES = {"ruler", "scale_bar"}
RULER_DIRECTIONS = {"horizontal", "vertical"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
PIXEL_TOLERANCE = 1.0                            # a keypoint may sit on the image border


def rel(path: Path) -> str:
    """Path shown to the user: relative to the repository when possible."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    severity: str
    check: str          # stable identifier, e.g. "pose.duplicate"
    source: str         # file or source concerned
    message: str        # what is wrong and what to do, shared by every item of a check
    item: str = ""      # the image (or measurement) concerned, "" for a whole file
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, severity: str, check: str, source: str, message: str,
            items=("",), details: dict | None = None) -> None:
        """One issue per item; nothing is recorded when `items` is empty."""
        details = details or {}
        for item in sorted({str(i) for i in items}):
            self.issues.append(Issue(severity, check, source, message, item,
                                     str(details.get(item, ""))))

    def count(self, severity: str) -> int:
        return sum(issue.severity == severity for issue in self.issues)

    def print(self, max_examples: int) -> None:
        groups: dict[tuple, list[Issue]] = defaultdict(list)
        for issue in self.issues:
            groups[(issue.severity, issue.check, issue.source, issue.message)].append(issue)
        if not groups:
            print("\nNo problem found.")
            return
        print("\nProblems")
        for (severity, _check, source, message), issues in sorted(
                groups.items(), key=lambda kv: (SEVERITY_ORDER[kv[0][0]], kv[0][1], kv[0][2])):
            named = [i for i in issues if i.item]
            count = f"{len(named)} item(s): " if named else ""
            print(f"  [{severity}] {source}: {count}{message}")
            if named:
                shown = [f"{i.item} ({i.detail})" if i.detail else i.item
                         for i in named[:max_examples]]
                more = f"  (+{len(named) - max_examples} more)" if len(named) > max_examples else ""
                print(f"      e.g. {'; '.join(shown)}{more}")
            else:
                for issue in issues[:max_examples]:
                    if issue.detail:
                        print(f"      {issue.detail}")

    def write(self, path: Path) -> None:
        pd.DataFrame([vars(i) for i in self.issues],
                     columns=["severity", "check", "source", "message", "item", "detail"]
                     ).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def names_of(frame: pd.DataFrame, column: str = KEY) -> pd.Series:
    """Image file names of a column, whatever the path separator (like the merge)."""
    return frame[column].astype(str).map(build._basename)


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


# --------------------------------------------------------------------------- #
# Label Studio exports
# --------------------------------------------------------------------------- #
def check_label_studio(root: Path, report: Report) -> dict[str, str] | None:
    """{image name: group} of every image with at least one annotated keypoint."""
    source = "label_studio"
    if not root.is_dir():
        report.add(WARNING, "label_studio.missing", source,
                   f"No Label Studio export folder at {rel(root)}: exports not checked.")
        return None

    files = sorted(root.rglob("*.json"))
    if not files:
        report.add(WARNING, "label_studio.empty", source, f"No JSON export under {rel(root)}.")
        return None

    groups: dict[str, set[str]] = defaultdict(set)
    exports: dict[str, list[str]] = defaultdict(list)
    for path in files:
        group = path.parent.name.lower()
        if group not in INSECT_GROUPS:
            report.add(ERROR, "label_studio.group", source,
                       "Export outside a folder named after an insect group "
                       f"({', '.join(INSECT_GROUPS)}): its images would get a wrong group.",
                       details={"": rel(path)})
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.add(ERROR, "label_studio.unreadable", source, "Unreadable export.",
                       details={"": f"{rel(path)}: {exc}"})
            continue

        unknown: set[str] = set()
        for task in tasks:
            name = image_name_of(task.get("data", {}).get("img", ""))
            for annotation in task.get("annotations", []):
                if annotation.get("was_cancelled"):
                    continue
                parsed = parse_annotation(annotation.get("result", []))
                unknown |= parsed["unknown_labels"]
                if parsed["keypoints"]:
                    groups[name].add(group)
                    exports[name].append(path.name)
        if unknown:
            report.add(WARNING, "label_studio.labels", source,
                       "Keypoint labels unknown to kp_infos.yaml: the conversion drops them.",
                       details={"": f"{rel(path)}: {sorted(unknown)}"})

    twice = {n for n, e in exports.items() if len(e) > 1}
    report.add(WARNING, "label_studio.duplicate", source,
               "Image annotated more than once: the merge only keeps its first annotation.",
               twice, {n: ", ".join(exports[n]) for n in twice})
    conflict = {n for n, g in groups.items() if len(g) > 1}
    report.add(ERROR, "label_studio.group_conflict", source,
               "Image annotated under several insect groups.",
               conflict, {n: ", ".join(sorted(groups[n])) for n in conflict})
    return {n: sorted(g)[0] for n, g in groups.items()}


# --------------------------------------------------------------------------- #
# Pose CSV
# --------------------------------------------------------------------------- #
def check_pose(path: Path, report: Report) -> pd.DataFrame | None:
    """The pose CSV, with its file names normalised. None if it cannot be used."""
    source = rel(path)
    if not path.is_file():
        report.add(ERROR, "pose.missing", source,
                   "Pose CSV missing: produce it with annotation_tools/labelstudio_to_csv.py.")
        return None
    frame = pd.read_csv(path, low_memory=False)

    required = [KEY, "group", "width", "height"]
    missing = [c for c in required if c not in frame.columns]
    missing_kp = [f"{p}_{a}" for p in KEYPOINT_NAMES for a in "xyv" if f"{p}_{a}" not in frame.columns]
    if missing or missing_kp:
        report.add(ERROR, "pose.columns", source,
                   "Columns missing: the file was not produced by labelstudio_to_csv.py with "
                   "the current kp_infos.yaml.",
                   details={"": f"{missing + missing_kp[:6]}"
                                 + (f" (+{len(missing_kp) - 6} more)" if len(missing_kp) > 6 else "")})
        if KEY not in frame.columns:
            return None

    frame = frame.copy()
    frame[KEY] = names_of(frame)
    extra = sorted({c[:-2] for c in frame.columns if c.endswith("_x")
                    and f"{c[:-2]}_y" in frame.columns and f"{c[:-2]}_v" in frame.columns
                    and c[:-2] not in KEYPOINT_NAMES})
    if extra:
        report.add(WARNING, "pose.unknown_keypoints", source,
                   "Keypoint columns unknown to kp_infos.yaml: ignored by every module.",
                   details={"": ", ".join(extra)})

    duplicated = frame[KEY][frame[KEY].duplicated()]
    report.add(WARNING, "pose.duplicate", source,
               "Image annotated more than once: the merge only keeps the first row.",
               duplicated)

    if "group" in frame.columns:
        group = frame["group"].astype(str).str.strip().str.lower()
        bad = frame.loc[~group.isin(INSECT_GROUPS)]
        report.add(ERROR, "pose.group", source,
                   f"Invalid insect group (expected one of {', '.join(INSECT_GROUPS)}).",
                   bad[KEY], dict(zip(bad[KEY], bad["group"].astype(str))))

    if {"width", "height"} <= set(frame.columns):
        width, height = as_number(frame["width"]), as_number(frame["height"])
        bad = frame.loc[~((width > 0) & (height > 0))]
        report.add(ERROR, "pose.size", source,
                   "Image size missing or invalid: pose training needs width and height.",
                   bad[KEY])
    else:
        width = height = pd.Series(np.nan, index=frame.index)

    present = [p for p in KEYPOINT_NAMES if f"{p}_v" in frame.columns and f"{p}_x" in frame.columns]
    annotated = pd.Series(0, index=frame.index)
    bad_visibility: dict[str, list[str]] = defaultdict(list)
    outside: dict[str, list[str]] = defaultdict(list)
    for point in present:
        vis = as_number(frame[f"{point}_v"]).fillna(0)
        x, y = as_number(frame[f"{point}_x"]), as_number(frame[f"{point}_y"])
        for name in frame.loc[~vis.isin([0, 1, 2]), KEY]:
            bad_visibility[name].append(point)
        on = vis > 0
        annotated += on.astype(int)
        off_image = on & ((x < -PIXEL_TOLERANCE) | (y < -PIXEL_TOLERANCE)
                          | (x > width + PIXEL_TOLERANCE) | (y > height + PIXEL_TOLERANCE))
        for name in frame.loc[off_image, KEY]:
            outside[name].append(point)

    report.add(WARNING, "pose.visibility", source,
               "Keypoint visibility outside {0, 1, 2}.",
               bad_visibility, {n: ", ".join(p) for n, p in bad_visibility.items()})
    report.add(WARNING, "pose.outside", source,
               "Annotated keypoint outside the image.",
               outside, {n: ", ".join(p[:4]) for n, p in outside.items()})
    report.add(WARNING, "pose.empty", source,
               "No annotated keypoint: the row is ignored by pose training.",
               frame.loc[annotated == 0, KEY])
    frame["_n_keypoints"] = annotated
    return frame


# --------------------------------------------------------------------------- #
# Measurement-validity labels
# --------------------------------------------------------------------------- #
def measurement_files(sources) -> list[Path]:
    files: list[Path] = []
    for raw in sources:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            files.append(path)
    return files


def check_measurements(sources, report: Report) -> dict[str, dict[str, str]] | None:
    """{image name: {measurement: status}}, first file winning, like the merge."""
    source = "measurements"
    files = measurement_files(sources)
    if not files:
        report.add(WARNING, "measurements.missing", source,
                   "No measurement-validity CSV: the validity classifiers cannot be trained.")
        return None

    expected = [f"{m}{STATUS_SUFFIX}" for m in MEASUREMENTS]
    labels: dict[str, dict[str, str]] = {}
    origin: dict[str, str] = {}
    conflicts: dict[str, str] = {}
    for path in files:
        name_src = rel(path)
        frame = pd.read_csv(path, low_memory=False)
        column = "image" if "image" in frame.columns else KEY
        if column not in frame.columns:
            report.add(ERROR, "measurements.columns", name_src,
                       f"Neither 'image' nor '{KEY}' column: file ignored by the merge.")
            continue
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            report.add(WARNING, "measurements.columns", name_src,
                       "Status columns missing (measurements added to kp_infos.yaml after "
                       "this batch?): these statuses stay empty.",
                       details={"": ", ".join(c[:-len(STATUS_SUFFIX)] for c in missing)})
        unknown = [c for c in frame.columns if c.endswith(STATUS_SUFFIX) and c not in expected]
        if unknown:
            report.add(WARNING, "measurements.unknown", name_src,
                       "Status columns of measurements unknown to kp_infos.yaml: ignored.",
                       details={"": ", ".join(unknown)})

        names = names_of(frame, column)
        report.add(WARNING, "measurements.duplicate", name_src,
                   "Image classified more than once in the same file: the merge keeps the first row.",
                   names[names.duplicated()])

        status_cols = [c for c in expected if c in frame.columns]
        invalid: dict[str, list[str]] = defaultdict(list)
        incomplete: list[str] = []
        for name, (_, row) in zip(names, frame.iterrows()):
            statuses = {}
            for col in status_cols:
                value = row[col]
                if blank(value):
                    continue
                value = str(value).strip().lower()
                if value not in STATUS_VALUES:
                    invalid[name].append(f"{col[:-len(STATUS_SUFFIX)]}={value}")
                statuses[col[:-len(STATUS_SUFFIX)]] = value
            if len(statuses) < len(expected):
                incomplete.append(name)
            if name in labels:
                if labels[name] != statuses and name not in conflicts:
                    conflicts[name] = f"{origin[name]} vs {path.name}"
                continue
            labels[name], origin[name] = statuses, path.name
        report.add(ERROR, "measurements.values", name_src,
                   f"Status other than {sorted(STATUS_VALUES)}.",
                   invalid, {n: ", ".join(v[:3]) for n, v in invalid.items()})
        report.add(WARNING, "measurements.incomplete", name_src,
                   "Some measurements have no status: train_measure_validity.py leaves these "
                   "images out.", incomplete)

    report.add(WARNING, "measurements.conflict", source,
               "Image classified in several files with different statuses: the merge keeps "
               "the first file.", conflicts, conflicts)
    return labels


# --------------------------------------------------------------------------- #
# Scale annotations
# --------------------------------------------------------------------------- #
def check_scale(path: Path, report: Report, sizes: dict[str, tuple[float, float]]) -> set[str] | None:
    source = rel(path)
    if not path.is_file():
        report.add(INFO, "scale.missing", source,
                   "No scale CSV: the scale columns of the merged table stay empty.")
        return None
    frame = pd.read_csv(path, low_memory=False)
    if KEY not in frame.columns:
        report.add(ERROR, "scale.columns", source, f"'{KEY}' column missing: file unusable.")
        return None
    unknown = [c for c in frame.columns if c not in (KEY, *build.SCALE_COLUMNS)]
    if unknown:
        report.add(WARNING, "scale.unknown", source,
                   "Unknown columns: copied as they are into the merged table.",
                   details={"": ", ".join(unknown)})

    frame = frame.copy()
    frame[KEY] = names_of(frame)
    report.add(WARNING, "scale.duplicate", source,
               "Image annotated more than once: the merge keeps the first row.",
               frame[KEY][frame[KEY].duplicated()])

    def column(name: str) -> pd.Series:
        return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)

    scale_type = column("scale_type").map(lambda v: "" if blank(v) else str(v).strip().lower())
    bad = frame.loc[(scale_type != "") & ~scale_type.isin(SCALE_TYPES)]
    report.add(ERROR, "scale.type", source,
               f"scale_type other than {sorted(SCALE_TYPES)}.",
               bad[KEY], dict(zip(bad[KEY], scale_type[bad.index])))

    px = as_number(column("scale_px_per_mm"))
    raw_px = column("scale_px_per_mm")
    bad = frame.loc[raw_px.map(lambda v: not blank(v)) & ~(px > 0)]
    report.add(ERROR, "scale.px_per_mm", source,
               "scale_px_per_mm is not a positive number.", bad[KEY])

    direction = column("ruler_direction").map(lambda v: "" if blank(v) else str(v).strip().lower())
    bad = frame.loc[(direction != "") & ~direction.isin(RULER_DIRECTIONS)]
    report.add(ERROR, "scale.ruler_direction", source,
               f"ruler_direction other than {sorted(RULER_DIRECTIONS)}.", bad[KEY])

    lo, hi = as_number(column("ruler_line_min")), as_number(column("ruler_line_max"))
    has_range = lo.notna() | hi.notna()
    bad = frame.loc[has_range & ~((lo >= 0) & (hi >= lo))]
    report.add(ERROR, "scale.ruler_range", source,
               "Ruler range invalid: needs 0 <= ruler_line_min <= ruler_line_max.", bad[KEY])

    beyond = []
    for idx in frame.index[has_range & (hi >= lo)]:
        name = frame.at[idx, KEY]
        if name not in sizes:
            continue
        width, height = sizes[name]
        limit = width if direction[idx] == "vertical" else height
        if np.isfinite(limit) and hi[idx] > limit + PIXEL_TOLERANCE:
            beyond.append(name)
    report.add(WARNING, "scale.ruler_beyond", source,
               "Ruler range beyond the image size given by the pose annotation.", beyond)

    ruler_fields = has_range | (direction != "")
    bar_fields = column("scale_bar_text").map(lambda v: not blank(v)) \
        | as_number(column("scale_bar_bbox_x")).notna()
    report.add(WARNING, "scale.type_mismatch", source,
               "Ruler fields filled although scale_type is 'scale_bar'.",
               frame.loc[ruler_fields & (scale_type == "scale_bar"), KEY])
    report.add(WARNING, "scale.type_mismatch", source,
               "Scale-bar fields filled although scale_type is 'ruler'.",
               frame.loc[bar_fields & (scale_type == "ruler"), KEY])

    info_cols = [c for c in build.SCALE_COLUMNS if c in frame.columns]
    empty = frame.loc[frame[info_cols].apply(lambda r: all(blank(v) for v in r), axis=1), KEY] \
        if info_cols else frame[KEY]
    report.add(INFO, "scale.empty", source, "Row without any scale information.", empty)
    return set(frame[KEY])


# --------------------------------------------------------------------------- #
# Cross-checks between sources
# --------------------------------------------------------------------------- #
def check_presence(report: Report, label_studio, pose, labels, scale) -> None:
    pose_names = set(pose[KEY]) if pose is not None else None

    if label_studio is not None and pose_names is not None:
        report.add(WARNING, "presence.label_studio_not_in_pose", "pose",
                   "Annotated in a Label Studio export but missing from the pose CSV: re-run "
                   "labelstudio_to_csv.py on annotation_data/label_studio_annotations.",
                   set(label_studio) - pose_names)
        report.add(WARNING, "presence.pose_not_in_label_studio", "pose",
                   "In the pose CSV but in no Label Studio export (deleted or renamed export?).",
                   pose_names - set(label_studio))
        groups = dict(zip(pose[KEY], pose["group"].astype(str).str.lower())) \
            if "group" in pose.columns else {}
        mismatch = {n for n in set(label_studio) & pose_names
                    if groups.get(n) and groups[n] != label_studio[n]}
        report.add(ERROR, "presence.group_mismatch", "pose",
                   "Group of the pose CSV differs from the folder of its Label Studio export.",
                   mismatch, {n: f"{groups[n]} vs {label_studio[n]}" for n in mismatch})

    if labels is not None and pose_names is not None:
        report.add(WARNING, "presence.measurements_not_in_pose", "measurements",
                   "Validity labels for an image without pose annotation: the classifiers "
                   "cannot use them (they need the keypoints).",
                   set(labels) - pose_names)
        annotated = set(pose.loc[pose["_n_keypoints"] > 0, KEY])
        report.add(INFO, "presence.pose_not_in_measurements", "measurements",
                   "Pose annotation without validity labels (batch not classified yet).",
                   annotated - set(labels))

        # A measurement can only be measurable if its keypoints were annotated: the
        # classification app enforces it. A mismatch means the pose was edited afterwards.
        first = pose.drop_duplicates(KEY).set_index(KEY)
        inconsistent: dict[str, list[str]] = defaultdict(list)
        for name, statuses in labels.items():
            if name not in first.index:
                continue
            row = first.loc[name]
            for measure, status in statuses.items():
                if status != "measurable" or measure not in MEAS_TO_KP:
                    continue
                vis = [row.get(f"{p}_v", 0) for p in MEAS_TO_KP[measure]]
                if any(blank(v) or float(v) <= 0 for v in vis):
                    inconsistent[name].append(measure)
        report.add(WARNING, "measurements.keypoints", "measurements",
                   "Measurement marked measurable although one of its keypoints is not "
                   "annotated (pose annotation edited after the classification?).",
                   inconsistent, {n: ", ".join(m[:3]) for n, m in inconsistent.items()})

    if scale is not None and pose_names is not None:
        report.add(INFO, "presence.scale_not_in_pose", "scale",
                   "Scale annotation for an image without pose annotation (still used by the "
                   "ruler evaluation).", scale - pose_names)
        report.add(INFO, "presence.pose_not_in_scale", "scale",
                   "Pose annotation without scale annotation.",
                   set(pose.loc[pose["_n_keypoints"] > 0, KEY]) - scale)


# --------------------------------------------------------------------------- #
# Merged table
# --------------------------------------------------------------------------- #
def rebuild_table(pose_path: Path, scale_path: Path, measurement_sources) -> pd.DataFrame | None:
    """The merged table build_annotation_data.py would write now (read back as a CSV)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            merged = build.merge(
                build.deduplicate(build.load_pose(pose_path), "pose"),
                build.deduplicate(build.load_scale(scale_path), "scale"),
                build.deduplicate(build.load_measurements(measurement_sources), "measurements"),
            )
    except (SystemExit, Exception):  # noqa: BLE001 - the sources were already reported
        return None
    buffer = io.StringIO()
    merged.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, low_memory=False)


def same_cells(expected: pd.Series, actual: pd.Series) -> pd.Series:
    both_missing = expected.isna() & actual.isna()
    num_e, num_a = as_number(expected), as_number(actual)
    numeric = num_e.notna() & num_a.notna()
    close = pd.Series(False, index=expected.index)
    close[numeric] = np.isclose(num_e[numeric], num_a[numeric], rtol=1e-6, atol=1e-6)
    text = expected.astype(str).str.strip() == actual.astype(str).str.strip()
    return both_missing | close | (~numeric & text & expected.notna() & actual.notna())


def check_table(path: Path, report: Report, expected: pd.DataFrame | None) -> set[str] | None:
    source = rel(path)
    if not path.is_file():
        report.add(ERROR, "table.missing", source,
                   "Merged table missing: run annotation_tools/build_annotation_data.py.")
        return None
    actual = pd.read_csv(path, low_memory=False)
    if KEY not in actual.columns:
        report.add(ERROR, "table.columns", source, f"'{KEY}' column missing.")
        return None
    names = set(names_of(actual))
    if expected is None:
        return names

    hint = "annotation_data.csv is out of date: re-run annotation_tools/build_annotation_data.py."
    expected_names = set(expected[KEY].astype(str))
    report.add(WARNING, "table.missing_images", source,
               f"In the annotation sources but missing from the merged table. {hint}",
               expected_names - names)
    report.add(WARNING, "table.extra_images", source,
               f"In the merged table but in no annotation source. {hint}",
               names - expected_names)
    missing_cols = sorted(set(expected.columns) - set(actual.columns))
    extra_cols = sorted(set(actual.columns) - set(expected.columns))
    if missing_cols or extra_cols:
        report.add(WARNING, "table.columns", source, f"Columns differ from its sources. {hint}",
                   details={"": f"missing {missing_cols[:5]}, extra {extra_cols[:5]}"})

    exp = expected.drop_duplicates(KEY).set_index(KEY)
    act = actual.assign(**{KEY: names_of(actual)}).drop_duplicates(KEY).set_index(KEY)
    common_rows = exp.index.intersection(act.index)
    common_cols = [c for c in exp.columns if c in act.columns]
    differing: dict[str, list[str]] = defaultdict(list)
    for col in common_cols:
        same = same_cells(exp.loc[common_rows, col], act.loc[common_rows, col])
        for name in same.index[~same.to_numpy()]:
            differing[name].append(col)
    report.add(WARNING, "table.values", source,
               f"Values differ from its sources. {hint}",
               differing, {n: ", ".join(c[:3]) for n, c in differing.items()})
    return names


# --------------------------------------------------------------------------- #
# Image files
# --------------------------------------------------------------------------- #
def check_image_files(report: Report, images_dir: Path, training_dir: Path,
                      pose: pd.DataFrame | None) -> tuple[set | None, set | None]:
    if pose is None:
        return None, None
    annotated = pose.loc[pose["_n_keypoints"] > 0].drop_duplicates(KEY)
    group_of = dict(zip(annotated[KEY], annotated["group"].astype(str).str.lower())) \
        if "group" in annotated.columns else {}

    in_database: set[str] | None = None
    if not images_dir.is_dir():
        report.add(WARNING, "images.database_missing", rel(images_dir),
                   "Image database not found: the pipeline cannot look up the insect group "
                   "of an image and the ruler evaluation has no image.")
    else:
        found: dict[str, set[str]] = defaultdict(set)
        for group in INSECT_GROUPS:
            folder = images_dir / group
            if not folder.is_dir():
                report.add(WARNING, "images.group_folder", rel(images_dir),
                           f"No '{group}' folder: images of that group cannot be found.")
                continue
            for file in folder.rglob("*"):
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    found[file.name].add(group)
        in_database = set(found)
        several = {n for n, g in found.items() if len(g) > 1}
        report.add(WARNING, "images.duplicate_name", rel(images_dir),
                   "Same file name in several group folders: the group lookup is ambiguous.",
                   several, {n: ", ".join(sorted(found[n])) for n in several})
        report.add(WARNING, "images.not_in_database", rel(images_dir),
                   "Annotated image not found in the image database.",
                   set(group_of) - in_database)
        wrong = {n for n in set(group_of) & in_database
                 if group_of[n] not in found[n]}
        report.add(ERROR, "images.group_mismatch", rel(images_dir),
                   "Annotated image found under another group folder than its annotated group.",
                   wrong, {n: f"annotated {group_of[n]}, found in {', '.join(sorted(found[n]))}"
                           for n in wrong})

    in_training: set[str] = set()
    for group in INSECT_GROUPS:
        expected = {n for n, g in group_of.items() if g == group}
        if not expected:
            continue
        folder = training_dir / group / "images"
        if not folder.is_dir():
            report.add(WARNING, "images.training_folder", rel(training_dir),
                       f"No {rel(folder)} folder: pose training cannot read the "
                       f"{len(expected)} '{group}' images.")
            continue
        present = {f.name for f in folder.iterdir() if f.is_file()}
        in_training |= expected & present
        report.add(WARNING, "images.not_in_training", rel(folder),
                   "Annotated image missing from the training folder of the pose module.",
                   expected - present)
    return in_database, in_training


# --------------------------------------------------------------------------- #
# Measurement-validity classifiers
# --------------------------------------------------------------------------- #
def check_class_balance(report: Report, table: pd.DataFrame | None, min_per_class: int) -> None:
    """Measurements that will get no classifier, computed like train_measure_validity.py."""
    if table is None:
        return
    status_cols = [f"{m}{STATUS_SUFFIX}" for m in MEASUREMENTS if f"{m}{STATUS_SUFFIX}" in table.columns]
    if not status_cols:
        return
    complete = table.dropna(subset=status_cols)
    complete = complete[complete[status_cols].notna().all(axis=1)]
    details, short = {}, []
    for col in status_cols:
        values = complete[col].astype(str).str.strip().str.lower()
        measurable = int((values == "measurable").sum())
        unmeasurable = len(values) - measurable
        if min(measurable, unmeasurable) < min_per_class:
            measure = col[:-len(STATUS_SUFFIX)]
            short.append(measure)
            details[measure] = f"{measurable} measurable / {unmeasurable} non measurable"
    report.add(INFO, "classifiers.too_few", "measurement classifiers",
               f"Fewer than {min_per_class} images in one class: no classifier will be trained "
               f"for this measurement ({len(complete)} fully classified images).",
               short, details)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
# (present in, missing from): the gaps printed at the top of the report.
PRESENCE_PAIRS = [
    ("label_studio", "pose_csv"),
    ("pose_csv", "label_studio"),
    ("pose_csv", "measurements"),
    ("measurements", "pose_csv"),
    ("pose_csv", "scale_csv"),
    ("scale_csv", "pose_csv"),
    ("annotation_sources", "annotation_table"),
    ("annotation_table", "annotation_sources"),
    ("pose_csv", "image_database"),
    ("pose_csv", "training_images"),
]


def print_presence(sets: dict[str, set | None]) -> None:
    print("\nImages present in one source but missing from another")
    for present, absent in PRESENCE_PAIRS:
        if sets.get(present) is None or sets.get(absent) is None:
            value = "not checked"
        else:
            value = f"{len(sets[present] - sets[absent]):6d}"
        print(f"  in {present:18} missing from {absent:18} {value}")


def presence_table(sources: dict[str, set | None], groups: dict[str, str]) -> pd.DataFrame:
    names = sorted(set().union(*[s for s in sources.values() if s is not None]))
    frame = pd.DataFrame({KEY: names, "group": [groups.get(n, "") for n in names]})
    for label, members in sources.items():
        frame[label] = "" if members is None else [int(n in members) for n in names]
    return frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check the annotation files and report the images missing from some of them.")
    p.add_argument("--label-studio", default=str(DEFAULT_LABEL_STUDIO),
                   help="Folder of the Label Studio exports, one sub-folder per group.")
    p.add_argument("--pose", default=str(build.DEFAULT_POSE), help="Pose CSV.")
    p.add_argument("--scale", default=str(build.DEFAULT_SCALE), help="Scale CSV.")
    p.add_argument("--measurements", nargs="*", default=[str(build.DEFAULT_MEASUREMENTS)],
                   help="Measurement-validity CSVs and/or folders holding them.")
    p.add_argument("--annotation-table", default=str(build.DEFAULT_OUTPUT),
                   help="Merged table (annotation_data.csv).")
    p.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR),
                   help="Image database, one sub-folder per group.")
    p.add_argument("--training-images", default=str(DEFAULT_TRAINING_IMAGES),
                   help="Image root of the pose module (<group>/images/ below it).")
    p.add_argument("--no-image-files", action="store_true",
                   help="Do not look for the image files.")
    p.add_argument("--min-per-class", type=int, default=20,
                   help="Images needed in each class for a measurement to get a classifier "
                        "(same default as train_measure_validity.py).")
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                   help="Where issues.csv and presence.csv are written.")
    p.add_argument("--no-report", action="store_true", help="Only print to the console.")
    p.add_argument("--max-examples", type=int, default=5,
                   help="Image names shown per problem in the console.")
    p.add_argument("--strict", action="store_true", help="Exit with 1 on warnings too.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()

    label_studio = check_label_studio(Path(args.label_studio), report)
    pose = check_pose(Path(args.pose), report)
    labels = check_measurements(args.measurements, report)
    sizes = {}
    if pose is not None and {"width", "height"} <= set(pose.columns):
        first = pose.drop_duplicates(KEY)
        sizes = dict(zip(first[KEY], zip(as_number(first["width"]), as_number(first["height"]))))
    scale = check_scale(Path(args.scale), report, sizes)
    check_presence(report, label_studio, pose, labels, scale)

    expected = rebuild_table(Path(args.pose), Path(args.scale), args.measurements) \
        if pose is not None else None
    table_names = check_table(Path(args.annotation_table), report, expected)
    check_class_balance(report, expected, args.min_per_class)

    in_database = in_training = None
    if not args.no_image_files:
        in_database, in_training = check_image_files(
            report, Path(args.images_dir), Path(args.training_images), pose)

    sources = {
        "label_studio": set(label_studio) if label_studio is not None else None,
        "pose_csv": set(pose[KEY]) if pose is not None else None,
        "measurements": set(labels) if labels is not None else None,
        "scale_csv": scale,
        "annotation_table": table_names,
        "image_database": in_database,
        "training_images": in_training,
    }
    groups = dict(label_studio or {})
    if pose is not None and "group" in pose.columns:
        groups.update(dict(zip(pose[KEY], pose["group"].astype(str).str.lower())))

    print("=" * 78)
    print("Annotation check")
    print("=" * 78)
    print("Images per source")
    for label, members in sources.items():
        print(f"  {label:18} {'not checked' if members is None else f'{len(members):6d}'}")
    print_presence({**sources,
                    "annotation_sources": set(expected[KEY].astype(str))
                    if expected is not None else None})
    report.print(args.max_examples)

    if not args.no_report:
        out = Path(args.report_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.write(out / "issues.csv")
        presence_table(sources, groups).to_csv(out / "presence.csv", index=False)
        print(f"\nFull report: {rel(out / 'issues.csv')}, {rel(out / 'presence.csv')}")

    errors, warnings = report.count(ERROR), report.count(WARNING)
    print(f"\nSummary: {errors} error(s), {warnings} warning(s), {report.count(INFO)} info.")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
