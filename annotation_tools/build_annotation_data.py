#!/usr/bin/env python3
"""
build_annotation_data.py
========================

Merge every annotation source into the single table the training modules read:
``annotation_data/annotation_data.csv``, one row per image.

Three CSV sources, joined on the image file name:

    pose           the Label Studio exports, ALREADY converted to CSV by
                   labelstudio_to_csv.py -- that conversion is a separate step,
                   run it first.
                   -> image size, group, bbox, remarks, <keypoint>_x/_y/_v
    scale          hand-written CSV, see annotation_data/scale/
                   -> scale_type, scale_px_per_mm, scale-bar box and text,
                      ruler direction and line range
    measurements   measurement_validation exports (one CSV per batch)
                   -> <measure>_status, <measure>_px

A cell is left empty when the source has nothing for that image: every image
known to any source gets a row, and no source is required.

Usage
-----
    # 1. conversion of the Label Studio exports to CSV
    python annotation_tools/labelstudio_to_csv.py \\
        --json_files annotation_data/label_studio_annotations \\
        --output annotation_data/pose/pose_annotations.csv

    # 2. merge of the three CSVs
    python annotation_tools/build_annotation_data.py

    python annotation_tools/build_annotation_data.py \\
        --pose pose.csv --scale my_scale.csv --measurements dir_or_file ... \\
        --output annotation_data/annotation_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POSE = REPO_ROOT / "annotation_data" / "pose" / "pose_annotations.csv"
DEFAULT_SCALE = REPO_ROOT / "annotation_data" / "scale" / "scale_annotations.csv"
DEFAULT_MEASUREMENTS = REPO_ROOT / "annotation_data" / "meas_classifier"
DEFAULT_OUTPUT = REPO_ROOT / "annotation_data" / "annotation_data.csv"

KEY = "image_name"

# Columns of the hand-written scale CSV (see its README). Any of them may be
# empty; only the image name is required.
SCALE_COLUMNS = [
    "scale_type",            # "ruler" | "scale_bar"
    "scale_px_per_mm",
    "scale_bar_bbox_x",      # scale bar only
    "scale_bar_bbox_y",      # scale bar only
    "scale_bar_text",        # scale bar only
    "ruler_direction",       # ruler only : "horizontal" | "vertical"
    "ruler_line_min",        # ruler only : line/column range the ruler spans
    "ruler_line_max",        # ruler only
]


def _basename(value) -> str:
    """File name of a path written with any separator, so a Windows path read on
    Linux still splits ('C:\\images\\a.png' -> 'a.png')."""
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def load_pose(source: Path) -> pd.DataFrame:
    """Pose annotations, as a CSV already converted by ``labelstudio_to_csv.py``.

    The conversion is a step of its own, on purpose: this script merges CSVs, it
    does not read Label Studio exports.
    """
    if source.suffix.lower() != ".csv" or not source.is_file():
        raise SystemExit(
            f"The pose annotations must be an already converted CSV: {source}\n"
            f"Produce it first with:\n"
            f"    python annotation_tools/labelstudio_to_csv.py "
            f"--json_files annotation_data/label_studio_annotations "
            f"--output {DEFAULT_POSE.relative_to(REPO_ROOT).as_posix()}"
        )

    frame = pd.read_csv(source)
    if KEY not in frame.columns:
        raise KeyError(f"Column '{KEY}' missing from the pose annotations ({source})")
    return frame


def load_scale(source: Path) -> pd.DataFrame:
    """Hand-written scale annotations. Missing file -> no scale columns filled."""
    if not source.is_file():
        print(f"/!\\ no scale CSV at {source}: scale columns left empty")
        return pd.DataFrame(columns=[KEY, *SCALE_COLUMNS])

    frame = pd.read_csv(source)
    if KEY not in frame.columns:
        raise KeyError(f"Column '{KEY}' missing from the scale CSV ({source})")
    unknown = [c for c in frame.columns if c not in (KEY, *SCALE_COLUMNS)]
    if unknown:
        print(f"/!\\ unknown columns in {source.name}, kept as is: {unknown}")
    return frame


def load_measurements(sources) -> pd.DataFrame:
    """measurement_validation exports: '<measure>_status' / '<measure>_px' per image."""
    files: list[Path] = []
    for raw in sources:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"/!\\ skipped (not found): {path}")
    if not files:
        print("/!\\ no measurement-validation export: measurement columns left empty")
        return pd.DataFrame(columns=[KEY])

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        # The app writes the image as a full local path, under the name 'image'.
        column = "image" if "image" in frame.columns else KEY
        if column not in frame.columns:
            print(f"/!\\ {path.name}: neither 'image' nor '{KEY}', file skipped")
            continue
        frame[KEY] = frame[column].map(_basename)
        frame = frame.drop(columns=[c for c in (column,) if c != KEY])
        frame["measurements_source"] = path.name
        frames.append(frame)
        print(f"  {len(frame):5} rows  {path.name}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[KEY])


def deduplicate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """One row per image: a second annotation of the same image is dropped, loudly."""
    if frame.empty:
        return frame
    frame = frame.copy()
    frame[KEY] = frame[KEY].astype(str).map(_basename)
    duplicated = frame[KEY].duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, KEY].unique()[:3]
        print(f"/!\\ {label}: {int(duplicated.sum())} duplicate image(s), "
              f"only the first row is kept (e.g. {list(examples)})")
        frame = frame[~duplicated]
    return frame


def merge(pose: pd.DataFrame, scale: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    """Outer join of the three sources on the image name."""
    merged = pose
    for other, label in ((scale, "scale"), (measurements, "measurements")):
        if other.empty:
            continue
        overlap = [c for c in other.columns if c != KEY and c in merged.columns]
        if overlap:
            print(f"/!\\ columns duplicated between pose and {label}, "
                  f"those of {label} get the suffix '_{label}': {overlap}")
            other = other.rename(columns={c: f"{c}_{label}" for c in overlap})
        merged = merged.merge(other, on=KEY, how="outer")
    return merged.sort_values(KEY).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather pose, scale and measurement validation into a single CSV.")
    parser.add_argument("--pose", default=str(DEFAULT_POSE),
                        help="Pose CSV, produced by labelstudio_to_csv.py "
                             f"(default: {DEFAULT_POSE.relative_to(REPO_ROOT)}).")
    parser.add_argument("--scale", default=str(DEFAULT_SCALE),
                        help=f"Scale CSV (default: {DEFAULT_SCALE.relative_to(REPO_ROOT)}).")
    parser.add_argument("--measurements", nargs="*", default=[str(DEFAULT_MEASUREMENTS)],
                        help="Measurement-validation CSVs, and/or folders "
                             f"(default: {DEFAULT_MEASUREMENTS.relative_to(REPO_ROOT)}).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Combined CSV (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).")
    args = parser.parse_args()

    print("pose:")
    pose = deduplicate(load_pose(Path(args.pose)), "pose")
    print(f"  {len(pose):5} images")
    print("scale:")
    scale = deduplicate(load_scale(Path(args.scale)), "scale")
    print(f"  {len(scale):5} images")
    print("measurements:")
    measurements = deduplicate(load_measurements(args.measurements), "measurements")
    print(f"  {len(measurements):5} images")

    merged = merge(pose, scale, measurements)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    filled = merged.notna().sum()
    print(f"\n{len(merged)} images, {len(merged.columns)} columns -> {output}")
    for label, column in (("pose", "task_id"), ("scale", "scale_px_per_mm"),
                          ("measures", "measurements_source")):
        if column in merged.columns:
            print(f"  {label:8}: {int(filled[column]):5} image(s) filled in")


if __name__ == "__main__":
    main()
