#!/usr/bin/env python3
"""
labelstudio_to_csv.py
=====================

Convert Label Studio keypoint exports (JSON) into a single flat CSV.

One row per annotation (in practice, one per annotated image). Keypoint
coordinates are written in ABSOLUTE PIXELS, so the CSV is independent of the
Label Studio percentage encoding and can feed any later step (YOLO dataset
building, measurements, statistics...).

Columns
-------
    image_name        file name of the image (Label Studio upload hash removed)
    image_path        path decoded from data.img when it holds a local path,
                      otherwise the raw data.img value
    source_json       name of the export the row comes from
    group             insect group (default: name of the folder holding the JSON)
    task_id           Label Studio task id
    annotation_id     Label Studio annotation id
    width, height     original image size, in pixels
    image_rotation    rotation applied in Label Studio, in degrees
    remarks           content of the "textarea" result, if any
    bbox_x, bbox_y, bbox_w, bbox_h
                      bounding box in pixels, only if the export holds a
                      "rectanglelabels" result (empty otherwise)
    <keypoint>_x, <keypoint>_y, <keypoint>_v
                      one triplet per keypoint of KEYPOINT_NAMES;
                      v = 2 when the keypoint was annotated, 0 when missing

Usage
-----
    python labelstudio_to_csv.py --json_files export.json
    python labelstudio_to_csv.py --json_files ../annotation_data/label_studio_annotations \\
                                 --output annotations.csv

--json_files accepts any number of JSON files and/or folders (folders are
searched recursively). All inputs are merged into a single CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import sys
from pathlib import Path
from urllib.parse import unquote

# Keypoint order (kp_infos.yaml): it defines the CSV column order and must stay in
# sync with the Label Studio labelling config (annotation_tools/label_studio_template.txt).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kp_infos import KEYPOINT_NAMES  # noqa: E402

META_COLUMNS = [
    "image_name", "image_path", "source_json", "group",
    "task_id", "annotation_id", "width", "height", "image_rotation", "remarks",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]

VISIBLE = 2
MISSING = 0

_LOCAL_FILES_PREFIX = "/data/local-files/?d="
_HASH_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8}-")
_FOUND_DRIVE = None  # drive letter of the last path resolved (see _restore_drive_letter)


def _restore_drive_letter(path: str) -> str:
    """Add back the Windows drive letter dropped by Label Studio local storage.

    Label Studio serves local files relative to LOCAL_FILES_DOCUMENT_ROOT, "/" by
    default: an exported path then looks like "Users/tombe/.../img.png". The
    drive is looked for among the available ones, and the path is left untouched
    when it cannot be found.
    """
    global _FOUND_DRIVE
    if os.name != "nt" or Path(path).anchor:
        return path
    letters = string.ascii_uppercase if _FOUND_DRIVE is None else _FOUND_DRIVE + string.ascii_uppercase
    for letter in letters:
        candidate = f"{letter}:/{path}"
        if Path(candidate).exists():
            _FOUND_DRIVE = letter
            return candidate
    return path


def image_name_of(data_img: str) -> str:
    """File name of the data.img field of a task, Label Studio upload hash removed.

    Reads nothing from the disk: this is the name every other annotation file uses
    (annotation_tools/check_annotations.py relies on it).
    """
    if data_img.startswith(_LOCAL_FILES_PREFIX):
        path = unquote(data_img[len(_LOCAL_FILES_PREFIX):])
    else:
        path = data_img.replace("%5C", "/")
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return _HASH_PREFIX_RE.sub("", name)


def decode_image(data_img: str):
    """Return (image_name, image_path) for the data.img field of a task.

    data.img comes in two flavours depending on how the images were imported:
      - "/data/local-files/?d=<url-encoded absolute path>" (local storage)
      - "/data/upload/<project>/<hash>-<name>"             (uploaded files)
    In the second case no local path is known, so the raw value is kept.
    """
    if data_img.startswith(_LOCAL_FILES_PREFIX):
        path = unquote(data_img[len(_LOCAL_FILES_PREFIX):]).replace("\\", "/")
        return image_name_of(data_img), _restore_drive_letter(path)
    return image_name_of(data_img), data_img


def parse_annotation(results: list) -> dict:
    """Extract size, keypoints (in pixels), bbox and remarks from a result list."""
    width = height = None
    rotation = 0
    keypoints = {}
    bbox = None
    remarks = ""
    unknown_labels = set()

    for res in results:
        res_type = res.get("type")
        value = res.get("value", {})
        res_w = res.get("original_width")
        res_h = res.get("original_height")

        if res_type == "keypointlabels":
            labels = value.get("keypointlabels") or []
            if not labels or not res_w or not res_h:
                continue
            width, height = res_w, res_h
            rotation = res.get("image_rotation", 0) or 0
            label = labels[0]
            if label not in KEYPOINT_NAMES:
                unknown_labels.add(label)
                continue
            keypoints[label] = (
                round(value["x"] / 100.0 * res_w, 2),
                round(value["y"] / 100.0 * res_h, 2),
            )

        elif res_type == "rectanglelabels" and res_w and res_h:
            width, height = res_w, res_h
            bbox = (
                round(value.get("x", 0) / 100.0 * res_w, 2),
                round(value.get("y", 0) / 100.0 * res_h, 2),
                round(value.get("width", 0) / 100.0 * res_w, 2),
                round(value.get("height", 0) / 100.0 * res_h, 2),
            )

        elif res_type == "textarea":
            texts = value.get("text") or []
            if texts:
                remarks = " ".join(str(t) for t in texts)

    return {
        "width": width,
        "height": height,
        "rotation": round(rotation / 90.0) % 4 * 90,
        "keypoints": keypoints,
        "bbox": bbox,
        "remarks": remarks,
        "unknown_labels": unknown_labels,
    }


def rows_from_export(json_path: Path, group: str | None):
    """Yield one CSV row per non-cancelled, non-empty annotation of an export."""
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    group = group if group is not None else json_path.parent.name
    unknown_labels = set()
    skipped = 0

    for task in tasks:
        image_name, image_path = decode_image(task.get("data", {}).get("img", ""))

        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            parsed = parse_annotation(annotation.get("result", []))
            unknown_labels |= parsed["unknown_labels"]

            if not parsed["keypoints"]:
                skipped += 1
                continue

            row = {
                "image_name": image_name,
                "image_path": image_path,
                "source_json": json_path.name,
                "group": group,
                "task_id": task.get("id", ""),
                "annotation_id": annotation.get("id", ""),
                "width": parsed["width"],
                "height": parsed["height"],
                "image_rotation": parsed["rotation"],
                "remarks": parsed["remarks"],
            }
            bbox = parsed["bbox"]
            for col, val in zip(("bbox_x", "bbox_y", "bbox_w", "bbox_h"), bbox or ("", "", "", "")):
                row[col] = val

            for name in KEYPOINT_NAMES:
                x, y = parsed["keypoints"].get(name, (0, 0))
                row[f"{name}_x"] = x
                row[f"{name}_y"] = y
                row[f"{name}_v"] = VISIBLE if name in parsed["keypoints"] else MISSING

            yield row

    if unknown_labels:
        print(f"  /!\\ labels absent from KEYPOINT_NAMES, ignored: {sorted(unknown_labels)}")
    if skipped:
        print(f"  {skipped} annotation(s) without keypoints ignored")


def collect_json_files(inputs) -> list:
    """Expand the --json_files arguments into a sorted list of JSON files."""
    files = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"/!\\ ignored (not found): {path}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio keypoint exports into a single CSV.")
    parser.add_argument("--json_files", nargs="+", required=True,
                        help="Label Studio JSON exports, and/or folders holding them.")
    parser.add_argument("--output", default="annotations.csv",
                        help="Output CSV path (default: annotations.csv).")
    parser.add_argument("--group", default=None,
                        help="Insect group written in the 'group' column "
                             "(default: name of the folder holding each JSON).")
    args = parser.parse_args()

    json_files = collect_json_files(args.json_files)
    if not json_files:
        parser.error("no JSON file found in --json_files")

    header = META_COLUMNS + [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y", "v")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for json_path in json_files:
            print(f"{json_path}")
            count = 0
            for row in rows_from_export(json_path, args.group):
                writer.writerow(row)
                count += 1
            print(f"  {count} annotation(s) written")
            total += count

    print(f"\n{total} annotation(s) from {len(json_files)} export(s) -> {output}")


if __name__ == "__main__":
    main()
