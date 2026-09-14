#!/usr/bin/env python3
"""
generate_ground_truth_csv.py
=============================

Standalone script (no dependency on process_folder / conf_classifier): walks
every annotated pose image under
    models/datasets/<insect_group>/labels/<split>/*.txt
(YOLO-pose format: class cx cy w h  x1 y1 v1  x2 y2 v2 ...) and writes one CSV
row per image with the ground-truth length of every measurement, in PIXELS,
computed directly from the annotated keypoints (no scale annotation exists in
these label files, so no mm conversion is done here).

Usage
-----
    python generate_ground_truth_csv.py
    python generate_ground_truth_csv.py --datasets-root ../datasets --output ground_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image

INSECT_GROUPS = ["coleoptera", "diptera", "hymenoptera", "lepidoptera"]
SPLITS = ["train", "val", "test"]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# --- skeleton (must match the order the datasets were annotated in) --------
KEYPOINT_NAMES = [
    "head-top", "head-left", "head-right", "left-eye", "right-eye", "neck",
    "thorax-left", "thorax-right", "thorax-bottom", "body-left", "body-right", "body-tip",
    "left-antenna-0", "left-antenna-1", "left-antenna-2",
    "right-antenna-0", "right-antenna-1", "right-antenna-2",
    "left-forewing-base", "left-forewing-tip", "left-forewing-front", "left-forewing-rear",
    "right-forewing-base", "right-forewing-tip", "right-forewing-front", "right-forewing-rear",
    "left-hindwing-base", "left-hindwing-tip", "left-hindwing-front", "left-hindwing-rear",
    "right-hindwing-base", "right-hindwing-tip", "right-hindwing-front", "right-hindwing-rear",
    "left-leg-0", "left-leg-1", "left-leg-2", "left-leg-3",
    "right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3",
]
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

MEASUREMENTS = {
    "total length":                 ["head-top", "neck", "thorax-bottom", "body-tip"],
    "head width":                   ["head-left", "head-right"],
    "head length":                  ["head-top", "neck"],
    "inter ocular distance":        ["right-eye", "left-eye"],
    "right antenna length":         ["right-antenna-0", "right-antenna-1", "right-antenna-2"],
    "left antenna length":          ["left-antenna-0", "left-antenna-1", "left-antenna-2"],
    "thorax width":                 ["thorax-left", "thorax-right"],
    "thorax length":                ["neck", "thorax-bottom"],
    "abdomen width":                ["body-left", "body-right"],
    "abdomen length":               ["thorax-bottom", "body-tip"],
    "intertegular distance":        ["left-forewing-base", "right-forewing-base"],
    "left hind wing length":        ["left-hindwing-base", "left-hindwing-tip"],
    "right hind wing length":       ["right-hindwing-base", "right-hindwing-tip"],
    "left hind wing width":         ["left-hindwing-front", "left-hindwing-rear"],
    "right hind wing width":        ["right-hindwing-front", "right-hindwing-rear"],
    "left fore wing length":        ["left-forewing-base", "left-forewing-tip"],
    "right fore wing length":       ["right-forewing-base", "right-forewing-tip"],
    "left fore wing width":         ["left-forewing-front", "left-forewing-rear"],
    "right fore wing width":        ["right-forewing-front", "right-forewing-rear"],
    "left hind leg length":         ["left-leg-0", "left-leg-1", "left-leg-2", "left-leg-3"],
    "left hind leg femur length":   ["left-leg-0", "left-leg-1"],
    "left hind leg tibia length":   ["left-leg-1", "left-leg-2"],
    "left hind leg tarsus length":  ["left-leg-2", "left-leg-3"],
    "right hind leg length":        ["right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3"],
    "right hind leg femur length":  ["right-leg-0", "right-leg-1"],
    "right hind leg tibia length":  ["right-leg-1", "right-leg-2"],
    "right hind leg tarsus length": ["right-leg-2", "right-leg-3"],
}
MEASUREMENT_NAMES = list(MEASUREMENTS.keys())


def parse_label_file(path: Path):
    """Return the largest-box instance as (keypoints_xy_norm, visibility) or None."""
    best, best_area = None, -1.0
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = list(map(float, parts[1:]))
        except ValueError:
            continue
        w, h = vals[2], vals[3]
        kp = vals[4:]
        if len(kp) == NUM_KEYPOINTS * 3:
            step = 3
        elif len(kp) == NUM_KEYPOINTS * 2:
            step = 2
        else:
            continue
        xs, ys = kp[0::step][:NUM_KEYPOINTS], kp[1::step][:NUM_KEYPOINTS]
        vis = kp[2::step][:NUM_KEYPOINTS] if step == 3 else None
        area = w * h
        if area > best_area:
            best_area = area
            best = (list(zip(xs, ys)), vis)
    return best


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTENSIONS:
        for candidate in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


def measurement_lengths_px(xy_px, vis) -> dict:
    lengths = {}
    for name, chain in MEASUREMENTS.items():
        idxs = [KEYPOINT_INDEX[kp] for kp in chain]
        if vis is not None and any(vis[i] == 0 for i in idxs):
            lengths[name] = ""
            continue
        total = 0.0
        for a, b in zip(idxs[:-1], idxs[1:]):
            (x1, y1), (x2, y2) = xy_px[a], xy_px[b]
            total += math.hypot(x2 - x1, y2 - y1)
        lengths[name] = round(total, 2)
    return lengths


def main():
    parser = argparse.ArgumentParser(
        description="Build a ground-truth measurement CSV (pixels) from the annotated pose datasets.")
    parser.add_argument("--datasets-root", default="./datasets",
                        help="Root holding <insect_group>/images|labels/<split> "
                             "(default: ../datasets, i.e. models/datasets).")
    parser.add_argument("--output", default="ground_truth.csv", help="Output CSV path.")
    args = parser.parse_args()

    root = Path(args.datasets_root)
    rows = []

    for group in INSECT_GROUPS:
        group_dir = root / group
        if not group_dir.is_dir():
            print(f"[skip] {group_dir} not found")
            continue
        for split in SPLITS:
            labels_dir = group_dir / "labels" / split
            images_dir = group_dir / "images" / split
            if not labels_dir.is_dir():
                continue
            for label_path in sorted(labels_dir.glob("*.txt")):
                image_path = find_image(images_dir, label_path.stem)
                if image_path is None:
                    print(f"[skip] no image for {label_path}")
                    continue
                instance = parse_label_file(label_path)
                if instance is None:
                    print(f"[skip] no valid instance in {label_path}")
                    continue
                xy_norm, vis = instance
                with Image.open(image_path) as im:
                    W, H = im.size
                xy_px = [(x * W, y * H) for x, y in xy_norm]

                row = {"image_name": image_path.name, "insect_group": group, "split": split}
                row.update(measurement_lengths_px(xy_px, vis))
                rows.append(row)

    out_path = Path(args.output)
    header = ["image_name", "insect_group", "split"] + MEASUREMENT_NAMES
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} annotated image(s) written to {out_path}")


if __name__ == "__main__":
    main()
