"""Locate the annotations whose number of keypoints deviates from the schema.

Run from the project root, after `prepare`:
    python check_schema.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

EXPECTED = 42

for file in sorted(glob.glob("data/processed/*/annotations.parquet")):
    frame = pd.read_parquet(file)
    frame["K"] = frame["kpts_xy"].map(len) // 2
    dataset = frame["dataset"].iloc[0]
    counts = dict(frame["K"].value_counts().sort_index())
    print(f"\n{dataset:14s} {len(frame):5d} instances | keypoints: {counts}")

    outliers = frame[frame["K"] != EXPECTED]
    if outliers.empty:
        continue
    print(f"  {len(outliers)} diverging instance(s):")
    for _, row in outliers.head(20).iterrows():
        # The original YOLO label is derived from the image path
        image = Path(str(row["image_path"]))
        label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
        print(f"    {row['K']:4d} kpts | {image}")
        print(f"           expected label: data/{label}")
    if len(outliers) > 20:
        print(f"    ... and {len(outliers) - 20} more")
