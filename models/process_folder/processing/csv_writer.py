"""
csv_writer.py
=============

Build the output CSV. One row per image.

Columns (in order):
    image name
    in_train, in_val
    <measurement> [px]     for every measurement          (pixel measurements)
    <measurement> [mm]     for every measurement          (converted measurements)
    <measurement> [conf]   for every measurement          (measurement confidences)
    overall_pose_confidence
    scale_px_per_mm
    scale_confidence
    <optional columns>     (only those enabled in config.OPTIONAL_COLUMNS)

Each per-image "record" is a plain dict (see build_record in process_folder.py).
Only the columns you requested are written by default; the optional ones stay
off until you flip them in config.OPTIONAL_COLUMNS.
"""

from __future__ import annotations

import csv
import math

from . import config
from .definitions import MEASUREMENT_NAMES


# Suffixes used to disambiguate the three columns of each measurement.
PX_SUFFIX = " [px]"
MM_SUFFIX = " [mm]"
CONF_SUFFIX = " [conf]"

# Optional columns, in a stable display order.
_OPTIONAL_ORDER = [
    "scale_method",
    "n_instances",
    "detection_confidence",
    "image_width",
    "image_height",
    "needs_review",
]


def _enabled_optional_columns() -> list[str]:
    return [c for c in _OPTIONAL_ORDER if config.OPTIONAL_COLUMNS.get(c, False)]


def build_header() -> list[str]:
    """Return the ordered list of column names."""
    header = ["image_name", "in_train", "in_val"]
    header += [m + PX_SUFFIX for m in MEASUREMENT_NAMES]
    header += [m + MM_SUFFIX for m in MEASUREMENT_NAMES]
    header += [m + CONF_SUFFIX for m in MEASUREMENT_NAMES]
    header += ["overall_pose_confidence", "scale_px_per_mm", "scale_confidence"]
    header += _enabled_optional_columns()
    return header


def _fmt(value) -> str:
    """Format a cell: blank for missing values, plain text otherwise."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def build_row(record: dict) -> list[str]:
    """Turn a per-image record dict into a list of formatted cells."""
    row = [
        _fmt(record.get("image_name")),
        _fmt(record.get("in_train")),
        _fmt(record.get("in_val")),
    ]
    pixels = record.get("pixels", {})
    mm = record.get("mm", {})
    conf = record.get("conf", {})

    row += [_fmt(pixels.get(m)) for m in MEASUREMENT_NAMES]
    row += [_fmt(mm.get(m)) for m in MEASUREMENT_NAMES]
    row += [_fmt(conf.get(m)) for m in MEASUREMENT_NAMES]

    row += [
        _fmt(record.get("overall_pose_confidence")),
        _fmt(record.get("scale_px_per_mm")),
        _fmt(record.get("scale_confidence")),
    ]

    for col in _enabled_optional_columns():
        row.append(_fmt(record.get(col)))
    return row


class CsvWriter:
    """Small helper that opens the file, writes the header, then rows."""

    def __init__(self, output_path):
        self.output_path = str(output_path)
        self._fh = None
        self._writer = None

    def __enter__(self):
        self._fh = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(build_header())
        return self

    def write_record(self, record: dict):
        self._writer.writerow(build_row(record))

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            self._fh.close()
