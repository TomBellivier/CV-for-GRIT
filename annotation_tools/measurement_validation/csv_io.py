"""CSV inputs/outputs.

Input: the CSV produced by annotation_tools/labelstudio_to_csv.py (one row per
annotation, keypoints in absolute pixels in the "<kp>_x/_y/_v" columns).

Output: a status CSV, one row per annotation, one "<measurement>_status" column per
measurement — it is the only file used afterwards
(modules/meas_classifier/dataset.py). No keypoint position appears in it.

The detailed classification state (status per edge, validated images) is saved
separately, as JSON, only to be able to resume the work where it stopped.
"""
import csv
import json
from pathlib import Path

from app_logging import logger
from models import ImageAnnotation, MEASURABLE, NON_MEASURABLE, edge_key

# The CSV of labelstudio_to_csv.py writes v=0 for a missing keypoint.
_MISSING = "0"


def _keypoint_names(fieldnames):
    """Derive the list of keypoints from the "<kp>_v" columns of the input CSV."""
    names = []
    for col in fieldnames or []:
        if col.endswith("_v"):
            name = col[:-2]
            if f"{name}_x" in fieldnames and f"{name}_y" in fieldnames:
                names.append(name)
    return names


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_annotations(csv_path):
    """Load the annotation CSV. Returns a list of ImageAnnotation."""
    csv_path = Path(csv_path)
    logger.info("load_annotations: reading %s", csv_path)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        kp_names = _keypoint_names(reader.fieldnames)
        if not kp_names:
            raise ValueError(
                f"{csv_path} contains no keypoint column (<kp>_x/_y/_v): "
                "expected a CSV produced by labelstudio_to_csv.py"
            )

        annotations = []
        for row in reader:
            keypoints = {}
            for name in kp_names:
                if (row.get(f"{name}_v") or _MISSING).strip() == _MISSING:
                    continue
                try:
                    keypoints[name] = (float(row[f"{name}_x"]), float(row[f"{name}_y"]))
                except (TypeError, ValueError):
                    continue
            if not keypoints:
                continue

            image_name = row.get("image_name") or Path(row.get("image_path", "")).name
            annotations.append(ImageAnnotation(
                image_name=image_name,
                image_path=row.get("image_path", ""),
                width=_to_int(row.get("width")),
                height=_to_int(row.get("height")),
                annotation_id=(row.get("annotation_id") or "").strip(),
                image_rotation=_to_int(row.get("image_rotation")) % 360,
                keypoints=keypoints,
            ))

    annotations.sort(key=lambda a: a.image_name)
    logger.info("load_annotations: %d annotation(s) loaded, %d possible keypoint(s)",
                len(annotations), len(kp_names))
    return annotations


def export_status_csv(annotations, config, out_path):
    """Write the final file: one row per annotation, the status of each measurement."""
    measurement_names = config.measurement_names()
    fieldnames = ["image"] + [f"{m}_status" for m in measurement_names]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ann in annotations:
            row = {"image": ann.image_name}
            for m_name in measurement_names:
                row[f"{m_name}_status"] = ann.measurement_status(config.measurement_edges[m_name])
            writer.writerow(row)
    logger.info("export_status_csv: %d row(s) written to %s", len(annotations), out_path)


def save_state(annotations, out_path):
    """Save the status per edge and the validated images, to be able to resume the
    classification exactly where it stopped (the exported CSV only keeps the status
    per measurement: it would not allow a faithful restoration)."""
    data = {}
    for ann in annotations:
        if not ann.edge_overrides and not ann.done:
            continue
        entry = {}
        if ann.edge_overrides:
            entry["edges"] = {f"{a}::{b}": status for (a, b), status in ann.edge_overrides.items()}
        if ann.done:
            entry["done"] = True
        data[ann.key()] = entry

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_state(state_path, annotations):
    """Restore on `annotations` (in place) the state saved by save_state."""
    state_path = Path(state_path)
    if not state_path.exists():
        return
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Cannot read the classification state: %s", state_path)
        return

    by_key = {a.key(): a for a in annotations}
    restored = 0
    for key, entry in data.items():
        ann = by_key.get(key)
        if ann is None:
            continue
        for edge_str, status in entry.get("edges", {}).items():
            a, b = edge_str.split("::", 1)
            ann.edge_overrides[edge_key(a, b)] = (
                MEASURABLE if status == MEASURABLE else NON_MEASURABLE
            )
        ann.done = bool(entry.get("done"))
        restored += 1
    logger.info("load_state: state restored for %d/%d annotation(s)", restored, len(annotations))
