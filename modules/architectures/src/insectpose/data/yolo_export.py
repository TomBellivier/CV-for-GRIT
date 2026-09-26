"""Export of the canonical format to the YOLO-pose format (CONVENTIONS.md §9.1).

DERIVED operation: the files produced are regenerated at each fold from the shared
splits, and written to the run folder (or to `data/interim/`), never to
`data/processed/`. The canonical format stays the only source of truth.

Format of a YOLO-pose label, one line per instance, everything normalised in [0, 1]:
    class cx cy w h  x1 y1 v1  x2 y2 v2  ...  xK yK vK
The bbox is CENTRED (cx, cy), unlike contract 1 which uses the top-left corner. It is
exactly the kind of divergence this module isolates in a single place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from insectpose.contracts import ContractError
from insectpose.data.keypoints import KeypointSchema
from insectpose.utils.logging import get_logger

log = get_logger("yolo_export")

CLASS_NAMES = {0: "insect"}   # a single class: "insect" (§9.1)


def flat_name(image_id: str) -> str:
    """Flat, unique file name from an image_id `<dataset>/<stem>`.

    Without this flattening, two datasets both having an `img001.png` would silently
    overwrite each other in the YOLO folder.
    """
    return str(image_id).replace("/", "__")


def to_label_lines(instances: pd.DataFrame, width: int, height: int,
                   n_keypoints: int, with_keypoints: bool = True) -> tuple[list[str], int]:
    """YOLO label lines of an image. Returns (lines, number of clipped values).

    Coordinates outside the image are clipped into [0, 1] (constraint of the format) and
    the count is reported: a massive clipping signals doubtful annotations.
    """
    lines: list[str] = []
    clipped = 0
    for row in instances.itertuples(index=False):
        x, y, w, h = np.asarray(row.bbox_xywh, dtype=float)
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        nw, nh = w / width, h / height
        box = np.array([cx, cy, nw, nh])
        clipped += int((box < 0).sum() + (box > 1).sum())
        box = np.clip(box, 0.0, 1.0)

        kpts = np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2)
        vis = np.asarray(row.kpts_vis, dtype=int)
        if len(kpts) != n_keypoints:
            raise ContractError(
                f"{row.instance_id}: {len(kpts)} keypoints for a schema of {n_keypoints}."
            )
        norm = kpts / np.array([width, height])
        clipped += int((norm[vis > 0] < 0).sum() + (norm[vis > 0] > 1).sum())
        norm = np.clip(norm, 0.0, 1.0)
        # A point not annotated is written (0, 0, 0): it is the YOLO convention for
        # "unsupervised". It is masked in the loss, never learnt as a zero.
        norm[vis == 0] = 0.0

        values = [0, *box.tolist()]
        if with_keypoints:
            for (px, py), v in zip(norm, vis, strict=True):
                values.extend([px, py, int(v)])
        lines.append(" ".join(
            str(v) if isinstance(v, int) else f"{v:.6f}" for v in values
        ))
    return lines, clipped


def parse_label_line(line: str, width: int, height: int) -> dict[str, Any]:
    """Read a YOLO label line back into the frame of the original image.

    Exact inverse of `to_label_lines`: it is what makes the export testable by a round
    trip, without ever launching a training.
    """
    parts = [float(v) for v in line.split()]
    cx, cy, nw, nh = parts[1:5]
    kpt_values = np.asarray(parts[5:], dtype=float).reshape(-1, 3)
    x = (cx - nw / 2) * width
    y = (cy - nh / 2) * height
    return {
        "class": int(parts[0]),
        "bbox_xywh": [x, y, nw * width, nh * height],
        "kpts_xy": (kpt_values[:, :2] * np.array([width, height])).reshape(-1).tolist(),
        "kpts_vis": kpt_values[:, 2].astype(int).tolist(),
    }


def export_split(image_set: Any, schema: KeypointSchema, root: Path, split: str,
                 link_images: bool = True, with_keypoints: bool = True) -> int:
    """Write images/<split>/ and labels/<split>/ for an ImageSet.

    The images are linked as they are: the canonical format stays the only source of
    truth and the coordinates undergo no scale transform.

    Side effect: creates images/ and labels/ under `root`. Returns the number of images.
    """
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    total_clipped = 0
    images = image_set.images.set_index("image_id")
    for image_id, group in image_set.annotations.groupby("image_id"):
        meta = images.loc[image_id]
        source = image_set.absolute_path(meta.image_path)
        target = img_dir / f"{flat_name(image_id)}{Path(str(meta.image_path)).suffix}"
        if not target.exists():
            if not source.exists():
                raise FileNotFoundError(
                    f"Image missing: {source}. The YOLO export requires the real images."
                )
            if link_images:
                target.symlink_to(source)
            else:
                target.write_bytes(source.read_bytes())
        lines, clipped = to_label_lines(
            group, int(meta.image_width), int(meta.image_height), schema.n_keypoints,
            with_keypoints=with_keypoints,
        )
        total_clipped += clipped
        (lbl_dir / f"{flat_name(image_id)}.txt").write_text("\n".join(lines) + "\n",
                                                            encoding="utf-8")
    if total_clipped:
        log.warning("[%s] %d coordinate(s) clipped into [0,1] at the YOLO export: "
                    "check the annotations outside the image.", split, total_clipped)
    return len(images)


def write_data_yaml(root: Path, schema: KeypointSchema, splits: dict[str, str],
                    with_keypoints: bool = True) -> Path:
    """Write the Ultralytics data.yaml. Side effect: creates <root>/data.yaml.

    `flip_idx` is MANDATORY as soon as a mirror augmentation is active: without it, a
    mirror swaps the left/right sides without permuting the labels and the training
    learns a wrong anatomy (§3.1).
    """
    payload: dict[str, Any] = {"path": str(root.resolve()), "names": CLASS_NAMES}
    if with_keypoints:
        payload["kpt_shape"] = [schema.n_keypoints, 3]
        payload["flip_idx"] = list(schema.flip_index)
    payload.update({split: f"images/{name}" for split, name in splits.items()})
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def export_fold(data: Any, schema: KeypointSchema, root: Path,
                splits: tuple[str, ...] = ("train", "val"),
                link_images: bool = True, with_keypoints: bool = True) -> Path:
    """Export a FoldData in the YOLO format and return the path of the data.yaml.

    Side effect: creates the YOLO tree under `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        n = export_split(data.role(split), schema, root, split, link_images, with_keypoints)
        log.info("YOLO export [%s]: %d images -> %s", split, n, root / "images" / split)
    return write_data_yaml(root, schema, {s: s for s in splits}, with_keypoints)