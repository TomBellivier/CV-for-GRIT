"""annotation_data.csv -> canonical format adapter (contract 1, CONVENTIONS.md §3.2).

Reads the single annotation table of the repository,
``annotation_data/annotation_data.csv`` (see
``annotation_tools/build_annotation_data.py``): one row per image, the keypoints in
absolute pixels in ``<point>_x`` / ``<point>_y`` / ``<point>_v`` columns, and the
``group`` column for the insect order.

The scale and measurement-validity columns of that table do not concern the pose:
they are ignored here, without being modified.

Like any adapter: reads, converts, does not filter and decides nothing. Rows without
any keypoint are the only ones left out -- they are not pose annotations -- and, only
with ``skip_missing_images: true``, the rows whose image file is absent (reported).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.adapters.base import BaseAdapter
from insectpose.data.keypoints import load_schema
from insectpose.registry import register_adapter
from insectpose.utils.logging import get_logger

log = get_logger("adapter")

_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@register_adapter("annotation_csv")
class AnnotationCsvAdapter(BaseAdapter):
    """Annotation table of the repository -> canonical annotations of a dataset.

    Options (``data.adapter_options``):
        csv_path        path of the CSV, absolute or relative to ``source_dir``
                        (default: ``annotation_data.csv``)
        images_subdir   folder of the images, relative to ``paths.data``, where the
                        file names of the CSV are looked for
                        (default: ``raw/<dataset>/images``)
        group_column    column holding the insect order (default: ``group``)
        margin          margin, in pixels, around the keypoints for the bbox when
                        the CSV carries none (default: 10)
        keypoint_schema name of the schema (default: ``insect42_v1``)
        skip_missing_images
                        leave out, with a warning, the rows whose image is not in
                        ``images_subdir`` (default: false; the YOLO export then fails
                        on the first missing image)
    """

    def _schema_points(self) -> list[str]:
        """Order of the keypoints of the schema (§3.1): it sets the order of the columns read.

        The order comes from the schema, never from the order of the CSV columns: a
        keypoint index is encoded in every artefact produced afterwards.
        """
        configs_dir = self.options.get("configs_dir")
        if not configs_dir:
            raise KeyError(
                "adapter_options.configs_dir missing: add "
                "'configs_dir: ${paths.configs}' to the configs/data/ file of the dataset."
            )
        name = str(self.options.get("keypoint_schema", "insect42_v1"))
        return list(load_schema(name, Path(configs_dir)).names)

    def read(self) -> pd.DataFrame:
        csv_path = Path(self.options.get("csv_path", "annotation_data.csv"))
        if not csv_path.is_absolute():
            # Relative to the project root (injected by cmd_prepare), not to the current
            # folder: the command must work wherever it is launched from.
            csv_path = (Path(self.options.get("project_root", self.source_dir)) / csv_path).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Annotation table missing: {csv_path}. Build it with "
                "python annotation_tools/build_annotation_data.py"
            )

        frame = pd.read_csv(csv_path)
        points = self._schema_points()
        missing = [p for p in points if f"{p}_x" not in frame.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} keypoint(s) of the schema missing from {csv_path.name} "
                f"(e.g. {missing[:3]}): the table does not follow the same skeleton."
            )

        group_column = str(self.options.get("group_column", "group"))
        if group_column in frame.columns:
            frame = frame[frame[group_column].astype(str).str.lower() == self.dataset]
        images_subdir = str(self.options.get("images_subdir", f"raw/{self.dataset}/images"))
        margin = float(self.options.get("margin", 10))
        schema_name = str(self.options.get("keypoint_schema", "insect42_v1"))
        skip_missing = bool(self.options.get("skip_missing_images", False))
        # images_subdir is relative to paths.data (the parent of paths.raw by default).
        data_dir = Path(self.options.get("data_dir", self.source_dir.parent.parent))

        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        for _, record in frame.iterrows():
            if skip_missing and not (data_dir / images_subdir / str(record["image_name"])).is_file():
                skipped.append(str(record["image_name"]))
                continue
            xy = np.array([[record[f"{p}_x"], record[f"{p}_y"]] for p in points], dtype=float)
            vis = np.array([record.get(f"{p}_v", 0) for p in points], dtype=float)
            vis = np.nan_to_num(vis, nan=0.0).astype(int)
            # A point that is not annotated comes out as (0, 0, 0): it is the convention
            # of the contract, and it keeps a missing coordinate from moving the bbox.
            absent = (vis == 0) | ~np.isfinite(xy).all(axis=1)
            xy[absent] = 0.0
            if not (~absent).any():
                continue

            width, height = record.get("width"), record.get("height")
            if not (np.isfinite(width) and np.isfinite(height)):
                raise ValueError(
                    f"Image size missing for {record['image_name']}: "
                    "the width/height column of the CSV is empty."
                )
            width, height = int(width), int(height)

            box = self._bbox(record, xy[~absent], margin, width, height)
            stem = Path(str(record["image_name"])).stem
            image_id = f"{self.dataset}/{stem}"
            rows.append({
                "dataset": self.dataset,
                "image_id": image_id,
                "image_path": f"{images_subdir}/{record['image_name']}",
                "image_width": width,
                "image_height": height,
                "instance_id": f"{image_id}#0",     # ADR-0017: one image = one insect
                "group_id": image_id,
                "bbox_xywh": [float(v) for v in box],
                "kpts_xy": [float(v) for v in xy.reshape(-1)],
                "kpts_vis": [int(v) for v in vis],
                "area": float(box[2] * box[3]),
                "keypoint_schema": schema_name,
                "split_source": "unknown",
            })

        if skipped:
            log.warning("[%s] %d annotated image(s) missing from %s, left out of the "
                        "dataset: %s", self.dataset, len(skipped), data_dir / images_subdir,
                        ", ".join(sorted(skipped)))
        return pd.DataFrame(rows)

    @staticmethod
    def _bbox(record: pd.Series, visible_xy: np.ndarray, margin: float,
              width: int, height: int) -> tuple[float, float, float, float]:
        """bbox of the CSV if it has one, else the envelope of the keypoints + margin."""
        values = [record.get(c) for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]
        if all(v is not None and np.isfinite(v) for v in values) and values[2] > 0:
            return tuple(float(v) for v in values)

        x_min, y_min = visible_xy.min(axis=0)
        x_max, y_max = visible_xy.max(axis=0)
        x = max(0.0, x_min - margin)
        y = max(0.0, y_min - margin)
        return (x, y, min(width, x_max + margin) - x, min(height, y_max + margin) - y)
