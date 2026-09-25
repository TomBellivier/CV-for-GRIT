"""Export d'instances recadrees au format YOLO-pose (CONVENTIONS.md §9.3).

Le modele de pose de l'approche C ne voit jamais l'image entiere : il travaille sur des
crops normalises. Deux precautions decident de la validite de l'approche :

1. les crops d'ENTRAINEMENT sont issus de bboxes GT **bruitees**. Sans ce bruit, le
   modele apprend sur des cadrages parfaits qu'il ne reverra jamais a l'inference,
   ou les bboxes viennent d'un detecteur ;
2. la transformation crop -> image est conservee, car toute prediction doit etre
   retro-projetee dans le repere de l'image d'origine avant ecriture (contrat 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from insectpose.data.keypoints import KeypointSchema
from insectpose.data.yolo_export import flat_name, write_data_yaml
from insectpose.utils.geometry import apply_affine, crop_affine, invert_affine, jitter_bbox
from insectpose.utils.logging import get_logger

log = get_logger("crop_export")


def expand_bbox(bbox_xywh: np.ndarray, padding: float) -> np.ndarray:
    """Elargit une bbox d'un facteur relatif, en gardant son centre.

    La marge evite que les extremites (tarses, antennes) tombent hors du crop : un
    keypoint hors cadre est irrecuperable, quelle que soit la qualite du modele.
    """
    x, y, w, h = np.asarray(bbox_xywh, dtype=float)
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * (1 + padding), h * (1 + padding)
    return np.array([cx - nw / 2, cy - nh / 2, nw, nh])


def crop_image(image: Any, bbox_xywh: np.ndarray,
               out_size: tuple[int, int]) -> tuple[Any, np.ndarray]:
    """Recadre une image PIL selon une bbox et retourne (crop, matrice image->crop)."""
    from PIL import Image

    matrix = crop_affine(bbox_xywh, out_size)
    crop = image.transform(
        out_size, Image.AFFINE, data=tuple(invert_affine(matrix).reshape(-1)),
        resample=Image.BILINEAR,
    )
    return crop, matrix


def crop_label_line(kpts_xy: np.ndarray, kpts_vis: np.ndarray, matrix: np.ndarray,
                    out_size: tuple[int, int], margin: float = 0.02) -> str:
    """Ligne de label YOLO-pose d'une instance dans le repere du crop.

    La bbox du label est l'enveloppe des keypoints visibles DANS le crop, et non le
    crop entier : sinon le modele apprendrait que la boite couvre toujours l'image,
    ce qui rendrait sa sortie de detection inutilisable.
    """
    width, height = out_size
    points = apply_affine(matrix, np.asarray(kpts_xy, dtype=float).reshape(-1, 2))
    visible = np.asarray(kpts_vis, dtype=int) > 0

    inside = visible & (points[:, 0] >= 0) & (points[:, 0] < width) \
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    reference = points[inside] if inside.any() else points[visible]
    if reference.size == 0:
        reference = points
    x0, y0 = reference.min(axis=0)
    x1, y1 = reference.max(axis=0)
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    box = np.clip([cx / width, cy / height, bw / width, bh / height], 0.0, 1.0)
    norm = np.clip(points / np.array([width, height]), 0.0, 1.0)
    # Un point non annote ou tombe hors du crop est marque non supervise (vis = 0) :
    # il ne doit ni etre appris comme un zero, ni compte comme une erreur.
    flags = np.where(inside, np.asarray(kpts_vis, dtype=int), 0)
    norm[flags == 0] = 0.0

    values: list[Any] = [0, *box.tolist()]
    for (px, py), v in zip(norm, flags, strict=True):
        values.extend([px, py, int(v)])
    return " ".join(str(v) if isinstance(v, int) else f"{v:.6f}" for v in values)


def export_crops(image_set: Any, schema: KeypointSchema, root: Path, split: str,  # noqa: ARG001
                 out_size: tuple[int, int], padding: float = 0.15,
                 jitter_scale: float = 0.0, jitter_shift: float = 0.0,
                 seed: int = 0) -> dict[str, int]:
    """Ecrit un crop et son label par instance. Effet de bord : cree `root`.

    `jitter_scale`/`jitter_shift` doivent etre > 0 pour le split d'entrainement et
    nuls pour la validation : on valide sur des cadrages non bruites, sinon la metrique
    de validation devient elle-meme bruitee (§9.3).
    """
    from PIL import Image

    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    counts = {"instances": 0, "images": 0, "keypoints_outside": 0}

    for image_id, group in image_set.annotations.groupby("image_id"):
        meta = image_set.images.set_index("image_id").loc[image_id]
        source = image_set.absolute_path(meta.image_path)
        if not source.exists():
            raise FileNotFoundError(f"Image absente : {source}")
        with Image.open(source) as handle:
            image = handle.convert("RGB")
            counts["images"] += 1
            for n, row in enumerate(group.itertuples(index=False)):
                bbox = expand_bbox(np.asarray(row.bbox_xywh, dtype=float), padding)
                if jitter_scale > 0 or jitter_shift > 0:
                    bbox = jitter_bbox(bbox, rng, jitter_scale, jitter_shift)
                crop, matrix = crop_image(image, bbox, out_size)
                stem = f"{flat_name(image_id)}__{n}"
                crop.save(img_dir / f"{stem}.jpg", quality=95)

                line = crop_label_line(row.kpts_xy, row.kpts_vis, matrix, out_size)
                (lbl_dir / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
                counts["instances"] += 1
                counts["keypoints_outside"] += int(
                    (np.asarray(row.kpts_vis, dtype=int) > 0).sum()
                    - sum(1 for v in line.split()[5::3] if float(v) > 0)
                )

    log.info("Export crops [%s] : %d instance(s) depuis %d image(s) -> %s",
             split, counts["instances"], counts["images"], img_dir)
    return counts


def export_crop_fold(data: Any, schema: KeypointSchema, root: Path, out_size: tuple[int, int],
                     padding: float, jitter_scale: float, jitter_shift: float,
                     seed: int = 0) -> Path:
    """Exporte train (avec bruit de cadrage) et val (sans) puis ecrit data.yaml."""
    root.mkdir(parents=True, exist_ok=True)
    export_crops(data.train, schema, root, "train", out_size, padding,
                 jitter_scale, jitter_shift, seed)
    export_crops(data.val, schema, root, "val", out_size, padding, 0.0, 0.0, seed + 1)
    return write_data_yaml(root, schema, {"train": "train", "val": "val"})


def crops_from_boxes(image: Any, boxes: np.ndarray, out_size: tuple[int, int],
                     padding: float) -> tuple[list[Any], list[np.ndarray]]:
    """Crops et matrices image->crop pour des bboxes PREDITES (inference).

    Aucun bruit de cadrage ici : a l'inference, le cadrage est celui que produit le
    detecteur, et c'est precisement ce que l'on veut mesurer.
    """
    crops, matrices = [], []
    for bbox in np.atleast_2d(np.asarray(boxes, dtype=float)):
        crop, matrix = crop_image(image, expand_bbox(bbox, padding), out_size)
        crops.append(np.asarray(crop))
        matrices.append(matrix)
    return crops, matrices