"""Export du format canonique vers le format YOLO-pose (CONVENTIONS.md §9.1).

Operation DERIVEE : les fichiers produits sont regeneres a chaque fold a partir des
splits partages, et ecrits dans le repertoire du run (ou dans `data/interim/`), jamais
dans `data/processed/`. Le format canonique reste la seule source de verite.

Format d'un label YOLO-pose, une ligne par instance, tout normalise dans [0, 1] :
    classe cx cy w h  x1 y1 v1  x2 y2 v2  ...  xK yK vK
La bbox est CENTREE (cx, cy), contrairement au contrat 1 qui est en coin haut-gauche.
C'est exactement le genre de divergence que ce module isole en un seul endroit.
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

CLASS_NAMES = {0: "insect"}   # une seule classe : "insecte" (§9.1)


def flat_name(image_id: str) -> str:
    """Nom de fichier plat et unique a partir d'un image_id `<dataset>/<stem>`.

    Sans cet aplatissement, deux datasets ayant un `img001.png` se recouvriraient
    silencieusement dans le repertoire YOLO.
    """
    return str(image_id).replace("/", "__")


def to_label_lines(instances: pd.DataFrame, width: int, height: int,
                   n_keypoints: int, with_keypoints: bool = True) -> tuple[list[str], int]:
    """Lignes de label YOLO d'une image. Retourne (lignes, nb de valeurs rognees).

    Les coordonnees hors image sont rognees dans [0, 1] (contrainte du format) et le
    compte est remonte : un rognage massif signale des annotations douteuses.
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
                f"{row.instance_id} : {len(kpts)} keypoints pour un schema a {n_keypoints}."
            )
        norm = kpts / np.array([width, height])
        clipped += int((norm[vis > 0] < 0).sum() + (norm[vis > 0] > 1).sum())
        norm = np.clip(norm, 0.0, 1.0)
        # Un point non annote est ecrit (0, 0, 0) : c'est la convention YOLO pour
        # "non supervise". Il est masque dans la loss, jamais appris comme un zero.
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
    """Relit une ligne de label YOLO vers le repere de l'image d'origine.

    Reciproque exacte de `to_label_lines` : c'est ce qui rend l'export testable
    par aller-retour, sans jamais lancer d'entrainement.
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
    """Ecrit images/<split>/ et labels/<split>/ pour un ImageSet.

    Les images sont liees telles quelles : le format canonique reste la seule source
    de verite et les coordonnees ne subissent aucune transformation d'echelle.

    Effet de bord : cree images/ et labels/ sous `root`. Retourne le nombre d'images.
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
                    f"Image absente : {source}. L'export YOLO exige les images reelles."
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
        log.warning("[%s] %d coordonnee(s) rognee(s) dans [0,1] a l'export YOLO : "
                    "verifier les annotations hors image.", split, total_clipped)
    return len(images)


def write_data_yaml(root: Path, schema: KeypointSchema, splits: dict[str, str],
                    with_keypoints: bool = True) -> Path:
    """Ecrit le data.yaml d'Ultralytics. Effet de bord : cree <root>/data.yaml.

    `flip_idx` est OBLIGATOIRE des lors qu'une augmentation par miroir est active :
    sans lui, un miroir echange les cotes gauche/droite sans permuter les labels et
    l'entrainement apprend une anatomie fausse (§3.1).
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
    """Exporte un FoldData au format YOLO et retourne le chemin du data.yaml.

    Effet de bord : cree l'arborescence YOLO sous `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        n = export_split(data.role(split), schema, root, split, link_images, with_keypoints)
        log.info("Export YOLO [%s] : %d images -> %s", split, n, root / "images" / split)
    return write_data_yaml(root, schema, {s: s for s in splits}, with_keypoints)