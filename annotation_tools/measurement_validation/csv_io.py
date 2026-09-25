"""Entrées/sorties CSV.

Entrée : le CSV produit par annotation_tools/labelstudio_to_csv.py (une ligne par
annotation, keypoints en pixels absolus dans les colonnes "<kp>_x/_y/_v").

Sortie : un CSV de statuts, une ligne par annotation, une colonne
"<mesure>_status" par mesure — c'est le seul fichier exploité ensuite
(modules/meas_classifier/dataset.py). Aucune position de keypoint n'y figure.

L'état de classement détaillé (statut par arête, images validées) est sauvegardé
à part, en JSON, uniquement pour pouvoir reprendre le travail là où il s'est
arrêté.
"""
import csv
import json
from pathlib import Path

from app_logging import logger
from models import ImageAnnotation, MEASURABLE, NON_MEASURABLE, edge_key

# Le CSV de labelstudio_to_csv.py écrit v=0 pour un keypoint absent.
_MISSING = "0"


def _keypoint_names(fieldnames):
    """Déduit la liste des keypoints des colonnes "<kp>_v" du CSV d'entrée."""
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
    """Charge le CSV d'annotations. Retourne une liste d'ImageAnnotation."""
    csv_path = Path(csv_path)
    logger.info("load_annotations : lecture de %s", csv_path)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        kp_names = _keypoint_names(reader.fieldnames)
        if not kp_names:
            raise ValueError(
                f"{csv_path} ne contient aucune colonne de keypoint (<kp>_x/_y/_v) : "
                "attendu un CSV produit par labelstudio_to_csv.py"
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
    logger.info("load_annotations : %d annotation(s) chargée(s), %d keypoint(s) possibles",
                len(annotations), len(kp_names))
    return annotations


def export_status_csv(annotations, config, out_path):
    """Écrit le fichier final : une ligne par annotation, le statut de chaque mesure."""
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
    logger.info("export_status_csv : %d ligne(s) écrite(s) dans %s", len(annotations), out_path)


def save_state(annotations, out_path):
    """Sauvegarde le statut par arête et les images validées, pour pouvoir reprendre
    le classement exactement là où il s'est arrêté (le CSV exporté, lui, ne retient
    que le statut par mesure : il ne permettrait pas de le restaurer fidèlement)."""
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
    """Restaure sur `annotations` (en place) l'état sauvegardé par save_state."""
    state_path = Path(state_path)
    if not state_path.exists():
        return
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Impossible de lire l'état de classement : %s", state_path)
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
    logger.info("load_state : état restauré pour %d/%d annotation(s)", restored, len(annotations))
