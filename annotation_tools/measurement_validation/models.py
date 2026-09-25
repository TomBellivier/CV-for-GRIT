"""Modèles de données internes : une annotation d'image = des keypoints (px absolus,
lus tels quels, jamais modifiés) + le statut mesurable/non-mesurable de ses mesures.

Le statut est stocké par arête (segment entre deux kp consécutifs) et non par
mesure : plusieurs mesures peuvent partager la même arête (ex. "fémur" et
"longueur de patte"), ce qui fait cascader le grisage d'une arête vers toutes
les mesures qui en dépendent.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

MEASURABLE = "measurable"
NON_MEASURABLE = "non_measurable"


def edge_key(a, b):
    return tuple(sorted((a, b)))


@dataclass
class ImageAnnotation:
    image_name: str
    image_path: str  # chemin vers l'image, tel qu'écrit dans le CSV d'entrée
    width: int
    height: int
    annotation_id: str = ""
    # rotation d'affichage en degrés (0/90/180/270) : n'affecte que la vue, jamais
    # les coordonnées des keypoints, qui restent dans le repère de l'image d'origine.
    image_rotation: int = 0
    keypoints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # (kpA, kpB) triés -> MEASURABLE | NON_MEASURABLE (override manuel d'une arête)
    edge_overrides: Dict[Tuple[str, str], str] = field(default_factory=dict)
    done: bool = False  # classement validé par l'utilisateur

    def key(self):
        """Identifiant stable, utilisé pour sauvegarder/restaurer l'état de classement
        (une même image peut apparaître deux fois si elle a été annotée deux fois)."""
        return f"{self.image_name}::{self.annotation_id}"

    def has_kp(self, name):
        return name in self.keypoints

    def edge_status(self, key):
        a, b = key
        if not (self.has_kp(a) and self.has_kp(b)):
            return NON_MEASURABLE
        return self.edge_overrides.get(key, MEASURABLE)

    def measurement_status(self, edge_keys):
        """Retourne MEASURABLE seulement si toutes les arêtes de la mesure le sont."""
        if not edge_keys:
            return NON_MEASURABLE
        return MEASURABLE if all(self.edge_status(k) == MEASURABLE for k in edge_keys) else NON_MEASURABLE


@dataclass
class Preset:
    name: str
    # "kpA::kpB" (triés) -> statut forcé
    overrides: Dict[str, str] = field(default_factory=dict)
