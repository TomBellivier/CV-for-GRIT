"""Mesures à classer : lecture du bloc `measurements` de kp_infos.yaml.

Une mesure est une chaîne de keypoints. Elle est découpée en arêtes (segments
entre deux keypoints consécutifs) : l'arête est l'unité manipulée par
l'application, car plusieurs mesures peuvent partager le même segment
(ex. "femur" et "longueur de patte"), ce qui fait cascader le grisage.
"""
from collections import OrderedDict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KP_INFOS = REPO_ROOT / "kp_infos.yaml"

_CASCADE_KEYWORDS = ("leg", "antenna")


def _is_cascade_measurement(measurement_name):
    """Seules les mesures de pattes et d'antennes propagent leur statut aux mesures
    voisines qui partagent une arête (voir AppConfig.cascade_edges)."""
    name = measurement_name.lower()
    return any(kw in name for kw in _CASCADE_KEYWORDS)


class AppConfig:
    def __init__(self, measurements):
        self.measurements = measurements  # OrderedDict nom -> [kp names]

        # Arêtes uniques (segments entre kp consécutifs), partagées entre mesures.
        # edge_key trié (kpA, kpB) -> (kpA, kpB) dans l'ordre de première apparition.
        self.edges = OrderedDict()
        # nom mesure -> liste ordonnée d'edge_key (triés) qu'elle traverse
        self.measurement_edges = OrderedDict()
        for m_name, kp_list in measurements.items():
            keys = []
            for a, b in zip(kp_list, kp_list[1:]):
                key = tuple(sorted((a, b)))
                if key not in self.edges:
                    self.edges[key] = (a, b)
                keys.append(key)
            self.measurement_edges[m_name] = keys

        # edge_key -> liste des mesures qui le traversent (mapping inverse), restreint
        # aux familles de mesures où la cascade a du sens (pattes, antennes) : une
        # mesure composite comme "total length" traverse aussi tête/thorax/abdomen et
        # ne doit pas les lier entre elles pour autant.
        self.edge_measurements = OrderedDict()
        for m_name, keys in self.measurement_edges.items():
            if not _is_cascade_measurement(m_name):
                continue
            for key in keys:
                self.edge_measurements.setdefault(key, []).append(m_name)

    def measurement_names(self):
        return list(self.measurements.keys())

    def cascade_edges(self, edge_key):
        """Ferme par transitivité l'ensemble des arêtes qui doivent partager le même
        statut mesurable/non-mesurable que `edge_key` : toutes les arêtes des mesures
        auxquelles il appartient (ex. griser le fémur grise toute la mesure "longueur
        de patte", donc aussi le tibia et le tarse qui la composent), et ainsi de
        suite tant que de nouvelles mesures/arêtes sont découvertes."""
        edges = {edge_key}
        changed = True
        while changed:
            changed = False
            measurements = set()
            for e in edges:
                measurements.update(self.edge_measurements.get(e, []))
            for m in measurements:
                for e in self.measurement_edges[m]:
                    if e not in edges:
                        edges.add(e)
                        changed = True
        return edges


def load_config(path=None):
    """Lit le bloc `measurements` de kp_infos.yaml. Retourne un AppConfig."""
    path = Path(path) if path else DEFAULT_KP_INFOS
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    measurements = OrderedDict(
        (name, list(kps)) for name, kps in (raw.get("measurements") or {}).items())

    if not measurements:
        raise ValueError(f"aucune mesure trouvée dans {path}")
    return AppConfig(measurements)


if __name__ == "__main__":
    import sys
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{len(cfg.measurements)} mesures, {len(cfg.edges)} arêtes")
