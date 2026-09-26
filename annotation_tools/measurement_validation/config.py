"""Measurements to classify: reading of the `measurements` block of kp_infos.yaml.

A measurement is a chain of keypoints. It is split into edges (segments between two
consecutive keypoints): the edge is the unit handled by the application, because
several measurements can share the same segment (e.g. "femur" and "leg length"),
which makes the greying cascade.
"""
from collections import OrderedDict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KP_INFOS = REPO_ROOT / "kp_infos.yaml"

_CASCADE_KEYWORDS = ("leg", "antenna")


def _is_cascade_measurement(measurement_name):
    """Only the leg and antenna measurements propagate their status to the
    neighbouring measurements that share an edge (see AppConfig.cascade_edges)."""
    name = measurement_name.lower()
    return any(kw in name for kw in _CASCADE_KEYWORDS)


class AppConfig:
    def __init__(self, measurements):
        self.measurements = measurements  # OrderedDict name -> [kp names]

        # Unique edges (segments between consecutive kp), shared between measurements.
        # sorted edge_key (kpA, kpB) -> (kpA, kpB) in order of first appearance.
        self.edges = OrderedDict()
        # measurement name -> ordered list of the (sorted) edge_keys it crosses
        self.measurement_edges = OrderedDict()
        for m_name, kp_list in measurements.items():
            keys = []
            for a, b in zip(kp_list, kp_list[1:]):
                key = tuple(sorted((a, b)))
                if key not in self.edges:
                    self.edges[key] = (a, b)
                keys.append(key)
            self.measurement_edges[m_name] = keys

        # edge_key -> list of the measurements crossing it (reverse mapping), restricted
        # to the measurement families where the cascade makes sense (legs, antennae): a
        # composite measurement such as "total length" also crosses head/thorax/abdomen
        # and must not link them together for all that.
        self.edge_measurements = OrderedDict()
        for m_name, keys in self.measurement_edges.items():
            if not _is_cascade_measurement(m_name):
                continue
            for key in keys:
                self.edge_measurements.setdefault(key, []).append(m_name)

    def measurement_names(self):
        return list(self.measurements.keys())

    def cascade_edges(self, edge_key):
        """Transitive closure of the set of edges that must share the same
        measurable/non-measurable status as `edge_key`: every edge of the measurements
        it belongs to (e.g. greying the femur greys the whole "leg length"
        measurement, hence also the tibia and the tarsus that make it up), and so on
        as long as new measurements/edges are discovered."""
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
    """Read the `measurements` block of kp_infos.yaml. Returns an AppConfig."""
    path = Path(path) if path else DEFAULT_KP_INFOS
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    measurements = OrderedDict(
        (name, list(kps)) for name, kps in (raw.get("measurements") or {}).items())

    if not measurements:
        raise ValueError(f"no measurement found in {path}")
    return AppConfig(measurements)


if __name__ == "__main__":
    import sys
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{len(cfg.measurements)} measurements, {len(cfg.edges)} edges")
