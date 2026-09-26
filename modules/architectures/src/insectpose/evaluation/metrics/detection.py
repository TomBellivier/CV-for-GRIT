"""Detection metrics (AP on bbox IoU).

Only apply to the predictions with `bbox_source == 'predicted'`: evaluating the
detection on GT bboxes makes no sense and would bias the comparison (§9.3).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from insectpose.evaluation.bundle import EvalBundle, record
from insectpose.evaluation.matching import assign_greedy, average_precision
from insectpose.registry import register_metric

_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def _ap_at(pairs: list, threshold: float, similarity: str) -> tuple[float, int]:
    """AP at a given similarity threshold, over a scope of images."""
    scores: list[np.ndarray] = []
    tps: list[np.ndarray] = []
    n_gt = 0
    for p in pairs:
        n_gt += p.n_gt
        sim = getattr(p, similarity)
        matched, _ = assign_greedy(sim, p.scores, threshold)
        scores.append(p.scores)
        tps.append(matched >= 0)
    if n_gt == 0:
        return float("nan"), 0
    return (
        average_precision(np.concatenate(scores) if scores else np.zeros(0),
                          np.concatenate(tps) if tps else np.zeros(0, dtype=bool), n_gt),
        n_gt,
    )


@register_metric("detection_ap")
def detection_ap(bundle: EvalBundle) -> list[dict[str, Any]]:
    """AP@0.5 and AP@[.5:.95] on bbox IoU."""
    if "predicted" not in bundle.bbox_sources():
        return []  # approach without detection: metric not applicable, never 0 by default
    out: list[dict[str, Any]] = []
    for scope, pairs in bundle.scopes():
        ap50, n = _ap_at(pairs, 0.5, "iou")
        out.append(record(scope, "det_ap@0.5", ap50, n))
        values = [_ap_at(pairs, float(t), "iou")[0] for t in _THRESHOLDS]
        out.append(record(scope, "det_ap@[.5:.95]", float(np.nanmean(values)), n))
    return out
