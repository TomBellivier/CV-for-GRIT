"""Cost metrics: latency, model size, training duration.

FIRST-ORDER metrics to choose a deployable model, not appendices (§7.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from insectpose.evaluation.bundle import EvalBundle, record
from insectpose.registry import register_metric


@register_metric("cost")
def cost(bundle: EvalBundle) -> list[dict[str, Any]]:
    """Inference latency per instance, when the approach reported it."""
    if "inference_ms" not in bundle.pred.columns or bundle.pred.empty:
        return []
    values = bundle.pred["inference_ms"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    return [
        record("overall", "latency_ms_per_instance", float(values.mean()), int(values.size)),
        record("overall", "latency_ms_p95", float(np.percentile(values, 95)), int(values.size)),
    ]
