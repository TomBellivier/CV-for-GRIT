"""
worker.py
=========

Per-thread machinery for the parallel run.

Thread-local models
-------------------
A single Ultralytics model is NOT safe to call from several threads at once
(each predict() call mutates internal state). The clean fix is to give every
worker thread its OWN model instances, created lazily the first time that
thread needs them. Threads still share everything read-only (config, the image
loader, dataset membership), so the only duplicated objects are the models.

    Memory cost: one (pose + scale-bar + EasyOCR) set per worker thread. With
    --workers 4 you hold 4 copies. Lower --workers if RAM is tight; raise it to
    overlap more inferences / downloads.

CPU sharing
-----------
PyTorch uses intra-op threads for a single inference and by default grabs all
cores. Running W worker threads that each grab all cores would oversubscribe
the 16 CPUs. configure_cpu_threads() therefore splits the cores across workers
(≈ cpus // workers) so the total stays around the physical core count.
"""

from __future__ import annotations

import os
import threading

from . import config
from .pipeline import Models, process_image

# Each thread gets its own attribute bag; models live here.
_local = threading.local()


def configure_cpu_threads(workers: int, cpus: int | None = None,
                          torch_threads: int | None = None) -> None:
    """Balance PyTorch/OpenCV threads across the worker threads.

    Called ONCE from the main thread before the pool starts.
    """
    cpus = cpus or os.cpu_count() or 1
    per_worker = torch_threads or max(1, cpus // max(1, workers))
    try:
        import torch
        torch.set_num_threads(per_worker)
    except Exception:  # noqa: BLE001 - torch always present with ultralytics, but be safe
        pass
    try:
        import cv2
        cv2.setNumThreads(per_worker)
    except Exception:  # noqa: BLE001
        pass
    print(f"[cpu] {cpus} CPUs, {workers} worker(s) -> {per_worker} compute thread(s) each")


def _get_models() -> Models:
    """Return this thread's models, loading them on first use."""
    if getattr(_local, "models", None) is None:
        # Imported here so a thread only loads models when it actually runs.
        from .pose_inference import load_pose_model
        from .scale import load_scale_bar_model

        tid = threading.get_ident()
        print(f"[worker {tid}] loading models for this thread...")
        pose = load_pose_model()
        scale_bar = load_scale_bar_model() if config.USE_SCALE_BAR else None
        _local.models = Models(pose_model=pose, scale_bar_model=scale_bar)
    return _local.models


def make_task(load_fn, membership):
    """Build the function run for each item: load the image, then process it.

    Returned callable maps  (key, image_name)  ->  record dict.
    Exceptions propagate to parallel.bounded_unordered_map, which reports them
    per-image instead of aborting the run.
    """
    def task(item):
        key, image_name = item
        img_bgr = load_fn(key)               # download/decode (HF) or read (local)
        models = _get_models()
        return process_image(img_bgr, image_name, models, membership)
    return task
