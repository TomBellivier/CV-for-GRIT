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
loader, the classifiers), so the only duplicated objects are the models.

    Memory cost: one (pose ensemble + scale-bar) set per worker thread, and the
    ensemble holds every model of retained_models/pose/ (5 after a `tune`); the
    EasyOCR reader is shared. hardware.plan_resources() sizes the number of
    workers so that these copies fit in the free RAM (and VRAM).

Devices and CPU sharing
-----------------------
Each worker thread runs its models on one device, handed out in turn by
hardware.next_device() (the GPUs one after the other, or the CPU). PyTorch and
OpenCV compute threads are split across workers (hardware.apply_threads), so
the workers share the cores instead of each grabbing all of them.
"""

from __future__ import annotations

import threading

from . import config
from .hardware import next_device
from .pipeline import Models, process_image

# Each thread gets its own attribute bag; models live here.
_local = threading.local()


def _get_models() -> Models:
    """Return this thread's models, loading them on first use."""
    if getattr(_local, "models", None) is None:
        # Imported here so a thread only loads models when it actually runs.
        from .pose_inference import load_pose_models
        from .scale import load_scale_bar_model

        tid = threading.get_ident()
        device = next_device()
        print(f"[worker {tid}] loading models for this thread on {device}...")
        pose = load_pose_models()
        scale_bar = load_scale_bar_model() if config.USE_SCALE_BAR else None
        _local.models = Models(pose_models=pose, scale_bar_model=scale_bar, device=device)
    return _local.models


def make_task(load_fn, measurement_classifiers=None, group_index=None):
    """Build the function run for each item: load the image, then process it.

    `measurement_classifiers` and `group_index` are shared read-only across
    every worker thread: unlike the YOLO models, scoring a fitted random forest
    does not mutate it, so one set of models is enough.

    Returned callable maps  (key, image_name)  ->  record dict.
    Exceptions propagate to parallel.bounded_unordered_map, which reports them
    per-image instead of aborting the run.
    """
    def task(item):
        key, image_name = item
        img_bgr = load_fn(key)               # download/decode (HF) or read (local)
        models = _get_models()
        return process_image(img_bgr, image_name, models,
                             measurement_classifiers, group_index)
    return task
