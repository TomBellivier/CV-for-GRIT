"""
hardware.py
===========

Adapts the run to the machine it runs on: CUDA GPU(s), Apple GPU, or CPU only,
with however many CPUs and however much memory it has. Everything is resolved
ONCE, in the main thread, before the worker pool starts (plan_resources), and
printed at start-up. Every value can be forced in config.py or on the command
line; "auto" is the default everywhere.

    devices   The worker threads run their models on the GPUs in turn (worker k
              on GPU k % n_gpus), or on the Apple GPU, or on the CPU.
    workers   With a GPU: config.WORKERS_PER_GPU threads per GPU, so that image
              decoding, OCR, the ruler FFT and the classifiers (CPU work) overlap
              with the GPU inference -- never more threads than CPUs.
              CPU only: one worker per config.CPU_THREADS_PER_WORKER cores, twice
              as many for a Hugging Face source (they mostly wait on downloads).
              In both cases capped by the free RAM (and free VRAM): every worker
              holds its own copy of the models (see worker.py).
    threads   PyTorch / OpenCV compute threads per worker = CPUs // workers (>= 1),
              so the workers share the cores instead of each grabbing all of them.
    buffer    Images in flight = 2 x workers: enough to keep every worker fed.
"""

from __future__ import annotations

import itertools
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from . import config

# Weights on disk -> memory of a loaded model: FP32 copy of (often FP16-saved)
# weights, fused layers and the runtime buffers of the first inference.
MODEL_MEMORY_FACTOR = 4.0


@dataclass
class ResourcePlan:
    """How the run uses the machine. Built once by plan_resources()."""
    devices: list[str]           # devices the workers take in turn
    workers: int                 # parallel worker threads
    threads_per_worker: int      # PyTorch / OpenCV compute threads per worker
    buffer: int                  # max images in flight
    cpus: int                    # CPUs this process may use
    worker_memory_mb: float      # estimated memory held by one worker
    reason: str                  # what decided the number of workers

    def describe(self) -> str:
        return (f"{self.workers} worker(s) on {', '.join(self.devices)} | "
                f"{self.threads_per_worker} compute thread(s) each | buffer {self.buffer} | "
                f"{self.cpus} CPU(s), ~{self.worker_memory_mb:.0f} MB per worker ({self.reason})")


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def usable_cpus() -> int:
    """CPUs this process may actually use (affinity / container quota included)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:          # Windows, macOS: no affinity exposed
        return max(1, os.cpu_count() or 1)


def available_devices(spec: str = "auto") -> list[str]:
    """Devices to run the models on: every CUDA GPU, else the Apple GPU, else the CPU.

    An explicit spec ("cpu", "cuda:1", "mps"...) is kept as is.
    """
    if str(spec) != "auto":
        return [str(spec)]
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return ["mps"]
    return ["cpu"]


def worker_memory_mb(model_paths) -> float:
    """Estimated memory one worker holds: its copy of every model + its buffers."""
    weights_mb = sum(Path(p).stat().st_size for p in model_paths if Path(p).is_file()) / 2**20
    return MODEL_MEMORY_FACTOR * weights_mb + config.WORKER_MEMORY_OVERHEAD_MB


def _free_ram_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 2**20


def _free_vram_mb(device: str) -> float | None:
    torch = _torch()
    if torch is None or not device.startswith("cuda"):
        return None
    index = int(device.split(":")[1]) if ":" in device else 0
    return torch.cuda.mem_get_info(index)[0] / 2**20


def plan_resources(model_paths, workers=None, threads=None, buffer=None,
                   source: str = "folder") -> ResourcePlan:
    """Decide devices, workers, threads and buffer for this machine.

    `workers` / `threads` / `buffer` override the automatic choice (command line);
    None falls back to config.WORKERS ("auto" or a number).
    """
    cpus = usable_cpus()
    devices = available_devices(config.DEVICE)
    on_gpu = devices[0] != "cpu"
    per_worker = worker_memory_mb(model_paths)
    requested = workers if workers is not None else config.WORKERS

    if str(requested) != "auto":
        n, reason = max(1, int(requested)), "set by hand"
    else:
        if on_gpu:
            n = min(cpus, config.WORKERS_PER_GPU * len(devices))
            reason = f"{config.WORKERS_PER_GPU} per GPU x {len(devices)} GPU(s), at most 1 per CPU"
        else:
            n = max(1, cpus // config.CPU_THREADS_PER_WORKER)
            reason = f"{cpus} CPU(s) / {config.CPU_THREADS_PER_WORKER} threads per worker"
            if source == "hf":
                n *= 2
                reason += ", x2 to overlap the downloads"

        budget = config.MEMORY_BUDGET_FRACTION
        free_ram = _free_ram_mb()
        if free_ram is not None:
            cap = max(1, int(budget * free_ram // per_worker))
            if cap < n:
                n, reason = cap, f"capped by the free RAM ({free_ram:.0f} MB)"
        vram = [v for v in (_free_vram_mb(d) for d in devices) if v is not None]
        if vram:
            cap = max(1, int(budget * min(vram) // per_worker)) * len(devices)
            if cap < n:
                n, reason = cap, f"capped by the free VRAM ({min(vram):.0f} MB per GPU)"

    threads_per_worker = int(threads) if threads else max(1, cpus // n)
    return ResourcePlan(
        devices=devices, workers=n, threads_per_worker=threads_per_worker,
        buffer=int(buffer) if buffer else 2 * n, cpus=cpus,
        worker_memory_mb=per_worker, reason=reason,
    )


def apply_threads(plan: ResourcePlan) -> None:
    """Give PyTorch and OpenCV `threads_per_worker` compute threads.

    Called ONCE from the main thread before the pool starts: without it, every
    worker's inference would grab all the cores and they would fight for them.
    """
    try:
        import torch
        torch.set_num_threads(plan.threads_per_worker)
    except Exception:  # noqa: BLE001 - torch always present with ultralytics, but be safe
        pass
    try:
        import cv2
        cv2.setNumThreads(plan.threads_per_worker)
    except Exception:  # noqa: BLE001
        pass


_PLAN: ResourcePlan | None = None
_NEXT_WORKER = itertools.count()
_LOCK = threading.Lock()


def set_plan(plan: ResourcePlan) -> None:
    """Make the plan visible to the worker threads (see next_device)."""
    global _PLAN
    _PLAN = plan


def next_device() -> str:
    """Device of the next worker thread: the plan's devices, in turn."""
    devices = _PLAN.devices if _PLAN is not None else available_devices(config.DEVICE)
    with _LOCK:
        k = next(_NEXT_WORKER)
    return devices[k % len(devices)]


def precision_kwargs(device: str) -> dict:
    """predict() argument for FP16 inference on a CUDA GPU; {} elsewhere (FP32).

    `half` is deprecated since Ultralytics 8.4 in favour of `quantize=16`: the
    installed default config is queried rather than the version number.
    """
    if not (config.HALF_PRECISION_ON_GPU and str(device).startswith("cuda")):
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError:
        return {"half": True}
    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
