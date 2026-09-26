"""Compute device: resolution, description, VRAM measurement (ADR-0019).

The hardware is part of the conditions of a comparison: two approaches trained on
different GPUs, or one with AMP and the other without, are not exactly comparable.
Everything is therefore resolved here, logged, and recorded in the manifest.
"""

from __future__ import annotations

import os
from typing import Any

from insectpose.utils.logging import get_logger

log = get_logger("device")

# Cap of the data-loading workers: beyond it, the GPU (or the CPU) no longer keeps up
# and each worker only adds a copy of the dataset in memory.
MAX_DATALOADER_WORKERS = 8


def usable_cpus() -> int:
    """CPUs this process can actually use (affinity / quota included)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:          # Windows, macOS: no affinity exposed
        return max(1, os.cpu_count() or 1)


def resolve_num_workers(spec: str | int | None = "auto") -> int:
    """Translate `train.num_workers` into a number of data-loading workers.

    'auto' -> one worker per CPU, minus one left to the training loop, capped at
    MAX_DATALOADER_WORKERS. On a single-CPU machine, 0: loading happens in the main
    process, without any extra process. An explicit value is respected as it is.
    """
    value = "auto" if spec is None else str(spec)
    if value != "auto":
        return int(value)
    return min(MAX_DATALOADER_WORKERS, usable_cpus() - 1)


def _torch() -> Any | None:
    """Torch if it can be imported, else None (CPU execution without torch is possible)."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def cuda_available() -> bool:
    """True if at least one CUDA GPU can be used."""
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def mps_available() -> bool:
    """True on a Mac whose Apple GPU (Metal) can be used by torch."""
    torch = _torch()
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(spec: str | int | None = "auto") -> str:
    """Translate `train.device` into a string understood by Ultralytics and torch.

    'auto' -> '0' if CUDA is available, else 'mps' (Apple GPU), else 'cpu'. An explicit
    value is respected as it is: asking for 'cpu' on a GPU machine is a legitimate choice
    (debugging), not an error to correct silently.
    """
    value = "auto" if spec is None else str(spec)
    if value != "auto":
        return value
    if cuda_available():
        return "0"
    return "mps" if mps_available() else "cpu"


def device_indices(device: str) -> list[int]:
    """GPU indices of a specification ('0', '0,1'). Empty list for 'cpu' or 'mps'."""
    if device in ("cpu", "mps"):
        return []
    return [int(part) for part in device.split(",") if part.strip().isdigit()]


def device_info(device: str | None = None) -> dict[str, Any]:
    """Description of the hardware, meant for the manifest (§3.5)."""
    torch = _torch()
    resolved = resolve_device(device or "auto")
    info: dict[str, Any] = {
        "requested": str(device or "auto"),
        "resolved": resolved,
        "cuda_available": cuda_available(),
        "cpus": usable_cpus(),
    }
    if torch is None:
        return info
    info["torch"] = torch.__version__
    if not info["cuda_available"]:
        return info
    info["cuda"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["device_count"] = torch.cuda.device_count()
    info["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(i)),
            "total_vram_mb": round(torch.cuda.get_device_properties(i).total_memory / 2**20, 1),
        }
        for i in device_indices(resolved) or range(torch.cuda.device_count())
    ]
    return info


def reset_peak_vram() -> None:
    """Reset the peak VRAM counter, before a training."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float | None:
    """Peak VRAM allocated since the last reset, or None outside CUDA.

    First-order cost metric (§7.2): an approach that does not fit in memory cannot be
    deployed, whatever its OKS.
    """
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def amp_enabled(cfg_amp: bool, mode: str, device: str) -> bool:
    """Decide whether mixed precision is used.

    AMP is disabled outside CUDA (no effect) and in `mode: debug`, where bit-for-bit
    reproducibility prevails over speed (§6.4).
    """
    if not bool(cfg_amp):
        return False
    if device == "cpu":
        return False
    if mode == "debug":
        log.info("mode=debug: AMP disabled in favour of reproducibility.")
        return False
    return True
