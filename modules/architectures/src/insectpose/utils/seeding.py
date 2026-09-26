"""Derivation and application of the seeds (§6.4).

A single seed in the config; all the others are derived from it in a stable way, so
that a fold or a dataloader never shares the same random stream by accident.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_for(run_id: str, fold: int, purpose: str, base: int = 0) -> int:
    """Deterministic seed for a (run, fold, purpose). Always in [0, 2**31)."""
    key = f"{base}|{run_id}|{fold}|{purpose}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") % (2**31)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set python / numpy / torch (if present).

    `deterministic=True` (debug mode) enables the deterministic algorithms of torch,
    which are slower. The choice is recorded in the manifest.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, seed: int = 0) -> None:
    """Init of the dataloader workers: each worker has its own stream."""
    np.random.seed((seed + worker_id) % (2**31))
    random.seed(seed + worker_id)
