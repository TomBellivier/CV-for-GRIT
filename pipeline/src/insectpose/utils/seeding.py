"""Derivation et application des seeds (§6.4).

Une seule seed en config ; toutes les autres en sont derivees de facon stable, pour
qu'un fold ou un dataloader ne partage jamais le meme flux aleatoire par accident.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_for(run_id: str, fold: int, purpose: str, base: int = 0) -> int:
    """Seed deterministe pour un (run, fold, usage). Toujours dans [0, 2**31)."""
    key = f"{base}|{run_id}|{fold}|{purpose}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") % (2**31)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Fixe python / numpy / torch (si present).

    `deterministic=True` (mode debug) active les algorithmes deterministes de torch,
    plus lents. Le choix est enregistre dans le manifeste.
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
    """Init des workers de dataloader : chaque worker a son propre flux."""
    np.random.seed((seed + worker_id) % (2**31))
    random.seed(seed + worker_id)
