"""Peripherique de calcul : resolution, description, mesure de VRAM (ADR-0019).

Le materiel fait partie des conditions d'une comparaison : deux approches entrainees
sur des GPU differents, ou l'une en AMP et l'autre non, ne sont pas exactement
comparables. Tout est donc resolu ici, journalise, et enregistre dans le manifeste.
"""

from __future__ import annotations

from typing import Any

from insectpose.utils.logging import get_logger

log = get_logger("device")


def _torch() -> Any | None:
    """Torch s'il est importable, sinon None (execution CPU sans torch possible)."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def cuda_available() -> bool:
    """True si au moins un GPU CUDA est utilisable."""
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def resolve_device(spec: str | int | None = "auto") -> str:
    """Traduit `train.device` en chaine comprise par Ultralytics et torch.

    'auto' -> '0' si CUDA est disponible, sinon 'cpu'. Une valeur explicite est
    respectee telle quelle : demander 'cpu' sur une machine a GPU est un choix
    legitime (debogage), pas une erreur a corriger en silence.
    """
    value = "auto" if spec is None else str(spec)
    if value != "auto":
        return value
    return "0" if cuda_available() else "cpu"


def device_indices(device: str) -> list[int]:
    """Indices GPU d'une specification ('0', '0,1'). Liste vide pour 'cpu' ou 'mps'."""
    if device in ("cpu", "mps"):
        return []
    return [int(part) for part in device.split(",") if part.strip().isdigit()]


def device_info(device: str | None = None) -> dict[str, Any]:
    """Description du materiel, destinee au manifeste (§3.5)."""
    torch = _torch()
    resolved = resolve_device(device or "auto")
    info: dict[str, Any] = {
        "requested": str(device or "auto"),
        "resolved": resolved,
        "cuda_available": cuda_available(),
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
    """Remet a zero le compteur de VRAM maximale, avant un entrainement."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float | None:
    """VRAM maximale allouee depuis le dernier reset, ou None hors CUDA.

    Metrique de cout de premier ordre (§7.2) : une approche qui ne tient pas en
    memoire n'est pas deployable, quel que soit son OKS.
    """
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def amp_enabled(cfg_amp: bool, mode: str, device: str) -> bool:
    """Decide de l'usage de la precision mixte.

    L'AMP est desactivee hors CUDA (sans effet) et en `mode: debug`, ou la
    reproductibilite bit a bit prime sur la vitesse (§6.4).
    """
    if not bool(cfg_amp):
        return False
    if device == "cpu":
        return False
    if mode == "debug":
        log.info("mode=debug : AMP desactivee au profit de la reproductibilite.")
        return False
    return True
