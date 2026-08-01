"""Import de dependances optionnelles avec message actionnable (jamais silencieux)."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Importe `module` ou echoue en indiquant l'extra pip a installer."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise ImportError(
            f"Le module '{module}' est requis ici. Installer : pip install -e \".[{extra}]\""
        ) from exc
