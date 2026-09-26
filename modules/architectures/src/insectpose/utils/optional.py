"""Import of optional dependencies with an actionable message (never silent)."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import `module` or fail, naming the pip extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"The module '{module}' is required here. Install: pip install -e \".[{extra}]\""
        ) from exc
