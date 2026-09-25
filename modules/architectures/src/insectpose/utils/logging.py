"""Journalisation uniforme. Aucun `print` ailleurs dans le projet."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    """Configure le logging racine. Idempotent.

    Effet de bord : ecrit dans `logfile` si fourni.
    """
    global _CONFIGURED
    root = logging.getLogger("insectpose")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(level.upper())
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Logger nomme sous l'espace `insectpose`."""
    setup_logging()
    return logging.getLogger(f"insectpose.{name}")
