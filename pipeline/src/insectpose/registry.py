"""Registre par nom des composants enfichables (CONVENTIONS.md §4.1).

Un composant s'enregistre par decorateur ; le pipeline ne connait que des noms.
Aucun `if approach == ...` ne doit exister ailleurs dans le projet.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from types import ModuleType
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Table nom -> objet, avec refus des doublons silencieux."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorateur d'enregistrement. Le nom DOIT etre celui du YAML correspondant."""

        def decorator(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise KeyError(
                    f"'{name}' est deja enregistre dans le registre '{self.namespace}'. "
                    "Choisir un autre nom plutot que d'ecraser."
                )
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Recupere un composant enregistre, ou echoue avec la liste des noms valides."""
        if name not in self._items:
            raise KeyError(
                f"'{name}' introuvable dans le registre '{self.namespace}'. "
                f"Disponibles : {sorted(self._items)}. "
                "Verifier que le module est importe (cf. load_plugins)."
            )
        return self._items[name]

    def available(self) -> list[str]:
        """Noms enregistres, tries."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


APPROACHES: Registry[Any] = Registry("approach")
METRICS: Registry[Any] = Registry("metric")
ADAPTERS: Registry[Any] = Registry("adapter")

register_approach = APPROACHES.register
register_metric = METRICS.register
register_adapter = ADAPTERS.register


def load_plugins(package: str) -> list[str]:
    """Importe tous les sous-modules d'un package pour declencher les enregistrements.

    Effet de bord : imports Python uniquement, aucune ecriture disque.
    """
    module: ModuleType = importlib.import_module(package)
    loaded: list[str] = []
    for info in pkgutil.iter_modules(module.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{info.name}")
        loaded.append(info.name)
    return loaded


def load_all_plugins() -> None:
    """Charge approches, metriques et adaptateurs. Appele une fois au demarrage du CLI."""
    load_plugins("insectpose.approaches")
    load_plugins("insectpose.evaluation.metrics")
    load_plugins("insectpose.data.adapters")
