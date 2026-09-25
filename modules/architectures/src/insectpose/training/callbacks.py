"""Callbacks agnostiques du framework : early stopping et report Optuna.

Une approche non elagable declare `prunable: false` dans sa config (§6.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EarlyStopping:
    """Arret sur stagnation de la metrique de validation."""

    patience: int = 20
    mode: str = "max"
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0
    best_epoch: int = -1

    def step(self, value: float, epoch: int) -> bool:
        """Retourne True s'il faut arreter l'entrainement."""
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best + self.min_delta)
            or (self.mode == "min" and value < self.best - self.min_delta)
        )
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@dataclass
class OptunaReporter:
    """Remonte une metrique intermediaire a Optuna et applique l'elagage."""

    trial: Any = None
    history: list[float] = field(default_factory=list)

    def step(self, value: float, epoch: int) -> None:
        """Leve optuna.TrialPruned si le trial doit etre elague."""
        self.history.append(float(value))
        if self.trial is None:
            return
        import optuna

        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned
