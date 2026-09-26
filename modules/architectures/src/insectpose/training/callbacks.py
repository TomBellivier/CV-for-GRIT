"""Framework-agnostic callbacks: early stopping and Optuna reporting.

An approach that cannot be pruned declares `prunable: false` in its config (§6.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EarlyStopping:
    """Stop when the validation metric stagnates."""

    patience: int = 20
    mode: str = "max"
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0
    best_epoch: int = -1

    def step(self, value: float, epoch: int) -> bool:
        """Return True if the training must stop."""
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
    """Report an intermediate metric to Optuna and apply the pruning."""

    trial: Any = None
    history: list[float] = field(default_factory=list)

    def step(self, value: float, epoch: int) -> None:
        """Raise optuna.TrialPruned if the trial must be pruned."""
        self.history.append(float(value))
        if self.trial is None:
            return
        import optuna

        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned
