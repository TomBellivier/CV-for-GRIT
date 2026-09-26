"""Declaration of the Optuna search spaces.

A space is declared in YAML (`approach.search_space`) and translated here. It keeps a
space from being buried in training code and lets one check at a glance that the
budgets are comparable across approaches (§6.3).
"""

from __future__ import annotations

from typing import Any


def suggest_from_spec(trial: Any, spec: Any, prefix: str = "") -> dict[str, Any]:
    """Translate a YAML declaration into Optuna suggestions.

    Supported formats:
      {type: float, low: .., high: .., log: bool}
      {type: int, low: .., high: .., step: ..}
      {type: categorical, choices: [...]}
    Returns a dict of Hydra overrides {path.key: value}.
    """
    overrides: dict[str, Any] = {}
    if not spec:
        return overrides
    for key, definition in dict(spec).items():
        name = f"{prefix}.{key}" if prefix else key
        kind = str(definition["type"])
        if kind == "float":
            value: Any = trial.suggest_float(
                name, float(definition["low"]), float(definition["high"]),
                log=bool(definition.get("log", False)),
            )
        elif kind == "int":
            value = trial.suggest_int(
                name, int(definition["low"]), int(definition["high"]),
                step=int(definition.get("step", 1)),
            )
        elif kind == "categorical":
            value = trial.suggest_categorical(name, list(definition["choices"]))
        else:
            raise ValueError(
                f"Unknown search space type: '{kind}' (key '{key}'). "
                "Expected: float | int | categorical."
            )
        overrides[name] = value
    return overrides


def to_hydra_overrides(values: dict[str, Any]) -> list[str]:
    """Convert {key: value} into Hydra overrides 'key=value'."""
    return [f"{k}={v}" for k, v in values.items()]
