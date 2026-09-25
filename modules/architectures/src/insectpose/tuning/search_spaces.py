"""Declaration des espaces de recherche Optuna.

Un espace se declare en YAML (`approach.search_space`) et se traduit ici. Cela evite
qu'un espace soit enterre dans du code d'entrainement et permet de verifier d'un coup
d'oeil que les budgets sont comparables entre approches (§6.3).
"""

from __future__ import annotations

from typing import Any


def suggest_from_spec(trial: Any, spec: Any, prefix: str = "") -> dict[str, Any]:
    """Traduit une declaration YAML en suggestions Optuna.

    Formats supportes :
      {type: float, low: .., high: .., log: bool}
      {type: int, low: .., high: .., step: ..}
      {type: categorical, choices: [...]}
    Retourne un dict de surcharges Hydra {chemin.cle: valeur}.
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
                f"Type d'espace de recherche inconnu : '{kind}' (cle '{key}'). "
                "Attendu : float | int | categorical."
            )
        overrides[name] = value
    return overrides


def to_hydra_overrides(values: dict[str, Any]) -> list[str]:
    """Convertit {cle: valeur} en surcharges Hydra 'cle=valeur'."""
    return [f"{k}={v}" for k, v in values.items()]
