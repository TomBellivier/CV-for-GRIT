"""Point d'entree unique : cinq verbes, pas plus (CONVENTIONS.md §5.4).

Usage : python -m insectpose.cli <verbe> [cle=valeur ...]
Les surcharges sont des surcharges Hydra standard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from insectpose import pipeline
from insectpose.registry import load_all_plugins
from insectpose.utils.logging import get_logger, setup_logging

log = get_logger("cli")

VERBS = ("prepare", "split", "train", "predict", "evaluate", "tune", "report")


def load_config(overrides: list[str], config_dir: Path | None = None) -> DictConfig:
    """Compose la configuration Hydra. Aucun effet de bord."""
    root = config_dir or Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _split_args(argv: list[str]) -> tuple[str, list[str], dict[str, str]]:
    """Separe le verbe, les surcharges Hydra et les arguments propres au CLI."""
    if not argv or argv[0] in ("-h", "--help"):
        raise SystemExit(
            "Usage : python -m insectpose.cli <verbe> [cle=valeur ...]\n"
            f"Verbes : {', '.join(VERBS)}\n"
            "Exemples :\n"
            "  python -m insectpose.cli prepare data=coleoptera\n"
            "  python -m insectpose.cli train experiment=exp_ref_mean_pose cv.fold=0\n"
            "  python -m insectpose.cli evaluate run_id=<run_id>\n"
        )
    verb, rest = argv[0], argv[1:]
    if verb not in VERBS:
        raise SystemExit(f"Verbe inconnu : '{verb}'. Attendu : {', '.join(VERBS)}")
    cli_args: dict[str, str] = {}
    overrides: list[str] = []
    for item in rest:
        if item.startswith("run_id="):
            cli_args["run_id"] = item.split("=", 1)[1]
        elif item.startswith("split=") and verb in ("predict",):
            cli_args["split"] = item.split("=", 1)[1]
        else:
            overrides.append(item)
    return verb, overrides, cli_args


def main(argv: list[str] | None = None) -> Any:
    """Dispatch vers `pipeline`. Effets de bord : ceux de l'etape appelee."""
    setup_logging()
    load_all_plugins()
    verb, overrides, cli_args = _split_args(list(argv if argv is not None else sys.argv[1:]))

    # `cv.fold=k` est un alias pratique de `fold=k` (le fold externe courant).
    overrides = [o.replace("cv.fold=", "fold=") for o in overrides]
    cfg = load_config(overrides)
    log.info("Verbe '%s' | approche=%s | data=%s | fold=%s",
             verb, cfg.approach.name, cfg.data.scope, cfg.fold)

    if verb == "prepare":
        return pipeline.cmd_prepare(cfg)
    if verb == "split":
        return pipeline.cmd_split(cfg)
    if verb == "train":
        return pipeline.cmd_train(cfg).run_id
    if verb == "tune":
        return pipeline.cmd_tune(cfg)
    if verb == "report":
        return pipeline.cmd_report(cfg)

    run_id = cli_args.get("run_id")
    if not run_id:
        raise SystemExit(f"Le verbe '{verb}' exige run_id=<run_id>.")
    if verb == "predict":
        return pipeline.cmd_predict(cfg, run_id, cli_args.get("split", "test"))
    return pipeline.cmd_evaluate(cfg, run_id)


if __name__ == "__main__":
    result = main()
    if isinstance(result, (str, Path)):
        print(result)
    elif isinstance(result, dict):
        print(OmegaConf.to_yaml(OmegaConf.create(result)))
