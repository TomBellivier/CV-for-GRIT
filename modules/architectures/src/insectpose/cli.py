"""Single entry point: five verbs, no more (CONVENTIONS.md §5.4).

Usage: python -m insectpose.cli <verb> [key=value ...]
The overrides are standard Hydra overrides.
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
    """Compose the Hydra configuration. No side effect."""
    root = config_dir or Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _split_args(argv: list[str]) -> tuple[str, list[str], dict[str, str]]:
    """Separate the verb, the Hydra overrides and the arguments specific to the CLI."""
    if not argv or argv[0] in ("-h", "--help"):
        raise SystemExit(
            "Usage: python -m insectpose.cli <verb> [key=value ...]\n"
            f"Verbs: {', '.join(VERBS)}\n"
            "Examples:\n"
            "  python -m insectpose.cli prepare data=coleoptera\n"
            "  python -m insectpose.cli train experiment=exp_ref_mean_pose cv.fold=0\n"
            "  python -m insectpose.cli train experiment=exp_a_yolo_pooled folds=[0,1,2]\n"
            "  python -m insectpose.cli tune experiment=exp_a_yolo_pooled folds=all\n"
            "  python -m insectpose.cli evaluate run_id=<run_id>\n"
        )
    verb, rest = argv[0], argv[1:]
    if verb not in VERBS:
        raise SystemExit(f"Unknown verb: '{verb}'. Expected: {', '.join(VERBS)}")
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
    """Dispatch to `pipeline`. Side effects: those of the step called."""
    setup_logging()
    load_all_plugins()
    verb, overrides, cli_args = _split_args(list(argv if argv is not None else sys.argv[1:]))

    # `cv.fold=k` is a handy alias of `fold=k` (the current outer fold).
    overrides = [o.replace("cv.fold=", "fold=") for o in overrides]
    cfg = load_config(overrides)
    log.info("Verb '%s' | approach=%s | data=%s | fold=%s",
             verb, cfg.approach.name, cfg.data.scope, cfg.fold)

    if verb == "prepare":
        return pipeline.cmd_prepare(cfg)
    if verb == "split":
        return pipeline.cmd_split(cfg)
    if verb == "train":
        run_ids = [ctx.run_id for ctx in pipeline.cmd_train_folds(cfg)]
        return run_ids[0] if len(run_ids) == 1 else run_ids
    if verb == "tune":
        return pipeline.cmd_tune(cfg)
    if verb == "report":
        return pipeline.cmd_report(cfg)

    run_id = cli_args.get("run_id")
    if not run_id:
        raise SystemExit(f"The '{verb}' verb requires run_id=<run_id>.")
    if verb == "predict":
        return pipeline.cmd_predict(cfg, run_id, cli_args.get("split", "test"))
    return pipeline.cmd_evaluate(cfg, run_id)


if __name__ == "__main__":
    result = main()
    if isinstance(result, (str, Path)):
        print(result)
    elif isinstance(result, list):
        print("\n".join(str(item) for item in result))
    elif isinstance(result, dict):
        print(OmegaConf.to_yaml(OmegaConf.create(result)))
