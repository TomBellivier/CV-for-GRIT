"""Orchestration of the five steps (CONVENTIONS.md §5.4).

prepare -> split -> train -> predict -> evaluate (+ tune, report).
Each step can be called on its own and picks up the artefacts of the previous one: it
must be possible to re-evaluate a three-month-old run without retraining (§1.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from insectpose.context import RunContext, make_run_id
from insectpose.contracts import DATASETS, ContractError
from insectpose.data.coverage import write_coverage
from insectpose.data.datamodule import build_fold_data, load_annotations
from insectpose.data.keypoints import load_schemas
from insectpose.data.measurements import load_measurements
from insectpose.data.schema import validate_single_instance
from insectpose.data.splits import (
    build_full_split,
    build_inner_splits,
    build_splits,
    fold_assignment,
    inner_split_id,
    load_splits,
    make_split_id,
    write_splits,
)
from insectpose.evaluation.aggregate import write_master
from insectpose.evaluation.evaluator import evaluate_run, primary_value
from insectpose.paths import ProjectPaths
from insectpose.registry import ADAPTERS, APPROACHES
from insectpose.reporting.qualitative import export_qualitative
from insectpose.retain import (
    retain_context,
    retain_existing_run,
    retainable_approach,
    write_ensemble_card,
)
from insectpose.utils.hashing import content_hash_annotations
from insectpose.utils.io import read_parquet
from insectpose.utils.logging import get_logger

log = get_logger("pipeline")


# --- shared helpers --------------------------------------------------------------
def _schema_names(cfg: DictConfig) -> list[str]:
    """Keypoint schemas to load for the current data scope.

    Nominal case (ADR-0006): `data.keypoint_schema` is common to the 4 datasets. If one
    day a dataset diverges, leaving this field at null is enough to fall back on one
    schema per dataset, without touching the rest of the pipeline.
    """
    shared = cfg.data.get("keypoint_schema")
    names = [str(shared)] if shared else [str(d) for d in cfg.data.datasets]
    union = cfg.data.get("union_space")
    if union and str(union) not in names:
        names.append(str(union))
    return sorted(set(names))


def _image_size_guard(cfg: DictConfig) -> None:
    """Refuse an input resolution diverging between approaches (ADR-0013)."""
    if not bool(cfg.strict.get("enforce_common_image_size", True)):
        return
    common = [int(v) for v in cfg.protocol.image_size]
    used = cfg.train.image_size
    used = [int(used), int(used)] if isinstance(used, int) else [int(v) for v in used]
    if used != common:
        raise ContractError(
            f"Input resolution {used} != common resolution of the protocol {common}. "
            "An approach trained at another resolution no longer compares the method but "
            "the resolution (ADR-0013). Set strict.enforce_common_image_size=false for an "
            "exploration outside the report."
        )


def _load_context_data(cfg: DictConfig, paths: ProjectPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load annotations + schemas, applying the strict guards."""
    annotations = load_annotations([str(d) for d in cfg.data.datasets], paths)
    schemas = load_schemas(
        _schema_names(cfg), paths.configs, strict=bool(cfg.strict.require_validated_keypoints)
    )
    return annotations, schemas


def _git_guard(cfg: DictConfig, paths: ProjectPaths) -> None:
    """Refuse a modified repository if strict.require_clean_git (untraceable results)."""
    if not bool(cfg.strict.require_clean_git):
        return
    from insectpose.context import _git_state

    state = _git_state(paths.root)
    if state.get("dirty"):
        raise ContractError(
            "Modified git repository and strict.require_clean_git=true: commit before "
            "producing quotable results."
        )


# --- steps ------------------------------------------------------------------------
def cmd_prepare(cfg: DictConfig) -> list[Path]:
    """raw -> canonical format (contract 1), then coverage report (ADR-0016).

    Side effect: data/processed/<dataset>/annotations.parquet and
    data/processed/coverage_*.{parquet,json}.
    """
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_writable_dirs()
    written: list[Path] = []
    for dataset in [str(d) for d in cfg.data.datasets]:
        adapter_cls = ADAPTERS.get(str(cfg.data.adapter))
        options = OmegaConf.to_container(cfg.data.adapter_options, resolve=True) or {}
        assert isinstance(options, dict)
        options.setdefault("keypoint_schema", str(cfg.data.get("keypoint_schema") or dataset))
        # Roots already resolved: a relative path option is read from the project root,
        # not from the folder the command is launched from.
        options.setdefault("project_root", str(paths.root))
        options.setdefault("configs_dir", str(paths.configs))
        source = paths.raw_dir(dataset, cfg.data.get("raw_subdir"))
        adapter = adapter_cls(dataset=dataset, source_dir=source, options=options)
        out = adapter.run(paths)
        log.info("[%s] canonical annotations written: %s", dataset, out)
        written.append(out)

    annotations = load_annotations([str(d) for d in cfg.data.datasets], paths)
    if bool(cfg.data.get("single_instance_per_image", True)):
        validate_single_instance(annotations)
    _write_coverage(cfg, paths)
    return written


def _write_coverage(cfg: DictConfig, paths: ProjectPaths) -> None:
    """Coverage report of the keypoints and measurements (ADR-0016).

    Covers ALL the datasets already prepared, not only the one that just was:
    otherwise `prepare data=<one dataset>` would overwrite the global report and the
    "absent everywhere" row would become wrong.
    """
    prepared = [d for d in DATASETS if paths.annotations(d).exists()]
    if not prepared:
        return
    annotations = load_annotations(prepared, paths)
    log.info("Coverage computed on %d prepared dataset(s): %s", len(prepared), prepared)
    schemas = load_schemas(_schema_names(cfg), paths.configs)
    spec = None
    measurements = cfg.eval.get("measurements")
    if measurements is not None and bool(measurements.enabled):
        spec = load_measurements(Path(str(measurements.file)))
    coverage = cfg.data.get("coverage") or {}
    write_coverage(
        annotations, schemas, paths.processed, spec=spec,
        absent_max=float(coverage.get("absent_max", 0.01)),
        rare_max=float(coverage.get("rare_max", 0.5)),
        measurement_min_rate=float(coverage.get("measurement_min_rate", 0.5)),
    )


def cmd_split(cfg: DictConfig) -> Path:
    """Generate the outer AND inner folds (contract 2).

    The inner splits serve the nested HPO: they only contain images of the outer train,
    so the outer test stays untouched by any search (ADR-0012).
    Side effect: data/splits/<split_id>*.{parquet,json}.
    """
    paths = ProjectPaths.from_config(cfg)
    annotations, _ = _load_context_data(cfg, paths)
    table, meta = build_splits(annotations, cfg)
    out = write_splits(table, meta, paths)
    log.info("Outer split '%s': %d images, %d groups, %d folds.",
             meta["split_id"], meta["n_images"], meta["n_groups"], meta["n_folds"])

    for outer_fold in range(int(meta["n_folds"])):
        inner_table, inner_meta = build_inner_splits(annotations, table, outer_fold, cfg)
        write_splits(inner_table, inner_meta, paths)
        log.info("  inner split '%s': %d images, %d folds.",
                 inner_meta["split_id"], inner_meta["n_images"], inner_meta["n_folds"])
    return out


def _prepare_run(cfg: DictConfig, extra: dict[str, Any] | None = None
                 ) -> tuple[RunContext, Any, Any]:
    """Assemble the context, the data of the fold and the approach instance."""
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_writable_dirs()
    _git_guard(cfg, paths)
    _image_size_guard(cfg)

    annotations, schemas = _load_context_data(cfg, paths)
    # cfg.split_id lets the HPO point at an INNER split (ADR-0012).
    split_id = str(cfg.split_id) if cfg.get("split_id") else make_split_id(cfg)
    if not paths.split_file(split_id).exists():
        raise FileNotFoundError(
            f"Split '{split_id}' missing. Run: python -m insectpose.cli split"
        )
    table, _ = load_splits(split_id, paths, annotations)
    assignment = fold_assignment(table, int(cfg.fold))
    data = build_fold_data(annotations, assignment, schemas, paths)

    cfg = cfg.copy()
    OmegaConf.update(cfg, "split_id", split_id, force_add=True)
    ctx = RunContext(
        run_id=make_run_id(cfg, content_hash_annotations(annotations)),
        cfg=cfg, paths=paths, fold=int(cfg.fold), split_id=split_id,
        content_hash=content_hash_annotations(annotations), extra=dict(extra or {}),
    )
    approach = APPROACHES.get(str(cfg.approach.name))(cfg)
    return ctx, data, approach


def cmd_train(cfg: DictConfig, extra: dict[str, Any] | None = None,
              do_evaluate: bool = True, retain_replace: bool = True) -> RunContext:
    """Train a fold, predict and evaluate. Idempotent: a complete run is skipped (§8.1).

    Ends with the export of the weights to retained_models/ (`retain.enabled`), from
    where `pipeline/` loads them. The export is redone on an already complete run:
    running the same command again is enough to regenerate a deleted copy.
    `retain_replace=True` replaces the ensemble of retained_models/pose/ by this single
    model; `tune` sets it to False to add its folds to one another.
    """
    ctx, data, approach = _prepare_run(cfg, extra)
    if ctx.is_complete() and not bool(cfg.force):
        log.info("Run already complete, skipped: %s (force=true to replay).", ctx.run_id)
        retain_context(ctx, replace=retain_replace)
        return ctx

    ctx.setup()
    log.info("Run %s | folds: %s", ctx.run_id, data.summary())
    approach.fit(data, ctx)

    for split in ("val", "test"):
        approach.predict(data.role(split), ctx, split)

    if do_evaluate:
        annotations, schemas = _load_context_data(cfg, ctx.paths)
        evaluate_run(ctx.run_id, ctx.paths, annotations, schemas, cfg.eval,
                     approach=str(cfg.approach.name), split_id=ctx.split_id)
        _export_qualitative(ctx, data, schemas)
    ctx.write_manifest()   # written LAST: marks the run as complete
    # after the manifest: only a complete run is exported (§8.2)
    retain_context(ctx, replace=retain_replace)
    return ctx


def selected_folds(cfg: DictConfig, default_all: bool) -> list[int]:
    """Outer folds a command runs (ADR-0039).

    `folds` if it is set: a list of fold indices, a single index, or "all". Otherwise
    `fold` alone (`train`, `default_all=False`) or every outer fold (`tune`). Each index
    is checked against `cv.n_folds`: an unknown fold would silently train nothing.
    """
    n_folds = int(cfg.cv.n_folds)
    raw = cfg.get("folds")
    if raw is None:
        return list(range(n_folds)) if default_all else [int(cfg.fold)]
    if isinstance(raw, str):
        if raw.strip().lower() != "all":
            raise ValueError(f"folds={raw!r}: expected a list of fold indices or 'all'.")
        return list(range(n_folds))
    values = [raw] if isinstance(raw, int) else list(raw)
    folds = sorted({int(value) for value in values})
    if not folds:
        raise ValueError("folds=[]: at least one outer fold is needed.")
    outside = [fold for fold in folds if not 0 <= fold < n_folds]
    if outside:
        raise ValueError(f"folds {outside} outside 0..{n_folds - 1} (cv.n_folds={n_folds}).")
    return folds


def cmd_train_folds(cfg: DictConfig) -> list[RunContext]:
    """`train` over one or several outer folds (`folds`), gathered into ONE ensemble.

    The first fold replaces the ensemble of retained_models/pose/, the next ones are
    added to it, like the final folds of `tune` (ADR-0039). `ensemble.json` then carries
    the estimate of these folds, each one measured on its own untouched test.
    """
    contexts: list[RunContext] = []
    for position, fold in enumerate(selected_folds(cfg, default_all=False)):
        fold_cfg = cfg.copy()
        OmegaConf.update(fold_cfg, "fold", fold)
        contexts.append(cmd_train(fold_cfg, retain_replace=position == 0))

    if bool(cfg.retain.enabled) and retainable_approach(cfg) and str(cfg.mode) != "smoke":
        paths = ProjectPaths.from_config(cfg)
        run_ids = {ctx.fold: ctx.run_id for ctx in contexts}
        card = write_ensemble_card(paths, {"source": "train", "folds": list(run_ids),
                                           "cv_estimate": _cv_estimate(cfg, paths, run_ids)})
        log.info("Retained ensemble: %s (%d fold(s))", card, len(run_ids))
    return contexts


def _export_qualitative(ctx: RunContext, data: Any, schemas: dict[str, Any]) -> None:
    """Pred vs GT figures of the run (§8.5). Side effect: runs/<run_id>/figures/."""
    cfg = ctx.cfg.eval.qualitative
    if not bool(cfg.enabled):
        return
    split = str(cfg.split)
    predictions = read_parquet(ctx.paths.predictions(ctx.run_id, split, ctx.fold))
    figures = export_qualitative(
        run_dir=ctx.run_dir, gt=data.role(split).annotations.reset_index(drop=True),
        pred=predictions, schemas=schemas, eval_cfg=ctx.cfg.eval,
        data_root=ctx.paths.data, seed=ctx.seed("qualitative"),
    )
    ctx.extra.setdefault("n_qualitative_figures", len(figures))


def cmd_fit_full(cfg: DictConfig, extra: dict[str, Any] | None = None,
                 card: dict[str, Any] | None = None) -> RunContext:
    """Train a model on ALL the images, with the hyperparameters already retained.

    This run neither predicts nor evaluates: no test is held out, so any metric computed
    here would be measured on images seen at training time (ADR-0012). It writes no
    `metrics.parquet`, which is enough to keep it out of `master.parquet` (§8.4), and
    its manifest carries `role_in_protocol: final_full`. The expected performance comes
    from the outer folds and travels in the model card (`card`).

    Side effect: writes data/splits/<split_id>__full.*, runs/<run_id>/ and the export
    to retained_models/.
    """
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_writable_dirs()
    annotations, _ = _load_context_data(cfg, paths)

    fraction = cfg.tuning.final_val_fraction
    table, meta = build_full_split(
        annotations, cfg, None if fraction is None else float(fraction)
    )
    write_splits(table, meta, paths)

    full_cfg = cfg.copy()
    OmegaConf.update(full_cfg, "split_id", meta["split_id"], force_add=True)
    OmegaConf.update(full_cfg, "fold", 0)
    OmegaConf.update(full_cfg, "tag", f"{cfg.tag}-final", force_add=True)

    run_extra = {"role_in_protocol": "final_full", **(extra or {})}
    ctx, data, approach = _prepare_run(full_cfg, run_extra)
    # Added to the ensemble of the outer folds, never in its place (tuning.final_full_fit).
    if ctx.is_complete() and not bool(cfg.force):
        log.info("Final model already trained, skipped: %s (force=true to replay).", ctx.run_id)
        retain_context(ctx, extra_card=card, replace=False)
        return ctx

    ctx.setup()
    log.info("Final model %s | %s", ctx.run_id, data.summary())
    approach.fit(data, ctx)
    ctx.write_manifest(evaluable=False)
    retain_context(ctx, extra_card=card, replace=False)
    return ctx


def cmd_predict(cfg: DictConfig, run_id: str, split: str = "test") -> Path:
    """Reload a run and regenerate its predictions, without retraining."""
    ctx, data, _ = _prepare_run(cfg)
    approach = APPROACHES.get(str(cfg.approach.name)).load(ctx.paths.run_dir(run_id), cfg)
    return approach.predict(data.role(split), ctx, split)


def cmd_evaluate(cfg: DictConfig, run_id: str) -> Path:
    """Re-evaluate an existing run from its predictions only (§7.1).

    Also exports its weights to retained_models/ (`retain.enabled`): it is the way to
    retain a three-month-old run without retraining it (§1.4). Like `train`, it
    replaces the ensemble in place by this single model.
    """
    paths = ProjectPaths.from_config(cfg)
    annotations, schemas = _load_context_data(cfg, paths)
    out = evaluate_run(run_id, paths, annotations, schemas, cfg.eval)
    log.info("Metrics written: %s", out)
    retain_existing_run(run_id, paths, cfg)
    return out


def cmd_tune(cfg: DictConfig) -> dict[str, Any]:
    """Optimise the hyperparameters, then retrain the outer folds (ADR-0012).

    Nested protocol:
      1. for each outer fold, the HPO runs on the INNER folds of its train;
      2. the best hyperparameters are then applied to the whole outer fold;
      3. the outer test fold has never been used to choose a hyperparameter.

    In `tune_once` mode, step 1 is only done on `tuning.tuning_outer_fold` and the
    result is reused for every outer fold (cheaper, to be documented).

    `folds` restricts the outer folds that are retrained (and, in `nested` mode,
    searched): all of them by default (ADR-0039).
    """
    from insectpose.tuning.objective import (
        build_study,
        completed_trials,
        make_objective,
        remaining_trials,
        save_best,
    )

    paths = ProjectPaths.from_config(cfg)
    outer_split_id = make_split_id(cfg)
    mode = str(cfg.tuning.mode)
    outer_folds = selected_folds(cfg, default_all=True)
    tuning_folds = outer_folds if mode == "nested" else [int(cfg.tuning.tuning_outer_fold)]

    results: dict[str, Any] = {"mode": mode, "outer": {}}
    for outer_fold in tuning_folds:
        inner_id = inner_split_id(outer_split_id, outer_fold)
        if not paths.split_file(inner_id).exists():
            raise FileNotFoundError(
                f"Inner split '{inner_id}' missing. Run 'split' again: the nested HPO "
                "requires inner folds built on the outer train only."
            )
        search_cfg = cfg.copy()
        OmegaConf.update(search_cfg, "split_id", inner_id, force_add=True)

        def run_fn(base_cfg: DictConfig, overrides: dict[str, Any]) -> float:
            return _run_trial_fold(base_cfg, overrides)

        study = build_study(search_cfg, paths, suffix=f"outer{outer_fold}")
        budget = int(cfg.tuning.n_trials)
        todo = remaining_trials(study, budget)
        if todo == 0:
            log.info("Outer fold %d: budget already reached (%d/%d trials), nothing to do.",
                     outer_fold, completed_trials(study), budget)
        else:
            if completed_trials(study) > 0:
                log.info("Outer fold %d: resuming, %d trial(s) already finished, %d to run "
                         "to reach the budget of %d.",
                         outer_fold, completed_trials(study), todo, budget)
            study.optimize(
                make_objective(search_cfg, run_fn),
                n_trials=todo,
                timeout=cfg.tuning.get("timeout_s"),
            )
        best = save_best(study, search_cfg, paths)
        results["outer"][outer_fold] = best
        log.info("Outer fold %d | best inner %s = %s",
                 outer_fold, cfg.eval.primary_metric, best["best_value"])

    # Retraining of the outer folds with the retained hyperparameters. Each one is
    # exported: together, they form the delivered model, which the pipeline averages.
    # The first one replaces the previous ensemble, the next ones are added to it.
    final_runs: dict[int, str] = {}
    for outer_fold in outer_folds:
        source = outer_fold if mode == "nested" else int(cfg.tuning.tuning_outer_fold)
        best = results["outer"][source]
        final_cfg = cfg.copy()
        OmegaConf.update(final_cfg, "split_id", outer_split_id, force_add=True)
        OmegaConf.update(final_cfg, "fold", outer_fold)
        OmegaConf.update(final_cfg, "tag", f"{cfg.tag}-tuned", force_add=True)
        for key, value in best["best_params"].items():
            OmegaConf.update(final_cfg, key, value, force_add=True)
        ctx = cmd_train(
            final_cfg,
            extra={"optuna_study": best["study_name"], "hpo_mode": mode,
                   "hpo_source_fold": source, "hpo_n_trials": best["n_trials_completed"],
                   # ADR-0012: each outer fold LEGITIMATELY retains its own
                   # hyperparameters. They are excluded from the identity of the model,
                   # otherwise each fold would form a distinct variant and the
                   # dispersion across folds would disappear from the tables.
                   "hpo_overridden_keys": list(best["best_params"])},
            retain_replace=outer_fold == outer_folds[0],
        )
        final_runs[outer_fold] = ctx.run_id
    results["final_runs"] = final_runs

    estimate = _cv_estimate(cfg, ProjectPaths.from_config(cfg), final_runs)
    if bool(cfg.retain.enabled) and retainable_approach(cfg) and str(cfg.mode) != "smoke":
        card = write_ensemble_card(paths, {"source": "tune", "hpo_mode": mode,
                                           "folds": list(final_runs),
                                           "cv_estimate": estimate})
        log.info("Retained ensemble: %s | estimate of the outer folds: %s", card, estimate)

    # Optional (tuning.final_full_fit): one more model, trained on ALL the images with
    # the retained hyperparameters, ADDED to the ensemble of the folds.
    if bool(cfg.tuning.final_full_fit):
        source, best = _best_study(results, cfg)
        full_cfg = cfg.copy()
        for key, value in best["best_params"].items():
            OmegaConf.update(full_cfg, key, value, force_add=True)
        ctx = cmd_fit_full(
            full_cfg,
            extra={"optuna_study": best["study_name"], "hpo_mode": mode,
                   "hpo_source_fold": source, "hpo_n_trials": best["n_trials_completed"],
                   "hpo_overridden_keys": list(best["best_params"])},
            card={"trained_on": "all_images", "hyperparameters": dict(best["best_params"]),
                  "cv_estimate": estimate},
        )
        results["full_run"] = ctx.run_id
        log.info("Full-data model: %s | estimate of the outer folds: %s", ctx.run_id, estimate)
    return results


def _best_study(results: dict[str, Any], cfg: DictConfig) -> tuple[int, dict[str, Any]]:
    """Study whose hyperparameters go into the final model.

    In `tune_once` there is only one. In `nested`, each outer fold retained its own
    (ADR-0012): the final model can only carry one set, so the one with the best inner
    value is taken, in the sense of `eval.primary_direction`.
    """
    outer = results["outer"]
    if len(outer) == 1:
        source = next(iter(outer))
        return source, outer[source]
    better = max if str(cfg.eval.primary_direction) == "maximize" else min
    source = better(outer, key=lambda fold: outer[fold]["best_value"])
    log.info("Hyperparameters of outer fold %s retained for the final model "
             "(best inner value: %s).", source, outer[source]["best_value"])
    return source, outer[source]


def _cv_estimate(cfg: DictConfig, paths: ProjectPaths, run_ids: dict[int, str]) -> dict[str, Any]:
    """Expected performance of the delivered model: the measured one of the outer folds.

    A model trained on every image cannot evaluate itself (no untouched test): this is
    the only honest estimate that can be attached to it, and it is labelled as such in
    the model card.
    """
    values: list[float] = []
    for run_id in run_ids.values():
        path = paths.metrics(run_id)
        if not path.exists():
            continue
        try:
            values.append(primary_value(read_parquet(path), cfg.eval))
        except Exception as exc:  # noqa: BLE001 - an incomplete card is better than a failure
            log.warning("Metrics unreadable for %s: %s", run_id, exc)
    if not values:
        return {}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return {
        "source": "outer_folds",
        "metric": str(cfg.eval.primary_metric),
        "mean": mean,
        "std": variance ** 0.5,
        "n_folds": len(values),
        "runs": list(run_ids.values()),
    }


def _run_trial_fold(base_cfg: DictConfig, overrides: dict[str, Any]) -> float:
    """Run an inner fold of a trial and return the primary metric.

    A trial is a complete run, with a run_id and a manifest: it stays auditable (§6.3).
    """
    trial_cfg = base_cfg.copy()
    fold = int(overrides.pop("fold"))
    trial_number = overrides.pop("trial_number", None)
    study = overrides.pop("optuna_study", None)
    for key, value in overrides.items():
        OmegaConf.update(trial_cfg, key, value, force_add=True)
    OmegaConf.update(trial_cfg, "fold", fold)
    OmegaConf.update(trial_cfg, "tag", f"{base_cfg.tag}-trial{trial_number}", force_add=True)
    # No qualitative export for the trials: useless noise, non-zero cost.
    OmegaConf.update(trial_cfg, "eval.qualitative.enabled", False)
    ctx = cmd_train(trial_cfg, extra={"trial_number": trial_number, "optuna_study": study,
                                      "role_in_protocol": "hpo_trial"})
    metrics = read_parquet(ctx.paths.metrics(ctx.run_id))
    return primary_value(metrics, base_cfg.eval)


def cmd_report(cfg: DictConfig) -> Path:
    """Aggregate every complete run and produce the report tables (§8.4)."""
    from insectpose.reporting.report import write_report

    paths = ProjectPaths.from_config(cfg)
    master = write_master(paths)
    log.info("Aggregate written: %s", master)
    return write_report(paths, cfg)
