"""Orchestration des cinq etapes (CONVENTIONS.md §5.4).

prepare -> split -> train -> predict -> evaluate (+ tune, report).
Chaque etape est appelable seule et reprend les artefacts de la precedente : on doit
pouvoir re-evaluer un run vieux de trois mois sans reentrainer (§1.4).
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
from insectpose.utils.hashing import content_hash_annotations
from insectpose.utils.io import read_parquet
from insectpose.utils.logging import get_logger

log = get_logger("pipeline")


# --- helpers partages -------------------------------------------------------
def _schema_names(cfg: DictConfig) -> list[str]:
    """Schemas de keypoints a charger pour le perimetre de donnees courant.

    Cas nominal (ADR-0006) : `data.keypoint_schema` est commun aux 4 datasets. Si un
    jour un dataset diverge, il suffit de laisser ce champ a null pour retomber sur un
    schema par dataset, sans toucher au reste du pipeline.
    """
    shared = cfg.data.get("keypoint_schema")
    names = [str(shared)] if shared else [str(d) for d in cfg.data.datasets]
    union = cfg.data.get("union_space")
    if union and str(union) not in names:
        names.append(str(union))
    return sorted(set(names))


def _image_size_guard(cfg: DictConfig) -> None:
    """Refuse une resolution d'entree divergente entre approches (ADR-0013)."""
    if not bool(cfg.strict.get("enforce_common_image_size", True)):
        return
    common = [int(v) for v in cfg.protocol.image_size]
    used = cfg.train.image_size
    used = [int(used), int(used)] if isinstance(used, int) else [int(v) for v in used]
    if used != common:
        raise ContractError(
            f"Resolution d'entree {used} != resolution commune du protocole {common}. "
            "Une approche entrainee a une autre resolution ne compare plus la methode "
            "mais la resolution (ADR-0013). Passer strict.enforce_common_image_size=false "
            "pour une exploration hors rapport."
        )


def _load_context_data(cfg: DictConfig, paths: ProjectPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge annotations + schemas, en appliquant les garde-fous stricts."""
    annotations = load_annotations([str(d) for d in cfg.data.datasets], paths)
    schemas = load_schemas(
        _schema_names(cfg), paths.configs, strict=bool(cfg.strict.require_validated_keypoints)
    )
    return annotations, schemas


def _git_guard(cfg: DictConfig, paths: ProjectPaths) -> None:
    """Refuse un depot modifie si strict.require_clean_git (resultats non tracables)."""
    if not bool(cfg.strict.require_clean_git):
        return
    from insectpose.context import _git_state

    state = _git_state(paths.root)
    if state.get("dirty"):
        raise ContractError(
            "Depot git modifie et strict.require_clean_git=true : commiter avant de "
            "produire des resultats citables."
        )


# --- etapes -----------------------------------------------------------------
def cmd_prepare(cfg: DictConfig) -> list[Path]:
    """raw -> format canonique (contrat 1), puis rapport de couverture (ADR-0016).

    Effet de bord : data/processed/<dataset>/annotations.parquet et
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
        source = paths.raw_dir(dataset, cfg.data.get("raw_subdir"))
        adapter = adapter_cls(dataset=dataset, source_dir=source, options=options)
        out = adapter.run(paths)
        log.info("[%s] annotations canoniques ecrites : %s", dataset, out)
        written.append(out)

    annotations = load_annotations([str(d) for d in cfg.data.datasets], paths)
    if bool(cfg.data.get("single_instance_per_image", True)):
        validate_single_instance(annotations)
    _write_coverage(cfg, paths)
    return written


def _write_coverage(cfg: DictConfig, paths: ProjectPaths) -> None:
    """Rapport de couverture des keypoints et des mesures (ADR-0016).

    Porte sur TOUS les datasets deja prepares, pas seulement sur celui qui vient de
    l'etre : sinon `prepare data=<un dataset>` ecraserait le rapport global et la
    ligne "absent partout" deviendrait fausse.
    """
    prepared = [d for d in DATASETS if paths.annotations(d).exists()]
    if not prepared:
        return
    annotations = load_annotations(prepared, paths)
    log.info("Couverture calculee sur %d dataset(s) prepare(s) : %s", len(prepared), prepared)
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
    """Genere les folds externes ET internes (contrat 2).

    Les decoupages internes servent a l'HPO nichee : ils ne contiennent que des images
    du train externe, donc le test externe reste vierge de toute recherche (ADR-0012).
    Effet de bord : data/splits/<split_id>*.{parquet,json}.
    """
    paths = ProjectPaths.from_config(cfg)
    annotations, _ = _load_context_data(cfg, paths)
    table, meta = build_splits(annotations, cfg)
    out = write_splits(table, meta, paths)
    log.info("Decoupage externe '%s' : %d images, %d groupes, %d folds.",
             meta["split_id"], meta["n_images"], meta["n_groups"], meta["n_folds"])

    for outer_fold in range(int(meta["n_folds"])):
        inner_table, inner_meta = build_inner_splits(annotations, table, outer_fold, cfg)
        write_splits(inner_table, inner_meta, paths)
        log.info("  decoupage interne '%s' : %d images, %d folds.",
                 inner_meta["split_id"], inner_meta["n_images"], inner_meta["n_folds"])
    return out


def _prepare_run(cfg: DictConfig, extra: dict[str, Any] | None = None
                 ) -> tuple[RunContext, Any, Any]:
    """Assemble contexte, donnees du fold et instance d'approche."""
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_writable_dirs()
    _git_guard(cfg, paths)
    _image_size_guard(cfg)

    annotations, schemas = _load_context_data(cfg, paths)
    # cfg.split_id permet de pointer un decoupage INTERNE pendant l'HPO (ADR-0012).
    split_id = str(cfg.split_id) if cfg.get("split_id") else make_split_id(cfg)
    if not paths.split_file(split_id).exists():
        raise FileNotFoundError(
            f"Decoupage '{split_id}' absent. Lancer : python -m insectpose.cli split"
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
              do_evaluate: bool = True) -> RunContext:
    """Entraine un fold, predit et evalue. Idempotent : un run complet est saute (§8.1)."""
    ctx, data, approach = _prepare_run(cfg, extra)
    if ctx.is_complete() and not bool(cfg.force):
        log.info("Run deja complet, saute : %s (force=true pour rejouer).", ctx.run_id)
        return ctx

    ctx.setup()
    log.info("Run %s | folds : %s", ctx.run_id, data.summary())
    approach.fit(data, ctx)

    for split in ("val", "test"):
        approach.predict(data.role(split), ctx, split)

    if do_evaluate:
        annotations, schemas = _load_context_data(cfg, ctx.paths)
        evaluate_run(ctx.run_id, ctx.paths, annotations, schemas, cfg.eval,
                     approach=str(cfg.approach.name))
        _export_qualitative(ctx, data, schemas)
    ctx.write_manifest()   # ecrit EN DERNIER : marque le run comme complet
    return ctx


def _export_qualitative(ctx: RunContext, data: Any, schemas: dict[str, Any]) -> None:
    """Figures pred vs GT du run (§8.4). Effet de bord : runs/<run_id>/figures/."""
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


def cmd_predict(cfg: DictConfig, run_id: str, split: str = "test") -> Path:
    """Recharge un run et regenere ses predictions, sans reentrainement."""
    ctx, data, _ = _prepare_run(cfg)
    approach = APPROACHES.get(str(cfg.approach.name)).load(ctx.paths.run_dir(run_id), cfg)
    return approach.predict(data.role(split), ctx, split)


def cmd_evaluate(cfg: DictConfig, run_id: str) -> Path:
    """Re-evalue un run existant a partir de ses seules predictions (§7.1)."""
    paths = ProjectPaths.from_config(cfg)
    annotations, schemas = _load_context_data(cfg, paths)
    out = evaluate_run(run_id, paths, annotations, schemas, cfg.eval)
    log.info("Metriques ecrites : %s", out)
    return out


def cmd_tune(cfg: DictConfig) -> dict[str, Any]:
    """Optimise les hyperparametres, puis reentraine les folds externes (ADR-0012).

    Protocole niche :
      1. pour chaque fold externe, l'HPO tourne sur les folds INTERNES de son train ;
      2. les meilleurs hyperparametres sont ensuite appliques au fold externe entier ;
      3. le fold de test externe n'a jamais servi a choisir un hyperparametre.

    En mode `tune_once`, l'etape 1 n'est faite que sur `tuning.tuning_outer_fold` et le
    resultat est reutilise pour tous les folds externes (moins couteux, a documenter).
    """
    from insectpose.tuning.objective import build_study, make_objective, save_best

    paths = ProjectPaths.from_config(cfg)
    outer_split_id = make_split_id(cfg)
    mode = str(cfg.tuning.mode)
    n_outer = int(cfg.cv.n_folds)
    outer_folds = list(range(n_outer))
    tuning_folds = outer_folds if mode == "nested" else [int(cfg.tuning.tuning_outer_fold)]

    results: dict[str, Any] = {"mode": mode, "outer": {}}
    for outer_fold in tuning_folds:
        inner_id = inner_split_id(outer_split_id, outer_fold)
        if not paths.split_file(inner_id).exists():
            raise FileNotFoundError(
                f"Decoupage interne '{inner_id}' absent. Relancer 'split' : l'HPO nichee "
                "exige des folds internes construits sur le seul train externe."
            )
        search_cfg = cfg.copy()
        OmegaConf.update(search_cfg, "split_id", inner_id, force_add=True)

        def run_fn(base_cfg: DictConfig, overrides: dict[str, Any]) -> float:
            return _run_trial_fold(base_cfg, overrides)

        study = build_study(search_cfg, paths, suffix=f"outer{outer_fold}")
        study.optimize(
            make_objective(search_cfg, run_fn),
            n_trials=int(cfg.tuning.n_trials),
            timeout=cfg.tuning.get("timeout_s"),
        )
        best = save_best(study, search_cfg, paths)
        results["outer"][outer_fold] = best
        log.info("Fold externe %d | meilleur %s interne = %s",
                 outer_fold, cfg.eval.primary_metric, best["best_value"])

    # Reentrainement des folds externes avec les hyperparametres retenus
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
                   # ADR-0012 : chaque fold externe retient LEGITIMEMENT ses propres
                   # hyperparametres. On les exclut de l'identite du modele, sinon
                   # chaque fold formerait une variante distincte et la dispersion
                   # inter-folds disparaitrait des tableaux.
                   "hpo_overridden_keys": list(best["best_params"])},
        )
        final_runs[outer_fold] = ctx.run_id
    results["final_runs"] = final_runs
    return results


def _run_trial_fold(base_cfg: DictConfig, overrides: dict[str, Any]) -> float:
    """Execute un fold interne d'un trial et retourne la metrique primaire.

    Un trial est un run complet, avec run_id et manifeste : il reste auditable (§6.3).
    """
    trial_cfg = base_cfg.copy()
    fold = int(overrides.pop("fold"))
    trial_number = overrides.pop("trial_number", None)
    study = overrides.pop("optuna_study", None)
    for key, value in overrides.items():
        OmegaConf.update(trial_cfg, key, value, force_add=True)
    OmegaConf.update(trial_cfg, "fold", fold)
    OmegaConf.update(trial_cfg, "tag", f"{base_cfg.tag}-trial{trial_number}", force_add=True)
    # Pas d'export qualitatif pour les trials : bruit inutile, cout non nul.
    OmegaConf.update(trial_cfg, "eval.qualitative.enabled", False)
    ctx = cmd_train(trial_cfg, extra={"trial_number": trial_number, "optuna_study": study,
                                      "role_in_protocol": "hpo_trial"})
    metrics = read_parquet(ctx.paths.metrics(ctx.run_id))
    return primary_value(metrics, base_cfg.eval)


def cmd_report(cfg: DictConfig) -> Path:
    """Agrege tous les runs complets et produit les tableaux du rapport (§8.3)."""
    from insectpose.reporting.report import write_report

    paths = ProjectPaths.from_config(cfg)
    master = write_master(paths)
    log.info("Agregat ecrit : %s", master)
    return write_report(paths, cfg)