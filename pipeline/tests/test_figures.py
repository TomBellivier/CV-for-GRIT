"""Tests des figures du rapport et de la comparaison multi-modeles (§8.3).

On ne verifie pas l'esthetique : on verifie que chaque figure est produite, non vide,
et qu'elle repose bien sur les artefacts (donc sur des metriques deja calculees).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.evaluation.aggregate import write_master
from insectpose.reporting import figures
from insectpose.reporting.compare import CompareFilter, write_comparison
from insectpose.utils.io import read_parquet


@pytest.fixture()
def reported(cfg, project):
    """Deux folds entraines, agreges : le minimum pour avoir une dispersion."""
    from insectpose.data.coverage import write_coverage
    from insectpose.data.datamodule import load_annotations
    from insectpose.data.keypoints import load_schemas
    from insectpose.data.measurements import load_measurements

    pipeline.cmd_split(cfg)
    for fold in range(2):
        OmegaConf.update(cfg, "fold", fold)
        pipeline.cmd_train(cfg)

    # La couverture est un artefact de `prepare` ; la fixture court-circuite l'adaptateur.
    annotations = load_annotations([str(d) for d in cfg.data.datasets], project)
    write_coverage(
        annotations, load_schemas([str(cfg.data.keypoint_schema)], project.configs),
        project.processed, spec=load_measurements(Path(str(cfg.eval.measurements.file))),
    )
    master = read_parquet(write_master(project))
    return cfg, project, master


# --- groupes anatomiques -----------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("head-top", "head"),
        ("neck", "head"),
        ("left-eye", "left eyes"),
        ("right-hindwing-tip", "right hindwings"),
        ("left-forewing-base", "left forewings"),
        ("left-leg-2", "left legs"),
        ("thorax-left", "thorax"),
        ("body-tip", "abdomen"),
    ],
)
def test_keypoint_group_mapping(name: str, expected: str) -> None:
    assert figures.keypoint_group(name) == expected


def test_group_color_ignores_side() -> None:
    assert figures.group_color("left legs") == figures.group_color("right legs")


# --- figures du rapport ------------------------------------------------------
def test_report_writes_all_figures(reported) -> None:
    cfg, project, master = reported
    produced = figures.write_figures(project, cfg, master)
    names = {p.name for p in produced}

    assert any(n.startswith("metric_") for n in names)      # une figure par metrique
    assert any(n.startswith("folds_") for n in names)        # boxplot par fold
    assert "keypoint_confidence_vs_error.png" in names
    assert "pck_curve.png" in names
    assert "pck_vs_coverage.png" in names
    assert "pck_vs_difficulty.png" in names
    assert "symmetry_pairs.png" in names
    assert all(p.stat().st_size > 1000 for p in produced)


def test_figures_are_produced_through_the_cli(reported) -> None:
    cfg, project, _ = reported
    pipeline.cmd_report(cfg)
    assert (project.results / "figures" / "pck_curve.png").exists()


def test_figures_can_be_disabled(reported) -> None:
    cfg, project, _ = reported
    OmegaConf.update(cfg, "report.figures", False)
    pipeline.cmd_report(cfg)
    assert not (project.results / "figures").exists()


def test_metric_figure_returns_none_when_metric_absent(reported) -> None:
    _, project, master = reported
    assert figures.fig_metric_by_dataset(master, "metrique_inexistante",
                                         project.results / "figures") is None


# --- comparaison multi-modeles ----------------------------------------------
def test_comparison_writes_heatmaps(reported) -> None:
    _, project, _ = reported
    produced = write_comparison(project, CompareFilter())
    names = {p.name for p in produced}
    assert any(n.startswith("heatmap_") for n in names)
    assert any(n.startswith("heatmap_keypoints") for n in names)
    assert (project.results / "comparison" / "comparison.parquet").exists()


def test_comparison_filters_are_applied(reported) -> None:
    _, project, _ = reported
    selection = CompareFilter(approaches=("mean_pose",))
    assert write_comparison(project, selection)

    with pytest.raises(ValueError, match="Aucun run"):
        write_comparison(project, CompareFilter(approaches=("approche_absente",)))


def test_comparison_excludes_hpo_trials(reported) -> None:
    """Un trial d'HPO n'est pas un modele candidat : il ne doit pas apparaitre."""
    _, project, master = reported
    trials = master.assign(role_in_protocol="hpo_trial",
                           run_id=master["run_id"].astype(str) + "-trial")
    selection = CompareFilter()
    kept = selection.apply(pd.concat([master, trials], ignore_index=True))
    assert set(kept["role_in_protocol"]) == {"final"}
    assert selection.excluded["hpo_trials"] > 0