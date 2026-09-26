"""Tests of the report figures and of the multi-model comparison (§8.4).

The aesthetics are not checked: we check that each figure is produced, not empty, and
that it rests on the artefacts (hence on metrics already computed).
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
    """Two trained folds, aggregated: the minimum to get a spread."""
    from insectpose.data.coverage import write_coverage
    from insectpose.data.datamodule import load_annotations
    from insectpose.data.keypoints import load_schemas
    from insectpose.data.measurements import load_measurements

    pipeline.cmd_split(cfg)
    for fold in range(2):
        OmegaConf.update(cfg, "fold", fold)
        pipeline.cmd_train(cfg)

    # The coverage is a `prepare` artefact; the fixture bypasses the adapter.
    annotations = load_annotations([str(d) for d in cfg.data.datasets], project)
    write_coverage(
        annotations, load_schemas([str(cfg.data.keypoint_schema)], project.configs),
        project.processed, spec=load_measurements(Path(str(cfg.eval.measurements.file))),
    )
    master = read_parquet(write_master(project))
    return cfg, project, master


# --- anatomical groups -------------------------------------------------------
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


# --- report figures ----------------------------------------------------------
def test_report_writes_all_figures(reported) -> None:
    cfg, project, master = reported
    produced = figures.write_figures(project, cfg, master)
    names = {p.name for p in produced}

    assert any(n.startswith("metric_") for n in names)      # one figure per metric
    assert any(n.startswith("folds_") for n in names)        # boxplot per fold
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
    assert figures.fig_metric_by_dataset(master, "nonexistent_metric",
                                         project.results / "figures") is None


# --- multi-model comparison --------------------------------------------------
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

    with pytest.raises(ValueError, match="No run"):
        write_comparison(project, CompareFilter(approaches=("missing_approach",)))


def test_comparison_excludes_hpo_trials(reported) -> None:
    """An HPO trial is not a candidate model: it must not appear."""
    _, project, master = reported
    trials = master.assign(role_in_protocol="hpo_trial",
                           run_id=master["run_id"].astype(str) + "-trial")
    selection = CompareFilter()
    kept = selection.apply(pd.concat([master, trials], ignore_index=True))
    assert set(kept["role_in_protocol"]) == {"final"}
    assert selection.excluded["hpo_trials"] > 0


# ===========================================================================
# Model identity and heatmap readability
# ===========================================================================
def test_two_variants_sharing_a_tag_are_not_averaged_together() -> None:
    """Regression: two starting weights under the same tag were taken for two folds."""
    from insectpose.evaluation.aggregate import model_label, summary_table

    master = pd.DataFrame({
        "approach": ["yolo_pooled"] * 4, "tag": ["pooled"] * 4,
        "variant_hash": ["aaaaaaaa", "aaaaaaaa", "bbbbbbbb", "bbbbbbbb"],
        "fold": [0, 1, 0, 1], "metric": ["oks_ap"] * 4, "scope": ["overall"] * 4,
        "split": ["test"] * 4, "value": [0.8, 0.82, 0.5, 0.52], "n": [10] * 4,
        "role_in_protocol": ["final"] * 4, "run_id": list("abcd"),
    })
    labels = model_label(master)
    assert labels.nunique() == 2
    assert all("aaaaaa" in v or "bbbbbb" in v for v in labels)

    summary = summary_table(master, "oks_ap")
    assert len(summary) == 2
    assert set(summary["n_folds"]) == {2}
    assert summary["mean"].max() > 0.79 and summary["mean"].min() < 0.53


def test_single_variant_keeps_a_short_label() -> None:
    """The hash only appears on a collision: otherwise the labels would be unreadable."""
    from insectpose.evaluation.aggregate import model_label

    master = pd.DataFrame({
        "approach": ["lora", "lora"], "tag": ["d", "d"],
        "variant_hash": ["aaaaaaaa", "aaaaaaaa"],
    })
    assert list(model_label(master).unique()) == ["lora · d"]


def test_outer_and_inner_folds_are_distinguished(reported) -> None:
    _, _, master = reported
    assert "outer_fold" in master.columns
    assert "inner_fold" in master.columns
    final = master[master["role_in_protocol"] == "final"]
    assert (final["outer_fold"] == final["fold"]).all()
    assert final["inner_fold"].isna().all()


@pytest.mark.parametrize(
    ("rgba", "expected"),
    [((0.99, 0.91, 0.14, 1.0), "black"),    # bright yellow (top of viridis)
     ((0.27, 0.00, 0.33, 1.0), "white"),    # dark purple (bottom of viridis)
     ((0.13, 0.57, 0.55, 1.0), "black")],   # median teal: black contrasts better
)
def test_heatmap_text_colour_follows_cell_luminance(rgba, expected) -> None:
    """A threshold based on the value is wrong at both ends of the palette."""
    from insectpose.reporting.compare import text_color

    assert text_color(rgba) == expected


def test_text_colour_threshold_is_the_wcag_crossover() -> None:
    """The chosen threshold (0.179) is the point where black and white contrast equally."""
    from insectpose.reporting.compare import text_color

    def contrast(luminance: float, other: float) -> float:
        lo, hi = sorted((luminance, other))
        return (hi + 0.05) / (lo + 0.05)

    for grey in (0.02, 0.10, 0.30, 0.60, 0.95):
        rgba = (grey, grey, grey, 1.0)
        linear = grey / 12.92 if grey <= 0.04045 else ((grey + 0.055) / 1.055) ** 2.4
        best = "black" if contrast(linear, 0.0) > contrast(linear, 1.0) else "white"
        assert text_color(rgba) == best


def test_per_run_figures_go_to_separate_folders(reported) -> None:
    """A run must not overwrite the figures of the previous one."""
    from insectpose.reporting.figures import write_per_run_figures

    cfg, project, master = reported
    produced = write_per_run_figures(project, cfg, master)
    assert produced
    run_ids = set(master["run_id"].dropna().unique())
    folders = {p.parent.name for p in produced}
    assert folders <= run_ids
    assert (project.results / "runs").exists()