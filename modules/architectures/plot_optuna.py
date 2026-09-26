#!/usr/bin/env python
"""Diagnostic figures of the Optuna studies (ADR-0031).

They do not measure the performance of a model — that is the role of `report` — but the
quality of the SEARCH: was the budget useful, which hyperparameters matter, and had the
search converged when it stopped?

This is what justifies the budget spent, and what guides the next search space. Without
these figures, one cannot tell whether 20 trials were enough or whether the search
stopped too early.

Usage:
    python plot_optuna.py                          # every study
    python plot_optuna.py --studies yolo_pooled     # filter by substring
    python plot_optuna.py --out-dir ../../results/pose/optuna --dpi 200

Output: one folder per study, under `<repo>/results/pose/optuna/<study_name>/`.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from insectpose.paths import POSE_RESULTS_DIR  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_studies(root: Path, name_filter: str | None = None) -> list[optuna.Study]:
    """Every study of the SQLite databases of `runs/optuna/`."""
    studies: list[optuna.Study] = []
    for database in sorted(glob.glob(str(root / "*.db"))):
        storage = f"sqlite:///{database}"
        for name in optuna.get_all_study_names(storage):
            if name_filter and name_filter not in name:
                continue
            studies.append(optuna.load_study(study_name=name, storage=storage))
    return studies


def completed_trials(study: optuna.Study) -> int:
    """Trials ACTUALLY completed: the pruned and interrupted ones do not count."""
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def plot_history(study: optuna.Study, out_dir: Path, dpi: int) -> Path | None:
    """Value of each trial and cumulative best value.

    A best-value curve still rising at the end signals a search cut too early; an early
    plateau signals on the contrary a sufficient budget, or a search space that is too
    narrow.
    """
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 2:
        return None

    numbers = [t.number for t in complete]
    values = [t.value for t in complete]
    maximise = study.direction == optuna.study.StudyDirection.MAXIMIZE
    cumulative, current = [], (-float("inf") if maximise else float("inf"))
    for value in values:
        current = max(current, value) if maximise else min(current, value)
        cumulative.append(current)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(numbers, values, s=28, alpha=0.75, label="trial", zorder=3)
    ax.plot(numbers, cumulative, color="crimson", lw=1.6, label="best so far", zorder=2)

    pruned = [t.number for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        # Pruned trials have no value: they are marked on the axis, because their number
        # tells whether the pruner did its job (ADR-0031).
        ax.scatter(pruned, [min(values)] * len(pruned), marker="x", s=30,
                   color="grey", label=f"pruned ({len(pruned)})", zorder=3)

    ax.set_xlabel("trial")
    ax.set_ylabel(study.metric_names[0] if study.metric_names else "objective")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Optimisation history — {completed_trials(study)} completed trials")
    return _save(fig, out_dir / "optuna_history.png", dpi)


def plot_importances(study: optuna.Study, out_dir: Path, dpi: int) -> Path | None:
    """Relative importance of each hyperparameter (fANOVA).

    A parameter with near-zero importance takes up a dimension of the budget for
    nothing: it is the first candidate for removal from the search space. Conversely, a
    dominant parameter may deserve wider bounds.
    """
    if completed_trials(study) < 4:
        return None
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:  # noqa: BLE001 - depends on the number of trials and the sampler
        print(f"    importance unavailable: {exc}")
        return None
    if not importances:
        return None

    names = list(importances)[::-1]
    values = [importances[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(names) + 2.2))
    ax.barh(names, values, color="#4c78a8", edgecolor="black", linewidth=0.4)
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("relative importance")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"Hyperparameter importance — {completed_trials(study)} trials")
    return _save(fig, out_dir / "optuna_importance.png", dpi)


def plot_contours(study: optuna.Study, out_dir: Path, dpi: int,
                  max_params: int = 4) -> Path | None:
    """Contours of the most important hyperparameter pairs.

    Shows WHERE the optimum lies in the space: stuck to a bound, the range must be
    widened; at the centre of a plateau, the search has converged.

    Limited to the `max_params` most influential parameters: beyond that, the grid
    becomes unreadable and the number of trials is no longer enough to estimate a
    surface.
    """
    if completed_trials(study) < 6:
        return None
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances)[:max_params]
    except Exception:  # noqa: BLE001
        params = list(study.best_params)[:max_params]
    if len(params) < 2:
        return None

    try:
        axes = optuna.visualization.matplotlib.plot_contour(study, params=params)
    except Exception as exc:  # noqa: BLE001
        print(f"    contours unavailable: {exc}")
        return None

    fig = axes.figure if hasattr(axes, "figure") else axes[0][0].figure
    size = max(3.0 * len(params), 8)
    fig.set_size_inches(size, size)
    fig.suptitle(f"Contour plots — {completed_trials(study)} trials", y=1.0)
    return _save(fig, out_dir / "optuna_contour.png", dpi)


def plot_parallel_coordinates(study: optuna.Study, out_dir: Path, dpi: int) -> Path | None:
    """Parallel coordinates: each line is a trial, coloured by its value.

    Complement of the contours: it shows at a glance which COMBINATIONS of values give
    good results, where the contours only take the parameters two by two.
    """
    if completed_trials(study) < 4:
        return None
    try:
        axis = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    except Exception as exc:  # noqa: BLE001
        print(f"    parallel coordinates unavailable: {exc}")
        return None
    fig = axis.figure
    fig.set_size_inches(11, 5.5)
    return _save(fig, out_dir / "optuna_parallel.png", dpi)


def write_table(study: optuna.Study, out_dir: Path) -> Path:
    """Table of the trials, for audit. Side effect: writes optuna_trials.csv."""
    table = study.trials_dataframe(attrs=("number", "value", "state", "params",
                                          "datetime_start", "duration"))
    path = out_dir / "optuna_trials.csv"
    table.to_csv(path, index=False)
    return path


def _save(fig: Any, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--studies", default=None,
                        help="Only process the studies whose name contains this string.")
    parser.add_argument("--out-dir", default=str(POSE_RESULTS_DIR / "optuna"),
                        help="Output folder (default: <repo>/results/pose/optuna).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    root = Path(args.root) / "runs" / "optuna"
    if not root.exists():
        raise SystemExit(f"{root} missing: no Optuna study to plot.")

    studies = load_studies(root, args.studies)
    if not studies:
        raise SystemExit("No study matches the filter.")

    summary = []
    for study in studies:
        complete = completed_trials(study)
        print(f"\n{study.study_name}  ({complete} completed trials)")
        if complete == 0:
            print("    no completed trial: nothing to plot.")
            continue

        out_dir = Path(args.root) / args.out_dir / study.study_name   # absolute: root ignored
        produced = [
            plot_history(study, out_dir, args.dpi),
            plot_importances(study, out_dir, args.dpi),
            plot_contours(study, out_dir, args.dpi),
            plot_parallel_coordinates(study, out_dir, args.dpi),
        ]
        write_table(study, out_dir)
        for path in [p for p in produced if p is not None]:
            print(f"    {path}")

        summary.append({
            "study": study.study_name,
            "trials": complete,
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
            "best_value": round(study.best_value, 4),
            "best_trial": study.best_trial.number,
        })

    if summary:
        print("\n=== Summary ===")
        print(pd.DataFrame(summary).to_string(index=False))
        print("\nReading guide:")
        print("  - 'best so far' still rising at the end => search cut too early;")
        print("  - a parameter with near-zero importance takes up a dimension for nothing;")
        print("  - an optimum stuck to a bound => range to widen (new ADR).")


if __name__ == "__main__":
    main()
