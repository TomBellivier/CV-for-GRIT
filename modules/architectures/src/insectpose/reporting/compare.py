"""Comparison of every trained model (CONVENTIONS.md §8.4).

Reads `results/master.parquet`, filters the wanted runs and produces heatmaps. Like the
rest of the reporting, this module recomputes no metric and ignores the HPO trials:
they are attempts, not results.

The filters are on manifest fields (approach, tag, data scope, split), never on the
values: filtering on a result to highlight it would be a posteriori selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from insectpose.evaluation.aggregate import final_runs  # noqa: E402
from insectpose.paths import ProjectPaths  # noqa: E402
from insectpose.reporting.figures import _dataset_of, _save  # noqa: E402
from insectpose.utils.io import read_parquet  # noqa: E402
from insectpose.utils.logging import get_logger  # noqa: E402

log = get_logger("compare")

# Metrics where a LOW value is better: the palette must be reversed, otherwise the
# visual reading of the table says the opposite of the numbers.
LOWER_IS_BETTER = (
    "nme", "nme_matched_only", "measurement_mape_median", "measurement_mape_worst",
    "symmetry_gap_median", "symmetry_gap_p90", "latency_ms_per_instance",
    "latency_ms_p95", "pck_normalizer_fallback_rate",
)


@dataclass
class CompareFilter:
    """Run selection filters. None is on a metric value."""

    approaches: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    data_scopes: tuple[str, ...] = ()
    split_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    split: str = "test"
    label_by: tuple[str, ...] = ("approach", "tag")
    metrics: tuple[str, ...] = ()
    # Keypoint patterns to exclude from the per-point heatmaps. Essential to compare fairly
    # an approach trained without some points (ADR-0027): its `overall` scopes are
    # mechanically worse, only the kept points are comparable.
    exclude_keypoints: tuple[str, ...] = ()
    excluded: dict[str, int] = field(default_factory=dict)

    def apply(self, master: pd.DataFrame) -> pd.DataFrame:
        """Apply the filters and log what was left out."""
        data = final_runs(master)
        self.excluded["hpo_trials"] = int(
            master["run_id"].nunique() - data["run_id"].nunique()
        )
        data = data[data["split"] == self.split]
        for column, wanted in (
            ("approach", self.approaches), ("tag", self.tags),
            ("data_scope", self.data_scopes), ("split_id", self.split_ids),
            ("run_id", self.run_ids),
        ):
            if wanted and column in data.columns:
                before = data["run_id"].nunique()
                data = data[data[column].astype(str).isin(wanted)]
                self.excluded[column] = before - data["run_id"].nunique()
        if self.metrics:
            data = data[data["metric"].isin(self.metrics)]
        return data

    def label(self, frame: pd.DataFrame) -> pd.Series:
        """Row label of the heatmaps.

        If two distinct VARIANTS share the same label — two starting weights under the
        same tag, for instance — the variant hash is added. Without it, two different
        models would be averaged together as if they were two folds.
        """
        columns = [c for c in self.label_by if c in frame.columns]
        if not columns:
            return frame["run_id"].astype(str)
        base = frame[columns].astype(str).agg(" · ".join, axis=1)
        if "variant_hash" not in frame.columns:
            return base
        variants = frame.groupby(base.rename("base"))["variant_hash"].transform("nunique")
        return base.where(variants <= 1,
                          base + " · " + frame["variant_hash"].astype(str).str[:6])


def load_master(paths: ProjectPaths) -> pd.DataFrame:
    """Load the aggregate. Fails if `report` never ran."""
    path = paths.master_results()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run first: python -m insectpose.cli report"
        )
    return read_parquet(path)


def text_color(rgba: tuple[float, ...]) -> str:
    """Black or white, depending on the ACTUAL luminance of the cell.

    A palette such as viridis goes from dark purple to bright yellow: a threshold based on
    the numeric value is wrong at both ends. The perceptual luminance of the colour
    actually drawn (sRGB coefficients) is therefore computed, after gamma linearisation.
    """
    channels = []
    for component in rgba[:3]:
        c = float(component)
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    # 0.179 is the point where the WCAG contrast of black equals that of white:
    # (L + 0.05)^2 = 0.05 x 1.05. Above it, black is more readable.
    return "black" if luminance > 0.179 else "white"


def _heatmap(matrix: pd.DataFrame, title: str, path: Path, lower_is_better: bool,
             dpi: int = 150, value_format: str = "{:.3f}") -> Path:
    """Draw an annotated heatmap. Side effect: writes `path`."""
    height = 0.5 * len(matrix) + 2.5
    width = 1.1 * len(matrix.columns) + 4
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("viridis_r" if lower_is_better else "viridis")
    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    vmin = float(finite_values.min()) if finite_values.size else 0.0
    vmax = float(finite_values.max()) if finite_values.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=14)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=14)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "-", ha="center", va="center", fontsize=14, color="grey")
                continue
            ax.text(j, i, value_format.format(value), ha="center", va="center",
                    fontsize=14, color=text_color(cmap(norm(value))))

    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title + ("  (lower is better)" if lower_is_better else ""), fontsize=14)
    fig.tight_layout()
    return _save(fig, path, dpi)


def heatmap_metric_by_dataset(data: pd.DataFrame, metric: str, selection: CompareFilter,
                              out_dir: Path, dpi: int = 150) -> Path | None:
    """Models x datasets heatmap for a metric (mean across folds)."""
    sub = data[data["metric"] == metric].copy()
    sub = sub[(sub["scope"] == "overall") | sub["scope"].str.startswith("dataset:")]
    if sub.empty:
        return None
    sub["dataset"] = sub["scope"].map(_dataset_of)
    sub["label"] = selection.label(sub)
    matrix = sub.pivot_table(index="label", columns="dataset", values="value", aggfunc="mean")
    columns = [c for c in matrix.columns if c != "overall"] + (
        ["overall"] if "overall" in matrix.columns else [])
    matrix = matrix[columns]
    return _heatmap(matrix, f"{metric} by model and dataset",
                    out_dir / f"heatmap_{metric.replace('@', '')}.png",
                    metric in LOWER_IS_BETTER, dpi)


def heatmap_keypoint_by_model(data: pd.DataFrame, selection: CompareFilter, out_dir: Path,
                              metric_prefix: str = "pck@", dataset: str | None = None,
                              dpi: int = 150) -> Path | None:
    """Keypoints x models heatmap: shows whether an approach wins everywhere or only on
    the easy points."""
    sub = data[data["scope"].str.startswith("keypoint:")
               & data["metric"].str.startswith(metric_prefix)].copy()
    if sub.empty:
        return None
    sub["dataset"] = sub["scope"].map(_dataset_of)
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
        if sub.empty:
            return None
    sub["keypoint"] = sub["scope"].map(lambda s: str(s).split(":")[-1])
    if selection.exclude_keypoints:
        pattern = "|".join(str(p) for p in selection.exclude_keypoints)
        sub = sub[~sub["keypoint"].str.contains(pattern, case=False, regex=True)]
        if sub.empty:
            return None
    sub["label"] = selection.label(sub)
    matrix = sub.pivot_table(index="keypoint", columns="label", values="value", aggfunc="mean")
    if selection.exclude_keypoints:
        # The mean over the KEPT points is the comparison number sought.
        matrix.loc["MEAN (retained)"] = matrix.mean(axis=0)
    suffix = f"_{dataset}" if dataset else ""
    title = "PCK by keypoint and model" + (f" - {dataset}" if dataset else "")
    return _heatmap(matrix, title, out_dir / f"heatmap_keypoints{suffix}.png", False, dpi)


def write_comparison(paths: ProjectPaths, selection: CompareFilter, out_dir: Path | None = None,
                     dpi: int = 150, per_dataset_keypoints: bool = True) -> list[Path]:
    """Produce every comparison heatmap. Side effect: writes `out_dir`."""
    master = load_master(paths)
    data = selection.apply(master)
    if data.empty:
        raise ValueError(
            "No run matches the filters. Available runs: "
            f"{sorted(final_runs(master)['approach'].dropna().unique())}"
        )
    out_dir = out_dir or (paths.results / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = sorted(data[
        (data["scope"] == "overall") | data["scope"].str.startswith("dataset:")
    ]["metric"].unique())
    written: list[Path] = []
    for metric in metrics:
        written.append(heatmap_metric_by_dataset(data, metric, selection, out_dir, dpi))
    written.append(heatmap_keypoint_by_model(data, selection, out_dir, dpi=dpi))
    if per_dataset_keypoints:
        datasets = sorted({_dataset_of(s) for s in data["scope"] if str(s).startswith("keypoint:")})
        for dataset in datasets:
            written.append(heatmap_keypoint_by_model(data, selection, out_dir,
                                                     dataset=dataset, dpi=dpi))

    produced = [p for p in written if p is not None]
    table = data.pivot_table(index=["approach", "tag", "fold"], columns=["scope", "metric"],
                             values="value", aggfunc="mean")
    table.to_parquet(out_dir / "comparison.parquet")
    log.info("%d heatmap(s) and 1 table written to %s | runs compared: %d | left out: %s",
             len(produced), out_dir, data["run_id"].nunique(),
             {k: v for k, v in selection.excluded.items() if v})
    return produced