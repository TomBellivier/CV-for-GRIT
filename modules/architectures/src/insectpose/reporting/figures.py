"""Report figures (CONVENTIONS.md §8.4).

Every figure is produced from the ARTEFACTS: `results/master.parquet`, the coverage
reports, the predictions, the manifests and the training logs. None recomputes a
metric — otherwise two numbers of the same report could diverge.

HPO runs are excluded by `final_runs`: they are attempts, not results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")   # headless backend: the report also runs without X
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from insectpose.data.keypoints import KeypointSchema, load_schema  # noqa: E402
from insectpose.data.measurements import load_measurements, measure_all  # noqa: E402
from insectpose.evaluation.aggregate import final_runs  # noqa: E402
from insectpose.paths import KP_INFOS_PATH, ProjectPaths  # noqa: E402
from insectpose.utils.io import read_json, read_parquet  # noqa: E402
from insectpose.utils.logging import get_logger  # noqa: E402

log = get_logger("figures")

# Anatomical groups used for the colour code. The order matters: the first matching
# rule wins (specific rules before generic ones).
_GROUP_RULES: tuple[tuple[str, str], ...] = (
    ("eye", "eyes"),
    ("antenna", "antennae"),
    ("forewing", "forewings"),
    ("hindwing", "hindwings"),
    ("leg", "legs"),
    ("thorax", "thorax"),
    ("body", "abdomen"),
    ("head", "head"),
    ("neck", "head"),
)

# Colours of the annotation interface (Label Studio), so that the figures and the
# annotation tool speak the same visual language. Declared in **BGR** in kp_infos.yaml
# (single definition of the repository), as in the interface: the conversion to RGB is
# done once, in `keypoint_color`.
with KP_INFOS_PATH.open(encoding="utf-8") as _f:
    _KP_INFOS_KEYPOINTS: list[dict[str, Any]] = yaml.safe_load(_f)["keypoints"]
_KEYPOINT_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    kp["name"]: tuple(kp["color"]) for kp in _KP_INFOS_KEYPOINTS if kp.get("color")
}


def keypoint_color(name: str) -> tuple[float, float, float]:
    """Normalised RGB colour of a keypoint, following the annotation palette.

    The table is in BGR (OpenCV convention, the one of the interface): the inversion is
    done here, once and for all. An unknown point falls back to the colour of its
    anatomical group, which avoids a uniform grey if the schema evolves.
    """
    bgr = _KEYPOINT_COLORS_BGR.get(name)
    if bgr is None:
        return matplotlib.colors.to_rgb(group_color(keypoint_group(name)))
    return tuple(channel / 255 for channel in reversed(bgr))


_GROUP_COLORS = {
    "head": "#d62728", "eyes": "#7f7f7f", "antennae": "#17becf",
    "thorax": "#ff7f0e", "abdomen": "#2ca02c", "forewings": "#e377c2",
    "hindwings": "#9467bd", "legs": "#1f77b4", "other": "#8c564b",
}


def keypoint_group(name: str, with_side: bool = True) -> str:
    """Anatomical group of a keypoint, e.g. 'right hindwings' or 'head'.

    Points of the median axis have no side; the others carry it, because a left/right
    asymmetry is in itself a diagnostic information.
    """
    lowered = name.lower()
    base = next((group for token, group in _GROUP_RULES if token in lowered), "other")
    if not with_side:
        return base
    if lowered.startswith("left-"):
        return f"left {base}"
    if lowered.startswith("right-"):
        return f"right {base}"
    return base


def group_color(group: str) -> str:
    """Stable colour of an anatomical group, side included or not."""
    base = group.replace("left ", "").replace("right ", "")
    return _GROUP_COLORS.get(base, _GROUP_COLORS["other"])


def _bgr(triplet: tuple[int, int, int]) -> tuple[float, float, float]:
    """Integer BGR (0-255) -> float RGB (0-1), the format expected by matplotlib."""
    blue, green, red = triplet
    return (red / 255, green / 255, blue / 255)


KEYPOINT_COLORS: dict[str, tuple[float, float, float]] = {
    name: _bgr(value) for name, value in _KEYPOINT_COLORS_BGR.items()
}


# --- helpers ----------------------------------------------------------------
def _save(fig: Any, path: Path, dpi: int = 150) -> Path:
    """Write a figure and close it. Side effect: creates `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _select(master: pd.DataFrame, metric: str, split: str = "test",
            scope_prefix: str | None = None, scope: str | None = None) -> pd.DataFrame:
    """Quotable subset of master.parquet for a metric."""
    data = final_runs(master)
    data = data[(data["metric"] == metric) & (data["split"] == split)]
    if scope is not None:
        data = data[data["scope"] == scope]
    if scope_prefix is not None:
        data = data[data["scope"].str.startswith(scope_prefix)]
    return data


def _dataset_of(scope: str) -> str:
    """Name of the dataset carried by a scope 'dataset:x' or 'keypoint:x:name'."""
    parts = str(scope).split(":")
    return parts[1] if len(parts) > 1 else "overall"


def available_metrics(master: pd.DataFrame, split: str = "test") -> list[str]:
    """Scalar metrics available at the 'overall' scope or per dataset."""
    data = final_runs(master)
    data = data[(data["split"] == split) & (
        (data["scope"] == "overall") | data["scope"].str.startswith("dataset:"))]
    return sorted(data["metric"].unique())


# --- 1. one figure per metric: bars per dataset ------------------------------
def fig_metric_by_dataset(master: pd.DataFrame, metric: str, out_dir: Path,
                          split: str = "test", dpi: int = 150) -> Path | None:
    """Bars per dataset, grouped by approach, with the standard deviation across folds.

    The standard deviation is not decorative: without it, a gap of a few points between
    two approaches cannot be interpreted (§8.4).
    """
    data = _select(master, metric, split, scope_prefix="dataset:")
    overall = _select(master, metric, split, scope="overall")
    data = pd.concat([data, overall], ignore_index=True)
    if data.empty:
        return None
    data = data.assign(dataset=data["scope"].map(_dataset_of))

    stats = data.groupby(["dataset", "approach"])["value"].agg(["mean", "std", "size"])
    stats = stats.reset_index()
    datasets = sorted(stats["dataset"].unique(), key=lambda d: (d != "overall", d))
    approaches = sorted(stats["approach"].unique())

    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) * max(len(approaches), 1) + 3, 4.5))
    width = 0.8 / max(len(approaches), 1)
    positions = np.arange(len(datasets), dtype=float)
    for i, approach in enumerate(approaches):
        sub = stats[stats["approach"] == approach].set_index("dataset")
        means = [sub["mean"].get(d, np.nan) for d in datasets]
        errors = [sub["std"].get(d, np.nan) for d in datasets]
        ax.bar(positions + i * width, means, width, yerr=errors, capsize=3, label=approach)

    ax.set_xticks(positions + width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(metric)
    n_folds = int(stats["size"].max())
    ax.set_title(f"{metric} by dataset ({split} split, mean ± std over "
                 f"{n_folds} fold(s))")
    ax.grid(axis="y", alpha=0.3)
    if len(approaches) > 1:
        ax.legend()
    return _save(fig, out_dir / f"metric_{metric.replace('@', '').replace('/', '_')}.png", dpi)


# --- 2. confidence vs error per keypoint, one panel per dataset --------------
def fig_confidence_vs_error(master: pd.DataFrame, out_dir: Path, split: str = "test",
                            dpi: int = 150) -> Path | None:
    """Predicted confidence vs normalised error, per keypoint, anatomical colour code.

    An L-shaped cloud (high confidence and low error) means the confidence can be used
    as a filter in production; a cloud without structure means the opposite, and that
    is a first-rank information for the real use.
    """
    conf = _select(master, "kpt_conf_mean", split, scope_prefix="keypoint:")
    err = _select(master, "nme", split, scope_prefix="keypoint:")
    if conf.empty or err.empty:
        return None

    merged = (
        conf.groupby("scope")["value"].mean().rename("conf").to_frame()
        .join(err.groupby("scope")["value"].mean().rename("error"), how="inner")
        .join(err.groupby("scope")["n"].sum().rename("n"), how="inner")
        .reset_index()
    )
    merged["dataset"] = merged["scope"].map(_dataset_of)
    merged["keypoint"] = merged["scope"].map(lambda s: str(s).split(":")[-1])
    merged["group"] = merged["keypoint"].map(keypoint_group)

    datasets = sorted(merged["dataset"].unique())
    cols = 2 if len(datasets) > 1 else 1
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for ax, dataset in zip(axes.flat, datasets, strict=False):
        sub = merged[merged["dataset"] == dataset]
        for group, part in sub.groupby("group"):
            ax.scatter(part["conf"], part["error"], s=30, alpha=0.85,
                       color=group_color(str(group)), label=str(group),
                       edgecolors="black", linewidths=0.3)
        ax.set_title(dataset)
        ax.set_xlabel("mean keypoint confidence")
        ax.set_ylabel("normalised error (NME)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))
    fig.suptitle("Predicted confidence vs error, per keypoint")
    fig.tight_layout()
    return _save(fig, out_dir / "keypoint_confidence_vs_error.png", dpi)


# --- 3. training curves -----------------------------------------------------
def fig_training_curves(paths: ProjectPaths, master: pd.DataFrame, out_dir: Path,
                        dpi: int = 150) -> Path | None:
    """Training curves per epoch, one line per run (if the framework produces them).

    These curves are for diagnosis (convergence, overfitting) and NEVER for comparing
    approaches: the quotable metrics come from the evaluator (§7.1).
    """
    curves: list[pd.DataFrame] = []
    for run_id in sorted(final_runs(master)["run_id"].dropna().unique()):
        csv = paths.run_dir(str(run_id)) / "logs" / "train" / "results.csv"
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        frame.columns = [c.strip() for c in frame.columns]
        frame["run_id"] = run_id
        meta = final_runs(master)
        meta = meta[meta["run_id"] == run_id]
        frame["label"] = f"{meta['approach'].iloc[0]} fold{int(meta['fold'].iloc[0])}"
        curves.append(frame)

    if not curves:
        log.info("No training curve: the framework does not produce any.")
        return None

    data = pd.concat(curves, ignore_index=True)
    columns = [c for c in data.columns
               if c.startswith(("train/", "val/", "metrics/")) and data[c].notna().any()]
    if not columns:
        return None

    cols = 3
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    epoch_col = "epoch" if "epoch" in data.columns else None
    for ax, column in zip(axes.flat, columns, strict=False):
        for label, part in data.groupby("label"):
            x = part[epoch_col] if epoch_col else np.arange(len(part))
            ax.plot(x, part[column], lw=1.2, label=str(label))
        ax.set_title(column, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(columns):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles and len(labels) <= 12:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training curves (diagnostic only - not comparable across approaches)")
    return _save(fig, out_dir / "training_curves.png", dpi)


# --- 4. spread across folds -------------------------------------------------
def fig_fold_boxplot(master: pd.DataFrame, metric: str, out_dir: Path,
                     split: str = "test", dpi: int = 150) -> Path | None:
    """Boxplot of the metric per approach, one point per fold.

    This is the figure that goes with the paired tests: it shows whether a gap in the
    mean survives the variability between folds.
    """
    data = _select(master, metric, split, scope="overall")
    if data.empty:
        return None
    groups = data.groupby("approach")["value"]
    labels = list(groups.groups)
    values = [groups.get_group(name).to_numpy() for name in labels]

    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 4.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    for i, series in enumerate(values, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, len(series))
        ax.scatter(np.full(len(series), i) + jitter, series, s=25, alpha=0.8, zorder=3)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per fold ({split} split)")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / f"folds_{metric.replace('@', '')}.png", dpi)


# --- 5. PCK vs alpha curve ---------------------------------------------------
def fig_pck_curve(master: pd.DataFrame, out_dir: Path, split: str = "test",
                  dpi: int = 150) -> Path | None:
    """PCK as a function of the alpha threshold, one line per (approach, dataset).

    A single threshold hides the shape of the error distribution: two models with the
    same PCK@0.25 can differ clearly at tight thresholds.
    """
    data = final_runs(master)
    data = data[data["metric"].str.startswith("pck@") & (data["split"] == split)
                & (data["scope"].str.startswith("dataset:") | (data["scope"] == "overall"))]
    if data.empty:
        return None
    data = data.assign(
        alpha=data["metric"].str.extract(r"pck@([0-9.]+)_")[0].astype(float),
        dataset=data["scope"].map(_dataset_of),
    ).dropna(subset=["alpha"])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for (approach, dataset), part in data.groupby(["approach", "dataset"]):
        curve = part.groupby("alpha")["value"].mean().sort_index()
        style = "--" if dataset == "overall" else "-"
        ax.plot(curve.index, curve.to_numpy(), style, marker="o", ms=3,
                lw=2 if dataset == "overall" else 1.2,
                label=f"{approach} · {dataset}")
    ax.set_xlabel("alpha (fraction of thorax width)")
    ax.set_ylabel("PCK")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("PCK curve")
    return _save(fig, out_dir / "pck_curve.png", dpi)


# --- 6. keypoint PCK vs annotation coverage ----------------------------------
def _keypoint_pck(master: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Mean PCK per (dataset, keypoint), at the reference threshold."""
    data = final_runs(master)
    data = data[data["scope"].str.startswith("keypoint:") & (data["split"] == split)
                & data["metric"].str.startswith("pck@")]
    if data.empty:
        return data
    frame = data.groupby(["approach", "scope"])["value"].mean().reset_index()
    frame["dataset"] = frame["scope"].map(_dataset_of)
    frame["keypoint"] = frame["scope"].map(lambda s: str(s).split(":")[-1])
    return frame


def fig_pck_vs_coverage(master: pd.DataFrame, coverage: pd.DataFrame, out_dir: Path,
                        split: str = "test", dpi: int = 150) -> Path | None:
    """PCK per keypoint vs annotation rate of that keypoint in the dataset.

    Avoids the wrong conclusion "this point is poorly predicted" when it is simply
    rarely annotated (ADR-0016).
    """
    pck = _keypoint_pck(master, split)
    if pck.empty or coverage.empty:
        return None
    merged = pck.merge(coverage[["dataset", "keypoint", "rate", "n_annotated"]],
                       on=["dataset", "keypoint"], how="inner")
    if merged.empty:
        return None
    merged["group"] = merged["keypoint"].map(keypoint_group)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for group, part in merged.groupby("group"):
        ax.scatter(part["rate"], part["value"], s=28, alpha=0.85,
                   color=group_color(str(group)), label=str(group),
                   edgecolors="black", linewidths=0.3)
    ax.set_xlabel("keypoint annotation rate")
    ax.set_ylabel("keypoint PCK")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Keypoint PCK vs annotation coverage")
    return _save(fig, out_dir / "pck_vs_coverage.png", dpi)


# --- 7. keypoint PCK vs expert difficulty ------------------------------------
def fig_pck_vs_difficulty(master: pd.DataFrame, schema: KeypointSchema, out_dir: Path,
                          split: str = "test", dpi: int = 150) -> Path | None:
    """PCK per keypoint vs the difficulty declared by the expert.

    This figure validates (or not) the difficulty scale on which the OKS sigmas are
    based (ADR-0007): a weak correlation would mean that the primary metric itself is
    badly calibrated.
    """
    pck = _keypoint_pck(master, split)
    if pck.empty:
        return None
    difficulty = dict(zip(schema.names, schema.difficulty, strict=True))
    pck = pck.assign(difficulty=pck["keypoint"].map(difficulty)).dropna(subset=["difficulty"])
    if pck.empty:
        return None
    pck["group"] = pck["keypoint"].map(keypoint_group)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for group, part in pck.groupby("group"):
        jitter = np.random.default_rng(1).normal(0, 0.5, len(part))
        ax.scatter(part["difficulty"] + jitter, part["value"], s=28, alpha=0.85,
                   color=group_color(str(group)), label=str(group),
                   edgecolors="black", linewidths=0.3)
    if len(pck) > 2:
        correlation = float(np.corrcoef(pck["difficulty"], pck["value"])[0, 1])
        ax.set_title(f"Keypoint PCK vs expert difficulty (r = {correlation:.2f})")
    else:
        ax.set_title("Keypoint PCK vs expert difficulty")
    ax.set_xlabel("declared difficulty (10 = easy, 40 = very hard)")
    ax.set_ylabel("keypoint PCK")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, out_dir / "pck_vs_difficulty.png", dpi)


# --- 8. left/right symmetry of the measurements ------------------------------
def fig_symmetry_scatter(paths: ProjectPaths, master: pd.DataFrame, cfg: Any,
                         out_dir: Path, split: str = "test", dpi: int = 150
                         ) -> Path | None:
    """Predicted left measurement vs right measurement, one panel per symmetric pair.

    A consistent model aligns the points on the diagonal. The distance to the diagonal
    is read without ground truth: it is a quality check usable in production.
    """
    spec = load_measurements(Path(str(cfg.eval.measurements.file)))
    if not spec.symmetric_pairs:
        return None
    schema = load_schema(spec.keypoint_schema, paths.configs)
    index = spec.indices(schema)

    frames: list[pd.DataFrame] = []
    for run_id in sorted(final_runs(master)["run_id"].dropna().unique()):
        for file in sorted((paths.run_dir(str(run_id)) / "predictions").glob(f"{split}_*.parquet")):
            frames.append(read_parquet(file))
    if not frames:
        return None
    predictions = pd.concat(frames, ignore_index=True)
    kpts = np.stack(
        predictions["kpts_xy"].map(lambda v: np.asarray(v, float)).to_numpy()
    ).reshape(len(predictions), -1, 2)
    values = measure_all(kpts, index)

    pairs = [(a, b) for a, b in spec.symmetric_pairs if a in values and b in values]
    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    datasets = predictions["dataset"].to_numpy()

    for ax, (left, right) in zip(axes.flat, pairs, strict=False):
        x, y = values[left], values[right]
        usable = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        for dataset in sorted(set(datasets)):
            mask = usable & (datasets == dataset)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=12, alpha=0.6, label=dataset)
        if usable.any():
            limit = float(np.percentile(np.concatenate([x[usable], y[usable]]), 99.5))
            ax.plot([0, limit], [0, limit], "k--", lw=1)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            gap = np.abs(x[usable] - y[usable]) / ((x[usable] + y[usable]) / 2)
            ax.set_title(f"{left.replace('left ', '')}\nmedian gap {np.median(gap):.1%}",
                         fontsize=9)
        ax.set_xlabel("left (px)", fontsize=8)
        ax.set_ylabel("right (px)", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Symmetry of predicted measurements (left vs right)")
    return _save(fig, out_dir / "symmetry_pairs.png", dpi)


# --- 9. performance vs cost --------------------------------------------------
def _higher_is_better(metric: str) -> bool:
    """Direction of the metric. The Pareto front would be reversed otherwise."""
    from insectpose.reporting.compare import LOWER_IS_BETTER

    return metric not in LOWER_IS_BETTER


def _run_costs(paths: ProjectPaths, master: pd.DataFrame) -> pd.DataFrame:
    """Costs per run, read from the MANIFESTS (§7.2).

    `master.parquet` does not carry every cost: the approach-specific fields
    (`n_models`, `n_adapter_sets`, trainable parameter ratios) live in the manifests.
    They are read here rather than recomputed.
    """
    rows: list[dict[str, Any]] = []
    for run_id in final_runs(master)["run_id"].dropna().unique():
        manifest = paths.manifest(str(run_id))
        if not manifest.exists():
            continue
        meta = read_json(manifest)

        # Parameters actually TRAINED. `model_params` does not fit: after merging the
        # adapters, a LoRA model counts as many parameters as a fully trained model
        # (ADR-0025). The field differs depending on the approach.
        trainable = None
        for prefix in ("lora", "head", "head_only"):
            if meta.get(f"{prefix}_trainable_params") is not None:
                trainable = meta[f"{prefix}_trainable_params"]
                break
        if trainable is None:
            report = meta.get("lora_final_report") or {}
            trainable = report.get("trainable_params") or meta.get("model_params")

        # Parameters that VARY from one group to another: this is the cost of the
        # specialisation, zero for a single-model approach.
        copies = int(meta.get("n_models") or meta.get("n_adapter_sets") or 1)
        specialised = int(trainable or 0) * copies if copies > 1 else 0

        rows.append({
            "run_id": str(run_id),
            "train_time_s": meta.get("train_time_s") or meta.get("duration_s"),
            "trainable_params": trainable,
            "specialised_params": specialised,
            "n_copies": copies,
        })
    return pd.DataFrame(rows)


def pareto_front(costs: np.ndarray, performances: np.ndarray,
                 higher_is_better: bool = True) -> np.ndarray:
    """Indices of the non-dominated points, sorted by increasing cost.

    A point is dominated if another point is both cheaper AND at least as good. The
    front therefore links the only defensible choices: everything below it can be
    replaced by a strictly better option.

    Pure function, testable without matplotlib.
    """
    order = np.argsort(costs, kind="stable")
    front: list[int] = []
    best = -np.inf if higher_is_better else np.inf
    for idx in order:
        value = performances[idx]
        if not np.isfinite(value):
            continue
        improves = value > best if higher_is_better else value < best
        if improves or not front:
            front.append(int(idx))
            best = value
    return np.asarray(front, dtype=int)


def _cost_scatter(master: pd.DataFrame, costs: pd.DataFrame, metric: str, cost_column: str,
                  xlabel: str, title: str, path: Path, split: str = "test",
                  log_x: bool = False, dpi: int = 150,
                  higher_is_better: bool = True) -> Path | None:
    """Performance vs cost scatter, one labelled point per model, with the Pareto front.

    No legend: each point carries its name, and a legend would duplicate the
    information while eating the useful area.
    """
    from insectpose.evaluation.aggregate import model_label

    data = _select(master, metric, split, scope="overall")
    if data.empty or costs.empty or cost_column not in costs.columns:
        return None
    # `master.parquet` already carries some manifest fields (including train_time_s):
    # without this removal, the merge would produce `train_time_s_x`/`_y` and the
    # requested column would no longer exist. The value of `costs` is kept, as it comes
    # directly from the manifest.
    duplicates = [c for c in costs.columns if c != "run_id" and c in data.columns]
    data = data.drop(columns=duplicates).merge(costs, on="run_id", how="left")
    data = data.dropna(subset=[cost_column])
    if data.empty:
        log.info("No usable cost '%s' for '%s': figure skipped.",
                 cost_column, metric)
        return None
    data = data.copy()
    data["model"] = model_label(data)

    stats = data.groupby("model").agg(
        perf=("value", "mean"), error=("value", "std"),
        cost=(cost_column, "mean"), folds=("value", "size"),
    ).reset_index()
    if log_x:
        # A single-model approach has a ZERO specialisation cost: on a log scale it
        # would disappear, while it is precisely the reference point. It is placed at
        # the left of the axis, one decade below the smallest non-zero cost.
        positive = stats.loc[stats["cost"] > 0, "cost"]
        floor = float(positive.min()) / 10 if len(positive) else 1.0
        stats["cost"] = stats["cost"].replace(0, floor)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.get_cmap("tab10")

    cost_values = stats["cost"].to_numpy(dtype=float)
    perfs = stats["perf"].to_numpy(dtype=float)
    front = pareto_front(cost_values, perfs, higher_is_better)
    if len(front) > 1:
        ax.plot(cost_values[front], perfs[front], "--", color="grey", lw=1.2, zorder=1)

    for i, row in enumerate(stats.itertuples(index=False)):
        on_front = i in set(front.tolist())
        ax.errorbar(row.cost, row.perf,
                    yerr=row.error if np.isfinite(row.error) else None,
                    fmt="o", ms=10 if on_front else 7, capsize=4,
                    color=colors(i % 10), zorder=3,
                    markeredgecolor="black" if on_front else "none",
                    markeredgewidth=1.0 if on_front else 0)
        ax.annotate(str(row.model).split(" · ")[0], (row.cost, row.perf),
                    textcoords="offset points", xytext=(9, 5), fontsize=8)

    if log_x:
        ax.set_xscale("log")
    ax.margins(x=0.12, y=0.12)   # room for the labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    return _save(fig, path, dpi)


def fig_performance_vs_training_cost(paths: ProjectPaths, master: pd.DataFrame, out_dir: Path,
                                     metric: str = "oks_ap", split: str = "test",
                                     dpi: int = 150) -> Path | None:
    """Performance vs training time.

    This is the cost axis on which the approaches really differ. The INFERENCE time is
    almost identical everywhere — a merged LoRA model runs the same forward pass as a
    fully trained model — whereas the adaptation cost varies by a factor of 2.
    This figure answers "which one to deploy?" rather than "which one is the best?".
    """
    return _cost_scatter(
        master, _run_costs(paths, master), metric, "train_time_s",
        xlabel="training time (s)",
        title=f"{metric} vs training cost ({split} split, mean ± std over folds)",
        path=out_dir / f"cost_training_{metric.replace('@', '')}.png",
        split=split, dpi=dpi, higher_is_better=_higher_is_better(metric),
    )


def fig_performance_vs_specialisation(paths: ProjectPaths, master: pd.DataFrame, out_dir: Path,
                                      metric: str = "oks_ap", split: str = "test",
                                      dpi: int = 150) -> Path | None:
    """Performance vs number of parameters SPECIALISED per insect group.

    Isolates the cost of the specialisation: zero for a single-model approach, around
    10^4 for a per-group normalisation, 10^5 for per-group adapters, 10^7 for N full
    models. Logarithmic scale, as these orders of magnitude span three decades.
    """
    costs = _run_costs(paths, master)
    if costs.empty or (costs["specialised_params"] == 0).all():
        log.info("No per-group specialised approach: specialisation figure skipped.")
        return None
    return _cost_scatter(
        master, costs, metric, "specialised_params",
        xlabel="parameters specialised per insect group (log scale, 0 shown at left)",
        title=f"{metric} vs specialisation cost ({split} split)",
        path=out_dir / f"cost_specialisation_{metric.replace('@', '')}.png",
        split=split, log_x=True, dpi=dpi, higher_is_better=_higher_is_better(metric),
    )



# --- 10. PCK per keypoint, sorted ------------------------------------------
def fig_keypoint_pck_bars(master: pd.DataFrame, out_dir: Path, split: str = "test",
                          dataset: str | None = None, dpi: int = 150,
                          alpha: float | None = None) -> Path | None:
    """PCK of each keypoint, sorted in increasing order, annotation colours.

    The sorting puts the problematic points first: this is the figure that shows where
    to put the effort. The colours are those of the annotation interface, so that this
    figure and an annotation screenshot can be read together without any matching
    effort.

    One panel per dataset when `dataset` is None: the insect orders have neither the
    same annotated points nor the same difficulties, and mixing them would erase
    precisely what the figure must show.

    To be crossed with `pck_vs_coverage.png`: a point at the top of the list AND rarely
    annotated is not a failure of the model (ADR-0016).
    """
    data = final_runs(master)
    prefix = "pck@" if alpha is None else f"pck@{alpha:g}"
    data = data[data["scope"].str.startswith("keypoint:") & (data["split"] == split)
                & data["metric"].str.startswith(prefix)]
    if data.empty:
        log.info("No per-keypoint metric '%s*' on split '%s': figure skipped.",
                 prefix, split)
        return None

    data = data.copy()
    data["dataset"] = data["scope"].map(_dataset_of)
    data["keypoint"] = data["scope"].map(lambda s: str(s).split(":")[-1])
    if dataset is not None:
        data = data[data["dataset"] == dataset]
        if data.empty:
            return None

    stats = (data.groupby(["dataset", "keypoint"])["value"]
             .agg(pck="mean", spread="std", folds="size")
             .reset_index())

    datasets = sorted(stats["dataset"].unique())
    width = max(8.0, 0.32 * stats["keypoint"].nunique() + 3)
    fig, axes = plt.subplots(len(datasets), 1, squeeze=False,
                             figsize=(width, 4.6 * len(datasets)))

    for ax, name in zip(axes.flat, datasets, strict=True):
        sub = stats[stats["dataset"] == name].sort_values("pck").reset_index(drop=True)
        positions = np.arange(len(sub))
        colors = [keypoint_color(point) for point in sub["keypoint"]]
        # Error bars only with several folds: on a single fold, the standard deviation
        # is NaN and matplotlib would draw empty whiskers.
        errors = sub["spread"].to_numpy() if (sub["folds"] > 1).any() else None
        ax.bar(positions, sub["pck"], color=colors, edgecolor="black", linewidth=0.4,
               yerr=errors, capsize=2, error_kw={"lw": 0.8})

        ax.set_xticks(positions)
        ax.set_xticklabels(sub["keypoint"], rotation=90, fontsize=7)
        ax.set_ylabel("PCK")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
        mean = float(sub["pck"].mean())
        ax.axhline(mean, color="grey", ls="--", lw=1)
        # Label outside the plot: placed on the line, it would overlap the last bars,
        # which are precisely the highest ones.
        ax.text(1.005, mean, f"mean\n{mean:.3f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color="grey")
        ax.set_title(f"{name} ({len(sub)} keypoints)", fontsize=10)

    threshold = "" if alpha is None else f"@{alpha:g}"
    fig.suptitle(f"Keypoint PCK{threshold}, sorted ({split} split)")
    suffix = f"_{dataset}" if dataset else ""
    return _save(fig, out_dir / f"keypoint_pck_sorted{suffix}.png", dpi)


# --- 11. mean PCK over the datasets ------------------------------------------
def fig_keypoint_pck_bars_pooled(master: pd.DataFrame, out_dir: Path, split: str = "test",
                                 alpha: float | None = None, dpi: int = 150) -> Path | None:
    """PCK per keypoint, AVERAGED over the datasets, in a single panel.

    Complement of `keypoint_pck_sorted.png`, which separates the orders: this view
    answers "which points are hard in general?", the other one "which points are hard
    for this order?".

    The mean is weighted by the number of instances (`n`) and not per dataset:
    otherwise Hymenoptera (192 images) would weigh as much as Coleoptera (1026), and
    the figure would describe a corpus that does not exist. A point missing from an
    order therefore distorts nothing — it simply does not contribute to it.
    """
    data = final_runs(master)
    prefix = "pck@" if alpha is None else f"pck@{alpha:g}"
    data = data[data["scope"].str.startswith("keypoint:") & (data["split"] == split)
                & data["metric"].str.startswith(prefix)]
    if data.empty:
        log.info("No per-keypoint metric '%s*': pooled figure skipped.", prefix)
        return None

    data = data.copy()
    data["keypoint"] = data["scope"].map(lambda scope: str(scope).split(":")[-1])
    data["weight"] = data["n"].fillna(1).clip(lower=1)
    data["product"] = data["value"] * data["weight"]

    stats = data.groupby("keypoint").agg(
        product=("product", "sum"), weight=("weight", "sum"),
        spread=("value", "std"), datasets=("scope", "nunique"),
    ).reset_index()
    stats["pck"] = stats["product"] / stats["weight"]
    stats = stats.sort_values("pck").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.34 * len(stats) + 3), 5.5))
    positions = np.arange(len(stats))
    colors = [keypoint_color(name) for name in stats["keypoint"]]
    errors = stats["spread"].to_numpy()
    ax.bar(positions, stats["pck"], color=colors, edgecolor="black", linewidth=0.4,
           yerr=errors if np.isfinite(errors).any() else None,
           capsize=2, error_kw={"lw": 0.8})

    ax.set_xticks(positions)
    ax.set_xticklabels(stats["keypoint"], rotation=90, fontsize=7)
    ax.set_ylabel("PCK")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    mean = float((stats["product"].sum() / stats["weight"].sum()))
    ax.axhline(mean, color="grey", ls="--", lw=1)
    ax.text(1.005, mean, f"mean\n{mean:.3f}", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8, color="grey")

    threshold = "" if alpha is None else f"@{alpha:g}"
    ax.set_title(f"Keypoint PCK{threshold}, all datasets pooled ({split} split, "
                 "weighted by instance count)")
    return _save(fig, out_dir / "keypoint_pck_pooled.png", dpi)


# --- 12. mean skeleton, encoded error ----------------------------------------
# Schematic anatomical template, in half-body-length units (x to the right, y
# downwards). It does not claim morphological accuracy: its role is to place the 42
# points in a READABLE way — no overlap, distinct left and right sides — so that a high
# error jumps out at the concerned place of the body.
# Positions declared per point in kp_infos.yaml (`layout` key).
_BODY_LAYOUT: dict[str, tuple[float, float]] = {
    kp["name"]: tuple(kp["layout"]) for kp in _KP_INFOS_KEYPOINTS if kp.get("layout")
}


def fig_error_skeleton(master: pd.DataFrame, out_dir: Path, schema: KeypointSchema,
                       split: str = "test", dpi: int = 150) -> Path | None:
    """Schematic skeleton in which each point encodes the median error by its colour
    and its size.

    A 42-row table does not tell WHERE the difficulties are on the animal; this figure
    shows it at a glance. The edges follow the skeleton declared in the keypoint schema,
    so they follow the real anatomy.

    The error shown is the NME — error normalised by the thorax width — averaged over
    the datasets and weighted by the number of instances. A point without a measurement
    is still drawn in light grey: its absence is an information, not a hole to hide.
    """
    data = final_runs(master)
    data = data[data["scope"].str.startswith("keypoint:") & (data["split"] == split)
                & (data["metric"] == "nme")]
    if data.empty:
        log.info("Per-keypoint 'nme' metric missing: error skeleton skipped.")
        return None

    data = data.copy()
    data["keypoint"] = data["scope"].map(lambda scope: str(scope).split(":")[-1])
    data["weight"] = data["n"].fillna(1).clip(lower=1)
    stats = data.groupby("keypoint").apply(
        lambda g: np.average(g["value"], weights=g["weight"]), include_groups=False
    ).rename("error").reset_index()
    errors = dict(zip(stats["keypoint"], stats["error"], strict=True))

    missing = [n for n in schema.names if n not in _BODY_LAYOUT]
    if missing:
        log.warning("%d keypoint(s) missing from the drawing template: %s. The skeleton "
                    "will be incomplete.", len(missing), missing[:5])

    positions = {n: _BODY_LAYOUT[n] for n in schema.names if n in _BODY_LAYOUT}
    if not positions:
        return None

    values = np.array([errors.get(n, np.nan) for n in positions])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    # Bounds at the 5th/95th percentile: a single outlier would crush the whole colour
    # scale and make the figure unreadable.
    vmin, vmax = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
    if vmin >= vmax:
        vmin, vmax = float(finite.min()), float(finite.max() + 1e-9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn_r")   # green = accurate, red = inaccurate

    fig, ax = plt.subplots(figsize=(9, 8))
    index = {name: i for i, name in enumerate(schema.names)}
    for a, b in schema.skeleton:
        names = (schema.names[a], schema.names[b])
        if all(n in positions for n in names):
            xs = [positions[n][0] for n in names]
            ys = [positions[n][1] for n in names]
            ax.plot(xs, ys, color="#bbbbbb", lw=1.2, zorder=1)

    for name, (x, y) in positions.items():
        value = errors.get(name, np.nan)
        if np.isfinite(value):
            # Size AND colour carry the same information: the redundancy keeps the
            # figure readable in greyscale as well as for a colour-blind reader.
            size = 90 + 420 * float(np.clip(norm(value), 0, 1))
            color, edge = cmap(norm(value)), "black"
        else:
            size, color, edge = 60, "#eeeeee", "#999999"
        ax.scatter(x, y, s=size, color=color, edgecolors=edge, linewidths=0.8,
                   zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=5.5, color="#444444")

    ax.set_aspect("equal")
    ax.invert_yaxis()          # y downwards in the template: the head must stay at the top
    ax.axis("off")
    ax.margins(0.12)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 shrink=0.7, label="median normalised error (NME)")
    ax.set_title(f"Error map on a schematic body ({split} split)\n"
                 "point size and colour encode the error; grey = not measured",
                 fontsize=11)
    return _save(fig, out_dir / "error_skeleton.png", dpi)


# --- entry point -------------------------------------------------------------
def write_figures(paths: ProjectPaths, cfg: Any, master: pd.DataFrame,
                  out_dir: Path | None = None) -> list[Path]:
    """Produce the figures of a set of runs. Side effect: writes into `out_dir`."""
    out_dir = out_dir or (paths.results / "figures")
    dpi = int(cfg.report.dpi)
    split = str(cfg.report.split)
    written: list[Path] = []

    for metric in available_metrics(master, split):
        written.append(fig_metric_by_dataset(master, metric, out_dir, split, dpi))
        written.append(fig_fold_boxplot(master, metric, out_dir, split, dpi))

    written.append(fig_confidence_vs_error(master, out_dir, split, dpi))
    written.append(fig_pck_curve(master, out_dir, split, dpi))
    reference_alpha = float(cfg.eval.pck.reference_alpha)
    # NAMED arguments: the 4th positional one is `dataset`, and passing the alpha there
    # filtered on a dataset named "0.25" — the figure came out empty without any message.
    written.append(fig_keypoint_pck_bars(
        master, out_dir, split=split, dpi=dpi, alpha=reference_alpha))
    written.append(fig_keypoint_pck_bars_pooled(
        master, out_dir, split=split, dpi=dpi, alpha=reference_alpha))
    written.append(fig_training_curves(paths, master, out_dir, dpi))

    coverage_file = paths.processed / "coverage_keypoints.parquet"
    if coverage_file.exists():
        written.append(fig_pck_vs_coverage(master, read_parquet(coverage_file), out_dir,
                                           split, dpi))

    schema_name = cfg.data.get("keypoint_schema")
    if schema_name:
        schema = load_schema(str(schema_name), paths.configs)
        written.append(fig_pck_vs_difficulty(master, schema, out_dir, split, dpi))
        written.append(fig_error_skeleton(master, out_dir, schema, split, dpi))

    if bool(cfg.eval.measurements.enabled):
        written.append(fig_symmetry_scatter(paths, master, cfg, out_dir, split, dpi))

    # Cost: TWO axes, because none is enough on its own. The training time separates
    # the approaches; the number of specialised parameters isolates the cost of the
    # specialisation.
    primary = str(cfg.eval.primary_metric)
    written.append(fig_performance_vs_training_cost(paths, master, out_dir, primary,
                                                    split, dpi))
    written.append(fig_performance_vs_specialisation(paths, master, out_dir, primary,
                                                     split, dpi))

    produced = [p for p in written if p is not None]
    log.info("%d figure(s) written to %s", len(produced), out_dir)
    return produced


def write_per_run_figures(paths: ProjectPaths, cfg: Any, master: pd.DataFrame) -> list[Path]:
    """One figure folder per run, under results/runs/<run_id>/.

    The global report compares the models with each other; these folders allow a run to
    be examined on its own without the next one overwriting it.
    Side effect: writes results/runs/<run_id>/.
    """
    written: list[Path] = []
    for run_id in sorted(final_runs(master)["run_id"].dropna().unique()):
        subset = master[master["run_id"] == run_id]
        written.extend(
            write_figures(paths, cfg, subset, paths.results / "runs" / str(run_id))
        )
    log.info("Per-run figures written for %d run(s).",
             final_runs(master)["run_id"].nunique())
    return written
