"""
Aggregate and compare YOLO-pose results produced by train_eval_pose.py.

Loads every result_*.xlsx workbook in a directory and produces:
  - a combined comparison workbook (all runs, plus best configuration per
    group across all model types),
  - one bar chart per run configuration comparing the insect groups,
  - one learning-curve figure per run configuration (losses and pose mAP as a
    function of the training epoch, one line per group),
  - cross-configuration bar charts comparing model types within each group,
  - configuration-by-dataset heatmaps for pose mAP50 and PCK.

Example
-------
python compare_pose_results.py --results-dir pose_results --out-dir comparison
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm

# Scalar metrics compared between groups. Lower is better only for the errors.
COMPARISON_METRICS = ["pose_map", "pose_map50", "mpjpe_px", "pck_0.1",
                      "mean_kpt_conf"]
LOWER_IS_BETTER = {"mpjpe_px", "nmpjpe"}

# Metrics rendered as configuration-by-dataset heatmaps (higher is better).
HEATMAP_METRICS = ["pose_map50", "pck_0.1"]

# Substrings used to locate the relevant columns inside learning_curves.
LOSS_HINTS = ["train/pose_loss", "train/kobj_loss", "val/pose_loss"]
MAP_HINT = "mAP50-95(P)"

# ---------------------------------------------------------------------------
# Example-inference visualisation
# ---------------------------------------------------------------------------
# Order in which the YOLO-pose models output their keypoints. Must match the
# order used when the models were trained (identical to the list used in
# plot_keypoint_conf_vs_error).
KEYPOINT_ORDER = [
    "head-top", "head-left", "head-right", "left-eye", "right-eye", "neck",
    "thorax-left", "thorax-right", "thorax-bottom", "body-left", "body-right",
    "body-tip", "left-antenna-0", "left-antenna-1", "left-antenna-2",
    "right-antenna-0", "right-antenna-1", "right-antenna-2",
    "left-forewing-base", "left-forewing-tip", "left-forewing-front",
    "left-forewing-rear", "right-forewing-base", "right-forewing-tip",
    "right-forewing-front", "right-forewing-rear", "left-hindwing-base",
    "left-hindwing-tip", "left-hindwing-front", "left-hindwing-rear",
    "right-hindwing-base", "right-hindwing-tip", "right-hindwing-front",
    "right-hindwing-rear", "left-leg-0", "left-leg-1", "left-leg-2",
    "left-leg-3", "right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3",
]

# Per-keypoint colour code (matplotlib colour names / shorthands).
KEYPOINT_COLORS = {
    "head-top": "red", "head-left": "red", "head-right": "red",
    "left-eye": "grey", "right-eye": "grey", "neck": "red",
    "thorax-left": "orange", "thorax-right": "orange", "thorax-bottom": "orange",
    "body-left": "green", "body-right": "green", "body-tip": "green",
    "left-antenna-0": "cyan", "left-antenna-1": "cyan", "left-antenna-2": "cyan",
    "right-antenna-0": "cyan", "right-antenna-1": "cyan",
    "right-antenna-2": "cyan", "left-forewing-base": "m",
    "left-forewing-tip": "m", "left-forewing-front": "m",
    "left-forewing-rear": "m", "right-forewing-base": "m",
    "right-forewing-tip": "m", "right-forewing-front": "m",
    "right-forewing-rear": "m", "left-hindwing-base": "purple",
    "left-hindwing-tip": "purple", "left-hindwing-front": "purple",
    "left-hindwing-rear": "purple", "right-hindwing-base": "purple",
    "right-hindwing-tip": "purple", "right-hindwing-front": "purple",
    "right-hindwing-rear": "purple", "left-leg-0": "blue", "left-leg-1": "blue",
    "left-leg-2": "blue", "left-leg-3": "blue", "right-leg-0": "blue",
    "right-leg-1": "blue", "right-leg-2": "blue", "right-leg-3": "blue",
}

# Anatomical skeleton: each edge links two keypoints. The drawn line takes the
# colour of its *start* keypoint. Adjust freely if a different topology is
# wanted; only the names above are valid endpoints.
KEYPOINT_SKELETON = [
    # head / eyes
    ("head-top", "head-left"), ("head-top", "head-right"),
    ("head-left", "left-eye"), ("head-right", "right-eye"),
    ("head-top", "neck"), ("head-left", "neck"), ("head-right", "neck"),
    # antennae
    ("head-left", "left-antenna-0"), ("left-antenna-0", "left-antenna-1"),
    ("left-antenna-1", "left-antenna-2"),
    ("head-right", "right-antenna-0"), ("right-antenna-0", "right-antenna-1"),
    ("right-antenna-1", "right-antenna-2"),
    # neck -> thorax
    ("neck", "thorax-left"), ("neck", "thorax-right"),
    ("thorax-left", "thorax-right"), ("thorax-left", "thorax-bottom"),
    ("thorax-right", "thorax-bottom"),
    # thorax -> abdomen
    ("thorax-bottom", "body-left"), ("thorax-bottom", "body-right"),
    ("body-left", "body-right"), ("body-left", "body-tip"),
    ("body-right", "body-tip"),
    # left forewing (closed quad)
    ("thorax-left", "left-forewing-base"),
    ("left-forewing-base", "left-forewing-front"),
    ("left-forewing-front", "left-forewing-tip"),
    ("left-forewing-tip", "left-forewing-rear"),
    ("left-forewing-rear", "left-forewing-base"),
    # right forewing
    ("thorax-right", "right-forewing-base"),
    ("right-forewing-base", "right-forewing-front"),
    ("right-forewing-front", "right-forewing-tip"),
    ("right-forewing-tip", "right-forewing-rear"),
    ("right-forewing-rear", "right-forewing-base"),
    # left hindwing
    ("thorax-left", "left-hindwing-base"),
    ("left-hindwing-base", "left-hindwing-front"),
    ("left-hindwing-front", "left-hindwing-tip"),
    ("left-hindwing-tip", "left-hindwing-rear"),
    ("left-hindwing-rear", "left-hindwing-base"),
    # right hindwing
    ("thorax-right", "right-hindwing-base"),
    ("right-hindwing-base", "right-hindwing-front"),
    ("right-hindwing-front", "right-hindwing-tip"),
    ("right-hindwing-tip", "right-hindwing-rear"),
    ("right-hindwing-rear", "right-hindwing-base"),
    # left legs (chain)
    ("thorax-left", "left-leg-0"), ("left-leg-0", "left-leg-1"),
    ("left-leg-1", "left-leg-2"), ("left-leg-2", "left-leg-3"),
    # right legs (chain)
    ("thorax-right", "right-leg-0"), ("right-leg-0", "right-leg-1"),
    ("right-leg-1", "right-leg-2"), ("right-leg-2", "right-leg-3"),
]

# Point / line sizes are scaled by the image diagonal so they stay
# proportional to the image size (fractions of the diagonal).
INFERENCE_POINT_SCALE = 0.010   # marker radius as a fraction of the diagonal
INFERENCE_LINE_SCALE = 0.0020   # line width as a fraction of the diagonal


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="pose_results",
                        help="Directory containing result_*.xlsx workbooks.")
    parser.add_argument("--out-dir", default="comparison",
                        help="Directory for the combined workbook and figures.")
    return parser.parse_args()


def metadata_to_dict(frame):
    """Convert a two-column (field, value) metadata sheet into a dict."""
    return dict(zip(frame["field"].astype(str), frame["value"]))


def load_runs(results_dir):
    """Load every workbook into a list of per-run dictionaries."""
    runs = []
    for path in sorted(Path(results_dir).glob("results_*.xlsx")):
        sheets = pd.read_excel(path, sheet_name=None)
        metadata = metadata_to_dict(sheets["metadata"])
        run_tag = metadata.get("run_tag", path.stem)

        summary = sheets["summary"].copy()
        summary["run_tag"] = run_tag
        summary["model"] = metadata.get("model", "unknown")

        per_keypoint = sheets.get("per_keypoint", pd.DataFrame()).copy()
        if not per_keypoint.empty:
            per_keypoint["run_tag"] = run_tag

        runs.append({
            "run_tag": run_tag,
            "metadata": metadata,
            "summary": summary,
            "per_keypoint": per_keypoint,
            "curves": sheets.get("learning_curves", pd.DataFrame()),
        })
    return runs


def plot_group_comparison(summary, run_tag, out_dir):
    """Bar chart comparing groups across the comparison metrics for one run."""
    metrics = [m for m in COMPARISON_METRICS if m in summary.columns]
    if not metrics or summary.empty:
        return

    groups = summary["group"].tolist()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.bar(groups, summary[metric].values)
        ax.set_title(metric)
        ax.set_xlabel("group")
        ax.tick_params(axis="x", rotation=45)
        if metric in LOWER_IS_BETTER:
            ax.set_ylabel("value (lower is better)")
        else:
            ax.set_ylabel("value (higher is better)")

    fig.suptitle(f"Group comparison - {run_tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_tag}__group_comparison.png", dpi=150)
    plt.close(fig)


def plot_learning_curves(curves, run_tag, out_dir):
    """Plot per-epoch losses and pose mAP on a 2x2 grid, one line per group."""
    if curves.empty or "epoch" not in curves.columns:
        return
 
    loss_cols = [c for c in curves.columns if c in LOSS_HINTS]
    map_col = next((c for c in curves.columns if MAP_HINT in c), None)
    panels = loss_cols + ([map_col] if map_col else [])
    if not panels:
        return
 
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    flat_axes = axes.flatten()
 
    for ax, col in zip(flat_axes, panels):
        for group, group_df in curves.groupby("group"):
            ax.plot(group_df["epoch"], group_df[col], label=str(group))
        ax.set_title(col)
        ax.set_xlabel("epoch")
        ax.set_ylabel(col)
        ax.legend(fontsize=8)
 
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
 
    fig.suptitle(f"Learning curves - {run_tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_tag}__learning_curves.png", dpi=150)
    plt.close(fig)



def plot_cross_config(all_summary, out_dir):
    """For each metric, compare model configurations grouped by insect group."""
    metrics = [m for m in COMPARISON_METRICS if m in all_summary.columns]
    groups = sorted(all_summary["group"].unique())
    configs = sorted(all_summary["run_tag"].unique())

    for metric in metrics:
        pivot = all_summary.pivot_table(
            index="group", columns="run_tag", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=groups, columns=configs)

        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(groups)), 5))
        n_configs = len(configs)
        width = 0.8 / max(n_configs, 1)
        positions = range(len(groups))

        for i, config in enumerate(configs):
            offsets = [p + (i - (n_configs - 1) / 2) * width for p in positions]
            ax.bar(offsets, pivot[config].values, width=width, label=config)

        ax.set_xticks(list(positions))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_ylabel(metric)
        suffix = " (lower is better)" if metric in LOWER_IS_BETTER \
            else " (higher is better)"
        ax.set_title(f"{metric} by group and configuration{suffix}")
        ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"cross_config__{metric}.png", dpi=150)
        plt.close(fig)


def best_config_per_group(all_summary):
    """Return the best run configuration per group, ranked by pose mAP."""
    if "pose_map" not in all_summary.columns:
        return pd.DataFrame()
    idx = all_summary.groupby("group")["pose_map"].idxmax()
    columns = ["group", "run_tag", "model", "pose_map", "pose_map50",
               "mpjpe_px", "pck_0.1"]
    columns = [c for c in columns if c in all_summary.columns]
    return all_summary.loc[idx, columns].reset_index(drop=True)


def plot_keypoint_conf_vs_error(per_keypoint, run_tag, out_dir):
    """Per-group scatter of keypoint confidence vs error on a square grid.

    One subplot per insect group, arranged on an m x m grid with
    m = ceil(sqrt(number_of_groups)). Every keypoint is labelled with its name.
    Axis limits and the median guide lines are shared so groups stay comparable.
    """
    needed = {"kpt_conf", "nmpjpe", "group", "kpt_name"}
    if per_keypoint.empty or not needed.issubset(per_keypoint.columns):
        return
    data = per_keypoint.dropna(subset=["kpt_conf", "nmpjpe"])
    if data.empty:
        return

    groups = sorted(data["group"].unique())
    m = math.ceil(math.sqrt(len(groups)))
    fig, axes = plt.subplots(m, m, figsize=(6 * m, 5 * m), squeeze=False)
    flat_axes = axes.flatten()

    x_min, x_max = data["kpt_conf"].min(), data["kpt_conf"].max()
    y_min, y_max = data["nmpjpe"].min(), data["nmpjpe"].max()
    conf_median = data["kpt_conf"].median()
    err_median = data["nmpjpe"].median()

    all_kp = [
    "head-top", "head-left","head-right","left-eye","right-eye","neck","thorax-left","thorax-right","thorax-bottom","body-left","body-right",
    "body-tip","left-antenna-0","left-antenna-1","left-antenna-2","right-antenna-0","right-antenna-1","right-antenna-2","left-forewing-base","left-forewing-tip","left-forewing-front",
    "left-forewing-rear","right-forewing-base","right-forewing-tip","right-forewing-front","right-forewing-rear","left-hindwing-base","left-hindwing-tip","left-hindwing-front","left-hindwing-rear",
    "right-hindwing-base","right-hindwing-tip","right-hindwing-front","right-hindwing-rear","left-leg-0","left-leg-1","left-leg-2","left-leg-3","right-leg-0","right-leg-1","right-leg-2","right-leg-3",
    ]

    C_base = ["red", "red", "red", "grey", "grey", "red", "orange", "orange", 
         "orange", "green", "green", "green", "cyan", "cyan",
         "cyan", "cyan", "cyan", "cyan", "m",
         "m", "m", "m", "m",
         "m", "m", "m", "purple",
         "purple", "purple", "purple", "purple",
         "purple", "purple", "purple", "blue", "blue",
         "blue", "blue", "blue", "blue", "blue", "blue"]

    C_names = {"red" : "head", "grey" : "eyes", "orange" : "thorax", "green" : "abdomen", 
               "cyan" : "antenna", "m" : "forewings", "purple" : "hindwings", "blue" : "legs"}

    color_dict = {kp:color for kp, color in zip(all_kp, C_base)}

    for ax, group in zip(flat_axes, groups):
        group_df = data[data["group"] == group]
        C = [color_dict[kp] for kp in group_df["kpt_name"]]
        for g in list(C_names.values()):
            c = next(key for key, value in C_names.items() if value == g)
            g_keypoints_colors = [color_dict[kpt] for kpt in group_df['kpt_name']]
            g_keypoints_group = [C_names[kpt] for kpt in g_keypoints_colors]
            g_keypoints_indexes = [x for x in range(len(g_keypoints_group)) if g_keypoints_group[x]==g]
            df_copy = group_df.copy()
            kpt_name_copy = df_copy["kpt_name"].to_numpy().tolist()
            kpt_conf_copy = df_copy["kpt_conf"].to_numpy().tolist()
            kpt_nmpjpe_copy = df_copy["nmpjpe"].to_numpy().tolist()

            curr_names = [kpt_name_copy[x] for x in g_keypoints_indexes]
            curr_confs = [kpt_conf_copy[x] for x in g_keypoints_indexes]
            curr_nmpjpes = [kpt_nmpjpe_copy[x] for x in g_keypoints_indexes]
            if len(g_keypoints_indexes) == 0:
                print(g)
            else:
                ax.scatter(curr_confs, curr_nmpjpes,
                        s=80, alpha=0.7, color=c, label=f"{g} keypoints")
        # for _, row in group_df.iterrows():
        #     ax.annotate(str(row["kpt_name"]),
        #                 (row["kpt_conf"], row["nmpjpe"]),
        #                 fontsize=10, xytext=(3, 3), textcoords="offset points")
        ax.axvline(conf_median, color="grey", ls="--", lw=0.8)
        ax.axhline(err_median, color="grey", ls="--", lw=0.8)
        if group != "Lepidoptera":
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_xlim((x_min + x_max) /2, x_max)
            ax.set_ylim(y_min, (y_min + y_max)/2)
        ax.set_title(str(group))
        ax.set_xlabel("predicted confidence (higher is better)")
        ax.set_ylabel("normalized error - nmpjpe (lower is better)")

    for ax in flat_axes[len(groups):]:
        ax.axis("off")

    handles, labels = flat_axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    fig.suptitle(f"Keypoint confidence vs error - {run_tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_tag}__keypoint_conf_error.png", dpi=150)
    plt.close(fig)


def plot_metric_heatmaps(all_summary, out_dir):
    """Heatmap per metric: rows are run configurations, columns are datasets.

    Rows are ordered by model family first, then by full configuration tag, so
    that the hyper-parameter variants of one model sit together. White lines
    separate the model families. Columns are the insect datasets (groups).
    """
    groups = sorted(all_summary["group"].unique())

    order = (all_summary[["run_tag", "model"]]
             .drop_duplicates()
             .sort_values(["model", "run_tag"]))
    row_order = order["run_tag"].tolist()
    row_models = order.set_index("run_tag")["model"].to_dict()

    for metric in HEATMAP_METRICS:
        if metric not in all_summary.columns:
            continue
        pivot = all_summary.pivot_table(index="run_tag", columns="group",
                                        values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=row_order, columns=groups)
        values = pivot.values
        n_rows, n_cols = pivot.shape

        fig, ax = plt.subplots(figsize=(1.6 * n_cols + 3, 0.5 * n_rows + 2))
        image = ax.imshow(values, aspect="auto", cmap="viridis")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_order, fontsize=7)
        ax.set_xlabel("dataset")
        ax.set_ylabel("model configuration")
        ax.set_title(f"{metric} by configuration and dataset (higher is better)")

        finite = values[np.isfinite(values)]
        threshold = (finite.min() + finite.max()) / 2 if finite.size else 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                value = values[i, j]
                if not np.isfinite(value):
                    continue
                color = "white" if value < threshold else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

        for idx in range(1, n_rows):
            if row_models[row_order[idx]] != row_models[row_order[idx - 1]]:
                ax.axhline(idx - 0.5, color="white", lw=2)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap__{metric}.png", dpi=150)
        plt.close(fig)


def discover_models(results_dir):
    """Group ``.pt`` model files by configuration.

    Model files are named ``models_<parameters>__<group>.pt``. The part before
    the final ``__`` is the configuration tag; the part after is the training
    group. Returns ``{config: {group: path}}``.
    """
    models_by_config = {}
    for path in sorted(Path(results_dir).glob("*.pt")):
        stem = path.stem
        if "__" not in stem:
            print(f"Skipping '{path.name}': name does not match "
                  f"'models_<params>__<group>.pt'.")
            continue
        config, group = stem.rsplit("__", 1)
        models_by_config.setdefault(config, {})[group] = path
    return models_by_config


def find_example_image(group, search_dirs):
    """Return the ``<group>_example.{png,jpg,jpeg}`` image path, or None."""
    for directory in search_dirs:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = Path(directory) / f"{group}_example{ext}"
            if candidate.exists():
                return candidate
    return None


def extract_keypoints(results):
    """Extract (xy, conf) for the most confident detected instance.

    Returns ``(xy, conf)`` where ``xy`` has shape ``(K, 2)`` and ``conf`` has
    shape ``(K,)`` (or None). Returns ``(None, None)`` if nothing was detected.
    """
    if not results:
        return None, None
    res = results[0]
    kpts = getattr(res, "keypoints", None)
    if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
        return None, None

    xy_all = kpts.xy.cpu().numpy()
    conf_all = kpts.conf.cpu().numpy() if kpts.conf is not None else None

    best = 0
    boxes = getattr(res, "boxes", None)
    if boxes is not None and boxes.conf is not None and len(boxes.conf):
        best = int(boxes.conf.cpu().numpy().argmax())

    xy = xy_all[best]
    conf = conf_all[best] if conf_all is not None else None
    return xy, conf


def draw_pose(ax, xy, conf, line_width, point_size):
    """Draw skeleton edges then keypoints onto ``ax``.

    Edges take the colour of their start keypoint. Keypoints with confidence
    <= 0 or sitting exactly at the origin (i.e. not detected) are skipped.
    """
    visible = {}
    for i in range(min(len(xy), len(KEYPOINT_ORDER))):
        name = KEYPOINT_ORDER[i]
        x, y = float(xy[i][0]), float(xy[i][1])
        c = float(conf[i]) if conf is not None else 1.0
        if c <= 0 or (x == 0.0 and y == 0.0):
            continue
        visible[name] = (x, y)

    # Connections first so the points are drawn on top of them.
    for start, end in KEYPOINT_SKELETON:
        if start in visible and end in visible:
            x0, y0 = visible[start]
            x1, y1 = visible[end]
            ax.plot([x0, x1], [y0, y1],
                    color=KEYPOINT_COLORS.get(start, "white"),
                    lw=line_width, alpha=0.85, zorder=2)

    # Keypoints on top.
    for name, (x, y) in visible.items():
        ax.scatter([x], [y], s=point_size,
                   color=KEYPOINT_COLORS.get(name, "white"),
                   edgecolors="black", linewidths=max(line_width * 0.3, 0.3),
                   zorder=3)


def plot_inference_grid(config, group_models, groups, search_dirs, out_dir,
                        yolo_cls):
    """Render one square grid (2x2 for 4 groups) for a model configuration.

    Each cell shows the model trained on a group inferring on that group's
    example image. Missing models, missing images, failed inferences or empty
    predictions only annotate their own cell and never abort the others.
    """
    m = math.ceil(math.sqrt(len(groups)))
    fig, axes = plt.subplots(m, m, figsize=(7 * m, 7 * m), squeeze=False)
    flat_axes = axes.flatten()

    for ax, group in zip(flat_axes, groups):
        ax.set_title(str(group))
        ax.axis("off")

        model_path = group_models.get(group)
        if model_path is None:
            message = f"No model for group '{group}'"
            print(f"[{config}] {message}")
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="red")
            continue

        image_path = find_example_image(group, search_dirs)
        if image_path is None:
            message = f"No example image '{group}_example.*'"
            print(f"[{config}] {message}")
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="red")
            continue

        try:
            image = plt.imread(str(image_path))
            model = yolo_cls(str(model_path))
            results = model.predict(str(image_path), verbose=False)
        except Exception as exc:  # an error must not break the other cells
            message = f"Inference failed ({group}): {exc}"
            print(f"[{config}] {message}")
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red", wrap=True)
            continue

        ax.imshow(image)
        height, width = image.shape[:2]
        diagonal = math.hypot(height, width)
        line_width = max(diagonal * INFERENCE_LINE_SCALE, 0.5)
        point_size = (diagonal * INFERENCE_POINT_SCALE) ** 2

        xy, conf = extract_keypoints(results)
        if xy is None:
            message = f"No keypoints detected ({group})"
            print(f"[{config}] {message}")
            ax.text(0.5, 0.97, message, ha="center", va="top",
                    transform=ax.transAxes, fontsize=10, color="red")
            continue

        draw_pose(ax, xy, conf, line_width, point_size)

    for ax in flat_axes[len(groups):]:
        ax.axis("off")

    fig.suptitle(f"Pose inference on examples - {config}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / f"{config}__example_inference.png", dpi=150)
    plt.close(fig)


def generate_example_inferences(results_dir, out_dir):
    """Discover models, infer on the example images and save the grids.

    Skips cleanly (with a message) if no model is found or if ultralytics is
    not importable, so the rest of the comparison still runs.
    """
    models_by_config = discover_models(results_dir)
    if not models_by_config:
        print(f"No '*.pt' models found in {results_dir}; "
              f"skipping example-inference figures.")
        return

    groups = sorted({group
                     for models in models_by_config.values()
                     for group in models})
    search_dirs = [Path(results_dir), Path(results_dir) / "examples"]

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed; "
              "skipping example-inference figures.")
        return

    for config in tqdm.tqdm(sorted(models_by_config),
                            desc="example inference"):
        plot_inference_grid(config, models_by_config[config], groups,
                            search_dirs, out_dir, YOLO)


def main():
    args = parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        print(f"No result_*.xlsx files found in {args.results_dir}")
        return

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for run in tqdm.tqdm(runs):
        plot_group_comparison(run["summary"], run["run_tag"], figures_dir)
        plot_learning_curves(run["curves"], run["run_tag"], figures_dir)
        plot_keypoint_conf_vs_error(
            run["per_keypoint"], run["run_tag"], figures_dir)

    generate_example_inferences(args.results_dir, figures_dir)

    all_summary = pd.concat([run["summary"] for run in runs],
                            ignore_index=True)
    front = ["run_tag", "model", "group"]
    ordered = front + [c for c in all_summary.columns if c not in front]
    all_summary = all_summary[ordered]

    plot_cross_config(all_summary, figures_dir)
    plot_metric_heatmaps(all_summary, figures_dir)
    best_df = best_config_per_group(all_summary)

    per_keypoint_frames = [run["per_keypoint"] for run in runs
                           if not run["per_keypoint"].empty]
    all_per_keypoint = pd.concat(per_keypoint_frames, ignore_index=True) \
        if per_keypoint_frames else pd.DataFrame()

    out_path = out_dir / "comparison_summary.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_summary.to_excel(writer, sheet_name="all_runs", index=False)
        best_df.to_excel(writer, sheet_name="best_per_group", index=False)
        if not all_per_keypoint.empty:
            all_per_keypoint.to_excel(
                writer, sheet_name="per_keypoint_all", index=False)
        for metric in COMPARISON_METRICS:
            if metric in all_summary.columns:
                pivot = all_summary.pivot_table(
                    index="group", columns="run_tag",
                    values=metric, aggfunc="mean")
                pivot.to_excel(writer, sheet_name=f"by_group_{metric}"[:31])

    print(f"Combined comparison written to {out_path}")
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()