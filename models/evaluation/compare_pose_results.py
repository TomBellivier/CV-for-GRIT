"""
Aggregate and compare YOLO-pose results produced by train_eval_pose.py.

Loads every result_*.xlsx workbook in a directory and produces:
  - a combined comparison workbook (all runs, plus best configuration per
    group across all model types),
  - one bar chart per run configuration comparing the insect groups,
  - one learning-curve figure per run configuration (losses and pose mAP as a
    function of the training epoch, one line per group),
  - cross-configuration bar charts comparing model types within each group.

Example
-------
python compare_pose_results.py --results-dir pose_results --out-dir comparison
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Scalar metrics compared between groups. Lower is better only for the errors.
COMPARISON_METRICS = ["pose_map", "pose_map50", "mpjpe_px", "pck_0.1",
                      "mean_kpt_conf"]
LOWER_IS_BETTER = {"mpjpe_px", "nmpjpe"}

# Substrings used to locate the relevant columns inside learning_curves.
LOSS_HINTS = ["train/pose_loss", "train/kobj_loss", "val/pose_loss"]
MAP_HINT = "mAP50-95(P)"


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
    """Plot per-epoch losses and pose mAP, one line per group, for one run."""
    if curves.empty or "epoch" not in curves.columns:
        return

    loss_cols = [c for c in curves.columns if c in LOSS_HINTS]
    map_col = next((c for c in curves.columns if MAP_HINT in c), None)

    n_panels = len(loss_cols) + (1 if map_col else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    panels = loss_cols + ([map_col] if map_col else [])

    for ax, col in zip(axes, panels):
        for group, group_df in curves.groupby("group"):
            ax.plot(group_df["epoch"], group_df[col], label=str(group))
        ax.set_title(col)
        ax.set_xlabel("epoch")
        ax.set_ylabel(col)
        ax.legend(fontsize=8)

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


def plot_keypoint_conf_vs_error(per_keypoint, run_tag, out_dir, n_annotate=6):
    """Scatter of per-keypoint confidence vs error, one point per keypoint.

    Each point is one keypoint of one insect group. Median guide lines split the
    plane into quadrants (well localised vs poorly localised, confident vs not),
    and only the worst keypoints are labelled so the figure stays readable even
    with many keypoints and several groups.
    """
    needed = {"kpt_conf", "nmpjpe", "group", "kpt_name"}
    if per_keypoint.empty or not needed.issubset(per_keypoint.columns):
        return
    data = per_keypoint.dropna(subset=["kpt_conf", "nmpjpe"])
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    groups = sorted(data["group"].unique())
    cmap = plt.get_cmap("tab10")

    for i, group in enumerate(groups):
        group_df = data[data["group"] == group]
        ax.scatter(group_df["kpt_conf"], group_df["nmpjpe"],
                   s=30, alpha=0.6, color=cmap(i % 10), label=str(group))

    ax.axvline(data["kpt_conf"].median(), color="grey", ls="--", lw=0.8)
    ax.axhline(data["nmpjpe"].median(), color="grey", ls="--", lw=0.8)

    worst = data.nlargest(n_annotate, "nmpjpe")
    for _, row in worst.iterrows():
        ax.annotate(f"{row['kpt_name']} ({row['group']})",
                    (row["kpt_conf"], row["nmpjpe"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("predicted confidence (higher is better)")
    ax.set_ylabel("normalized error - nmpjpe (lower is better)")
    ax.set_title(f"Keypoint confidence vs error - {run_tag}")
    ax.legend(fontsize=8, title="group")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_tag}__keypoint_conf_error.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        print(f"No result_*.xlsx files found in {args.results_dir}")
        return

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        plot_group_comparison(run["summary"], run["run_tag"], figures_dir)
        plot_learning_curves(run["curves"], run["run_tag"], figures_dir)
        plot_keypoint_conf_vs_error(
            run["per_keypoint"], run["run_tag"], figures_dir)

    all_summary = pd.concat([run["summary"] for run in runs],
                            ignore_index=True)
    front = ["run_tag", "model", "group"]
    ordered = front + [c for c in all_summary.columns if c not in front]
    all_summary = all_summary[ordered]

    plot_cross_config(all_summary, figures_dir)
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
