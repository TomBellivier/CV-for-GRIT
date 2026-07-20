#!/usr/bin/env python3
"""
analyze_results.py
==================

Read the CSV produced by process_folder.py and write a set of figures (PNG) and
a text summary (TXT) describing the confidence distributions of a run.

Usage
-----
    python analyze_results.py --input results.csv --output-dir analysis_output
    python analyze_results.py --input results.csv --review-threshold 0.5

What it produces
----------------
Always (these columns are always in the CSV):
    confidence_boxplot_per_measurement.png   box plot of per-measurement
                                             confidence (mean shown)
    confidence_mean_std_per_measurement.png  bar chart of mean confidence per
                                             measurement with std error bars
    confidence_heatmap_per_measurement.png   density heatmap: x = measurement,
                                             y = confidence value, colour = share
                                             of images in that confidence bin
    hist_overall_pose_confidence.png         frequency histogram
    hist_scale_confidence.png                frequency histogram
    needs_review_pct.png                     % of rows flagged for review
                                             (derived here from the thresholds)
    scatter_pose_vs_scale_confidence.png     relationship between the two
    summary.txt                              all the numbers, incl. the %s

Only if the matching OPTIONAL column was enabled in config before the run:
    hist_detection_confidence.png            needs OPTIONAL_COLUMNS["detection_confidence"]
    scale_type_pct.png                       needs OPTIONAL_COLUMNS["scale_method"]

A note on "per keypoint"
------------------------
The CSV stores confidences PER MEASUREMENT (one value per measured distance),
not per keypoint, so the box plot / heatmap are per measurement. Getting true
per-keypoint figures would mean adding keypoint-confidence columns to the export
(a small change, but it means re-running the folder).
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # no display needed; render straight to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

CONF_SUFFIX = " [conf]"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def measurement_conf_columns(df: pd.DataFrame) -> list[str]:
    """All per-measurement confidence columns, in CSV order."""
    return [c for c in df.columns if c.endswith(CONF_SUFFIX)]


def short_label(conf_col: str) -> str:
    """'total length [conf]' -> 'total length'."""
    return conf_col[: -len(CONF_SUFFIX)]


def save(fig, output_dir: str, name: str):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def _series(df, col):
    """Numeric series for a column, NaNs dropped (empty if column absent)."""
    if col not in df.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_boxplot(df, conf_cols, output_dir):
    data = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in conf_cols]
    labels = [short_label(c) for c in conf_cols]
    fig, ax = plt.subplots(figsize=(max(8, len(conf_cols) * 0.5), 6))
    ax.boxplot(data, showmeans=True,
               flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-measurement confidence - distribution (mean = green triangle)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "confidence_boxplot_per_measurement.png")


def fig_mean_std(df, conf_cols, output_dir):
    means, stds, labels = [], [], []
    for c in conf_cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        means.append(v.mean())
        stds.append(v.std())
        labels.append(short_label(c))
    x = np.arange(len(conf_cols))
    fig, ax = plt.subplots(figsize=(max(8, len(conf_cols) * 0.5), 6))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("mean confidence (+/- std)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-measurement confidence - mean +/- standard deviation")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "confidence_mean_std_per_measurement.png")


def fig_heatmap(df, conf_cols, output_dir, n_bins=20):
    bins = np.linspace(0, 1, n_bins + 1)
    matrix = np.full((n_bins, len(conf_cols)), np.nan)
    for j, c in enumerate(conf_cols):
        v = pd.to_numeric(df[c], errors="coerce").dropna().values
        if v.size:
            counts, _ = np.histogram(v, bins=bins)
            total = counts.sum()
            matrix[:, j] = counts / total if total else 0.0
    fig, ax = plt.subplots(figsize=(max(8, len(conf_cols) * 0.5), 6))
    im = ax.imshow(matrix, origin="lower", aspect="auto",
                   extent=[0, len(conf_cols), 0, 1], cmap="viridis")
    ax.set_xticks(np.arange(len(conf_cols)) + 0.5)
    ax.set_xticklabels([short_label(c) for c in conf_cols], rotation=90, fontsize=7)
    ax.set_ylabel("confidence value")
    ax.set_title("Per-measurement confidence - density heatmap")
    fig.colorbar(im, ax=ax, label="share of images in bin")
    save(fig, output_dir, "confidence_heatmap_per_measurement.png")


def fig_hist(series, title, name, output_dir, xlabel="confidence", rng=(0, 1)):
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(series):
        ax.hist(series.values, bins=30, range=rng, color="#55A868", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency (images)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, name)


def fig_needs_review(df, output_dir, review_threshold):
    """Derive needs_review from overall pose and scale confidence thresholds."""
    pose = pd.to_numeric(df.get("overall_pose_confidence"), errors="coerce")
    scale = pd.to_numeric(df.get("scale_confidence"), errors="coerce")
    # A missing pose confidence (no insect detected) counts as needing review.
    flagged = ((pose < review_threshold) | (scale < review_threshold))
    flagged = flagged.fillna(True)
    pct = 100.0 * flagged.mean() if len(flagged) else 0.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["OK", "needs review"], [100 - pct, pct],
           color=["#4C72B0", "#C44E52"], alpha=0.85)
    ax.set_ylabel("% of images")
    ax.set_ylim(0, 100)
    ax.set_title(f"Needs review (threshold={review_threshold}): {pct:.1f}%")
    for i, val in enumerate([100 - pct, pct]):
        ax.text(i, val + 1, f"{val:.1f}%", ha="center", fontsize=10)
    save(fig, output_dir, "needs_review_pct.png")
    return pct


def fig_scale_type(df, output_dir):
    if "scale_method" not in df.columns:
        return None
    counts = df["scale_method"].fillna("none").astype(str).value_counts()
    pct = 100.0 * counts / counts.sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(pct.index.tolist(), pct.values, color="#8172B3", alpha=0.85)
    ax.set_ylabel("% of images")
    ax.set_ylim(0, 100)
    ax.set_title("Scale method used")
    for i, val in enumerate(pct.values):
        ax.text(i, val + 1, f"{val:.1f}%", ha="center", fontsize=10)
    save(fig, output_dir, "scale_type_pct.png")
    return pct.to_dict()


def fig_scatter_pose_scale(df, output_dir):
    pose = pd.to_numeric(df.get("overall_pose_confidence"), errors="coerce")
    scale = pd.to_numeric(df.get("scale_confidence"), errors="coerce")
    mask = pose.notna() & scale.notna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pose[mask], scale[mask], s=6, alpha=0.25, color="#4C72B0")
    ax.set_xlabel("overall pose confidence")
    ax.set_ylabel("scale confidence")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Pose vs scale confidence")
    ax.grid(alpha=0.3)
    save(fig, output_dir, "scatter_pose_vs_scale_confidence.png")


# --------------------------------------------------------------------------- #
# Summary text
# --------------------------------------------------------------------------- #
def write_summary(df, conf_cols, output_dir, review_threshold,
                  needs_review_pct, scale_type_pct):
    n = len(df)
    pose = _series(df, "overall_pose_confidence")
    scale = _series(df, "scale_confidence")
    n_no_pose = int(df["overall_pose_confidence"].isna().sum()) \
        if "overall_pose_confidence" in df.columns else n

    lines = []
    lines.append("=" * 64)
    lines.append("RESULTS SUMMARY")
    lines.append("=" * 64)
    lines.append(f"rows (images)                  : {n}")
    lines.append(f"images with a detected pose    : {n - n_no_pose} "
                 f"({100.0 * (n - n_no_pose) / n:.1f}%)" if n else "n/a")
    lines.append(f"images without a pose          : {n_no_pose} "
                 f"({100.0 * n_no_pose / n:.1f}%)" if n else "n/a")
    if "in_train" in df.columns:
        lines.append(f"in_train = 1                   : "
                     f"{int(pd.to_numeric(df['in_train'], errors='coerce').fillna(0).sum())}")
    if "in_val" in df.columns:
        lines.append(f"in_val = 1                     : "
                     f"{int(pd.to_numeric(df['in_val'], errors='coerce').fillna(0).sum())}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("GLOBAL CONFIDENCE")
    lines.append("-" * 64)
    if len(pose):
        lines.append(f"overall_pose_confidence  mean={pose.mean():.3f}  "
                     f"std={pose.std():.3f}  median={pose.median():.3f}")
    if len(scale):
        lines.append(f"scale_confidence         mean={scale.mean():.3f}  "
                     f"std={scale.std():.3f}  median={scale.median():.3f}")
    det = _series(df, "detection_confidence")
    if len(det):
        lines.append(f"detection_confidence     mean={det.mean():.3f}  "
                     f"std={det.std():.3f}  median={det.median():.3f}")
    lines.append("")

    lines.append("-" * 64)
    lines.append(f"NEEDS REVIEW (threshold = {review_threshold})")
    lines.append("-" * 64)
    lines.append(f"flagged: {needs_review_pct:.1f}% of images")
    lines.append("")

    lines.append("-" * 64)
    lines.append("SCALE METHOD")
    lines.append("-" * 64)
    if scale_type_pct:
        for k, v in scale_type_pct.items():
            lines.append(f"  {k:<12}: {v:.1f}%")
    else:
        lines.append("  (column 'scale_method' not in CSV - enable it in "
                     "config.OPTIONAL_COLUMNS and re-run to get this breakdown)")
    lines.append("")

    lines.append("-" * 64)
    lines.append("PER-MEASUREMENT CONFIDENCE (mean / std / median / n)")
    lines.append("-" * 64)
    for c in conf_cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v):
            lines.append(f"  {short_label(c):<30} mean={v.mean():.3f}  "
                         f"std={v.std():.3f}  median={v.median():.3f}  n={len(v)}")
        else:
            lines.append(f"  {short_label(c):<30} (no values)")
    lines.append("")

    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[txt] {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Make figures from a results CSV.")
    p.add_argument("--input", default="results.csv", help="Results CSV path.")
    p.add_argument("--output-dir", default="analysis_output", help="Where to write PNG/TXT.")
    p.add_argument("--review-threshold", type=float, default=0.5,
                   help="Confidence below which a row counts as 'needs review'.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    conf_cols = measurement_conf_columns(df)
    print(f"[data] {len(df)} rows, {len(conf_cols)} measurement-confidence columns")

    # Always-available figures.
    if conf_cols:
        fig_boxplot(df, conf_cols, args.output_dir)
        fig_mean_std(df, conf_cols, args.output_dir)
        fig_heatmap(df, conf_cols, args.output_dir)
    fig_hist(_series(df, "overall_pose_confidence"),
             "Overall pose confidence", "hist_overall_pose_confidence.png", args.output_dir)
    fig_hist(_series(df, "scale_confidence"),
             "Scale confidence", "hist_scale_confidence.png", args.output_dir)
    fig_scatter_pose_scale(df, args.output_dir)
    needs_review_pct = fig_needs_review(df, args.output_dir, args.review_threshold)

    # Figures that depend on optional columns.
    det = _series(df, "detection_confidence")
    if len(det):
        fig_hist(det, "Detection confidence", "hist_detection_confidence.png", args.output_dir)
    else:
        print("[skip] detection_confidence not in CSV "
              "(enable OPTIONAL_COLUMNS['detection_confidence'] and re-run).")
    scale_type_pct = fig_scale_type(df, args.output_dir)
    if scale_type_pct is None:
        print("[skip] scale_method not in CSV "
              "(enable OPTIONAL_COLUMNS['scale_method'] and re-run).")

    write_summary(df, conf_cols, args.output_dir, args.review_threshold,
                  needs_review_pct, scale_type_pct)
    print(f"\nDone. Figures and summary in: {args.output_dir}")


if __name__ == "__main__":
    main()