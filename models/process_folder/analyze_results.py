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
    python analyze_results.py --input results.csv --print --images-dir photos/

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
    hist_scale_px_per_mm.png                 scale distribution (spot bimodality)
    missing_rate_per_measurement.png         % missing per measurement (posed)
    lr_symmetry_scatter.png                  left vs right pairs (mm), y=x line
    confidence_correlation_matrix.png        confidence correlation between
                                             measurements
    cumulative_confidence.png                retention vs quality cutoff curve
    summary.txt                              all the numbers, incl. the %s

Only if the matching OPTIONAL column was enabled in config before the run:
    hist_detection_confidence.png            needs OPTIONAL_COLUMNS["detection_confidence"]
    scale_type_pct.png                       needs OPTIONAL_COLUMNS["scale_method"]
    scale_method_confusion_matrix.png        needs OPTIONAL_COLUMNS["scale_method"]
                                             AND the manual annotation JSON
                                             (--annotations, default
                                             annotations.json): per-class
                                             correct-classification rate over
                                             the annotated images only

Ground-truth error analysis (only if the YOLO label files are found under the
datasets root; labels are datasets/<dataset>/labels/<split>/<stem>.txt):
    error_vs_confidence_correlation.png      Spearman(error, confidence) per measurement
    error_vs_confidence_scatter.png          pooled confidence vs error + calibration line
    mean_error_vs_needs_review.png           mean error, flagged vs not
    rel_error_boxplot_per_measurement.png    error distribution per measurement
    mean_error_by_split.png                  error per train/val/test split
Keypoint-level (only if the raw keypoints were exported, EXPORT_KEYPOINTS=True):
    oks_vs_overall_confidence.png            OKS vs overall confidence (+corr)
    kp_error_vs_confidence_correlation.png   per-keypoint Spearman(conf, error)
    kp_error_vs_confidence_heatmap.png       error vs confidence heatmap per kp
    oks_histogram.png                        OKS distribution
    kp_mean_error.png                        mean error per keypoint (worst first)
Use --no-gt to skip it, --datasets-root to point elsewhere, --gt-splits to choose splits.

Annotated image copies (only with --print, needs --images-dir and Pillow):
    annotated_images/<image name>            copy of every image of the CSV with
                                             the pose keypoints, the measurement
                                             segments and the scale detection
                                             (scale_bar_box + scale_text_box OR
                                             ruler_line, following 'scale_method')
                                             drawn on it.
                                             See the "Annotated image copies"
                                             section for the columns it reads.

A note on "per keypoint"
------------------------
The CSV stores confidences PER MEASUREMENT (one value per measured distance),
not per keypoint, so the box plot / heatmap are per measurement. Getting true
per-keypoint figures would mean adding keypoint-confidence columns to the export
(a small change, but it means re-running the folder).
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display needed; render straight to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

# Project definitions are needed to rebuild ground-truth measurements from the
# YOLO label files. If the script is run outside the project, GT analysis is
# simply skipped (the rest of the figures still work).
try:
    from processing.definitions import (
        MEASUREMENT_INDICES, MEASUREMENT_NAMES as DEF_MEASUREMENT_NAMES,
        NUM_KEYPOINTS, KEYPOINT_NAMES,
    )
    from processing import config as proj_config
    HAVE_PROJECT = True
except Exception:                # noqa: BLE001
    HAVE_PROJECT = False

CONF_SUFFIX = " [conf]"
KP_X_SUFFIX = " [kp_x]"
KP_Y_SUFFIX = " [kp_y]"
KP_CONF_SUFFIX = " [kp_conf]"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def measurement_conf_columns(df: pd.DataFrame) -> list[str]:
    """All per-measurement confidence columns, in CSV order.

    Excludes the per-keypoint confidence columns ('... [kp_conf]').
    """
    return [c for c in df.columns
            if c.endswith(CONF_SUFFIX) and not c.endswith(KP_CONF_SUFFIX)]


def keypoint_names_in(df: pd.DataFrame) -> list[str]:
    """Keypoints that have x, y AND conf columns in the CSV (in CSV order)."""
    names = []
    for c in df.columns:
        if c.endswith(KP_CONF_SUFFIX):
            kp = c[: -len(KP_CONF_SUFFIX)]
            if (kp + KP_X_SUFFIX) in df.columns and (kp + KP_Y_SUFFIX) in df.columns:
                names.append(kp)
    return names


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


def px_col(m):   return f"{m} [px]"
def mm_col(m):   return f"{m} [mm]"
def conf_col(m): return f"{m} [conf]"


def measurement_names(df: pd.DataFrame) -> list[str]:
    """Measurement base names, taken from the ' [px]' columns, in CSV order."""
    return [c[: -len(" [px]")] for c in df.columns if c.endswith(" [px]")]


def lr_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Pair each 'left ...' measurement with its 'right ...' counterpart."""
    nameset = set(names)
    pairs = []
    for n in names:
        if "left" in n:
            r = n.replace("left", "right")
            if r in nameset:
                pairs.append((n, r))
    return pairs


def posed_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows where a pose was detected (overall_pose_confidence set)."""
    if "overall_pose_confidence" in df.columns:
        return pd.to_numeric(df["overall_pose_confidence"], errors="coerce").notna()
    return pd.Series(True, index=df.index)


def read_results(path) -> pd.DataFrame:
    """Read the results file, comma- or tab-separated (detected on the header)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)


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


# --------------------------------------------------------------------------- #
# Scale method vs manual annotations (confusion matrix)
# --------------------------------------------------------------------------- #
# annotate_gui.py writes {image path: int}; 0 = nothing, 1 = scale bar, 2 = ruler.
ANNOTATION_CLASSES = {0: "none", 1: "scale_bar", 2: "ruler"}

# Everything the pipeline may write in 'scale_method', folded onto those 3 names.
SCALE_METHOD_ALIASES = {
    "": "none", "none": "none", "nan": "none", "no_scale": "none", "unknown": "none",
    "scale_bar": "scale_bar", "scalebar": "scale_bar", "scale bar": "scale_bar",
    "ruler": "ruler", "regle": "ruler", "règle": "ruler",
}


def load_scale_annotations(path) -> dict[str, str]:
    """{image base name (lowercase) -> class name} read from the annotation JSON.

    The JSON is keyed by full image path while the CSV only stores the file
    name, so the join is done on the base name. Base names appearing several
    times in the JSON with conflicting labels are dropped: they cannot be
    matched to a CSV row without ambiguity.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    truth: dict[str, str] = {}
    conflicts: set[str] = set()
    for img_path, label in raw.items():
        # replace() so Windows-style keys still split correctly when run on Linux
        name = Path(str(img_path).replace("\\", "/")).name.lower()
        cls = ANNOTATION_CLASSES.get(int(label))
        if cls is None:
            continue
        if name in truth and truth[name] != cls:
            conflicts.add(name)
        truth[name] = cls

    for name in conflicts:
        truth.pop(name, None)
    if conflicts:
        print(f"[gt-scale] {len(conflicts)} base name(s) annotated twice with "
              f"different classes -> dropped (ambiguous match).")
    return truth


def fig_scale_method_confusion(df, output_dir, annotations_path):
    """Confusion matrix of 'scale_method' against the manual annotations.

    Only the CSV rows whose image is present in the annotation JSON are used;
    the rest of the CSV is ignored. Cells are row-normalised, so the diagonal
    reads directly as the per-class correct-classification rate (recall).
    """
    if "scale_method" not in df.columns or "image_name" not in df.columns:
        return None
    truth_by_name = load_scale_annotations(annotations_path)
    if not truth_by_name:
        print(f"[gt-scale] no usable annotations in {annotations_path} "
              f"-> confusion matrix skipped.")
        return None

    key = df["image_name"].astype(str).map(
        lambda s: Path(s.replace("\\", "/")).name.lower())
    truth = key.map(truth_by_name)
    pred = (df["scale_method"].fillna("none").astype(str).str.strip().str.lower()
            .map(lambda v: SCALE_METHOD_ALIASES.get(v, v)))

    matched = truth.notna()
    truth, pred = truth[matched], pred[matched]
    n = int(matched.sum())
    print(f"[gt-scale] {len(truth_by_name)} annotated images, "
          f"{n} matched in the CSV.")
    if n == 0:
        print("[gt-scale] no annotated image found in the CSV "
              "-> confusion matrix skipped (check the image names).")
        return None

    # get the three first encountered example for each class pair (truth, pred) for the summary
    example_names = {"none" : {"none" : [], "scale_bar": [], "ruler": []}, 
                     "scale_bar": {"none" : [], "scale_bar": [], "ruler": []}, 
                     "ruler": {"none" : [], "scale_bar": [], "ruler": []}}

    rows = ["none", "scale_bar", "ruler"]
    # any unexpected value in scale_method gets its own column instead of being
    # silently folded into 'none'
    extra = sorted(set(pred) - set(rows))
    cols = rows + extra
    if extra:
        print(f"[gt-scale] unexpected scale_method value(s): {', '.join(extra)}")

    cm = np.zeros((len(rows), len(cols)), dtype=int)
    for t, p in zip(truth, pred):
        if len(example_names[t][p]) < 3:
            example_names[t][p].append(key[matched][(truth == t) & (pred == p)].iloc[0])
        cm[rows.index(t), cols.index(p)] += 1

    for e, v in example_names.items():
        print(f"{e}: {v}")

    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = 100.0 * cm / support[:, None]          # row-normalised
    pct = np.where(support[:, None] > 0, pct, np.nan)
    recall = {r: (100.0 * cm[i, cols.index(r)] / support[i]) if support[i] else float("nan")
              for i, r in enumerate(rows)}
    accuracy = 100.0 * sum(cm[i, cols.index(r)] for i, r in enumerate(rows)) / n

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 5.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)),
                  [f"{r}\n(n={support[i]})" for i, r in enumerate(rows)])
    ax.set_xlabel("predicted (scale_method)")
    ax.set_ylabel("annotated (ground truth)")
    ax.set_title(f"Scale method vs manual annotations\n"
                 f"overall accuracy = {accuracy:.1f}%  (n = {n} annotated images)")

    for i in range(len(rows)):
        for j in range(len(cols)):
            if support[i] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=10, color="white" if pct[i, j] > 55 else "black")
    fig.colorbar(im, ax=ax, label="% of the annotated class (row)")
    save(fig, output_dir, "scale_method_confusion_matrix.png")

    return {"matrix": cm.tolist(), "rows": rows, "cols": cols,
            "support": support.tolist(), "recall": recall,
            "accuracy": accuracy, "n": n, "n_annotated": len(truth_by_name), "example_names": example_names}


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
# Extra figures (scale distribution, missing rate, L/R symmetry,
# confidence correlation, cumulative "quality cutoff" curve)
# --------------------------------------------------------------------------- #
def fig_scale_distribution(df, output_dir):
    """Histogram of scale_px_per_mm to reveal outliers / bimodality."""
    v = _series(df, "scale_px_per_mm")
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(v):
        ax.hist(v.values, bins=80, color="#DD8452", alpha=0.85)
        ax.axvline(v.median(), color="k", ls="--", lw=1,
                   label=f"median = {v.median():.1f}")
        ax.legend()
    ax.set_xlabel("scale (px/mm)")
    ax.set_ylabel("frequency (images)")
    ax.set_title("Distribution of scale_px_per_mm "
                 "(look for a second peak = a wrong scale mode)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "hist_scale_px_per_mm.png")
    return {"n": int(len(v)),
            "mean": float(v.mean()) if len(v) else float("nan"),
            "std": float(v.std()) if len(v) else float("nan"),
            "median": float(v.median()) if len(v) else float("nan"),
            "min": float(v.min()) if len(v) else float("nan"),
            "max": float(v.max()) if len(v) else float("nan")}


def fig_missing_rate(df, output_dir):
    """Per-measurement share of missing values, among images WITH a pose.

    Restricting to posed images isolates measurement-specific dropouts (e.g. a
    measurement set to NaN because a keypoint was below its visibility
    threshold) from the global 'no insect detected' case, which is reported
    separately in the summary.
    """
    names = measurement_names(df)
    sub = df[posed_mask(df)]
    denom = len(sub)
    rates = []
    for m in names:
        v = pd.to_numeric(sub[px_col(m)], errors="coerce") if px_col(m) in sub else pd.Series([], dtype=float)
        rates.append(100.0 * v.isna().mean() if denom else 0.0)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 6))
    ax.bar(x, rates, color="#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("% missing (among posed images)")
    ax.set_title(f"Missing-value rate per measurement (posed images: {denom})")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "missing_rate_per_measurement.png")
    return dict(zip(names, rates)), denom


def fig_lr_symmetry(df, output_dir):
    """Scatter of each left/right measurement pair (mm) with the y=x line.

    Points far from the diagonal reveal asymmetric errors (one side mis-placed).
    """
    pairs = lr_pairs(measurement_names(df))
    if not pairs:
        return
    ncols = 3
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (left, right) in zip(axes, pairs):
        lv = pd.to_numeric(df.get(mm_col(left)), errors="coerce")
        rv = pd.to_numeric(df.get(mm_col(right)), errors="coerce")
        mask = lv.notna() & rv.notna()
        if mask.any():
            ax.scatter(lv[mask], rv[mask], s=6, alpha=0.3, color="#4C72B0")
            hi = float(np.nanmax([lv[mask].max(), rv[mask].max()]))
            ax.plot([0, hi], [0, hi], "r--", lw=1)
        else:
            ax.text(0.5, 0.5, "no mm data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
        ax.set_xlabel(f"{left} [mm]", fontsize=7)
        ax.set_ylabel(f"{right} [mm]", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[len(pairs):]:          # hide unused cells
        ax.axis("off")
    fig.suptitle("Left/right symmetry (mm) - points off the red y=x line are asymmetric")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, output_dir, "lr_symmetry_scatter.png")


def fig_conf_correlation(df, conf_cols, output_dir):
    """Correlation matrix between the per-measurement confidences."""
    data = df[conf_cols].apply(pd.to_numeric, errors="coerce")
    corr = data.corr()                    # pairwise-complete Pearson
    labels = [short_label(c) for c in conf_cols]
    fig, ax = plt.subplots(figsize=(max(8, len(conf_cols) * 0.45),
                                    max(7, len(conf_cols) * 0.45)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Correlation of confidences between measurements")
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    save(fig, output_dir, "confidence_correlation_matrix.png")


def fig_cumulative(df, conf_cols, output_dir):
    """Survival curves: share of images whose confidence is >= a threshold.

    Read a curve top-down to pick a quality cutoff: at threshold t, the y value
    is the percentage of images you would keep. The 'worst measurement' curve
    uses, per image, the minimum confidence across all its measurements -- the
    strictest per-image criterion.
    """
    thr = np.linspace(0, 1, 101)

    def survival(s):
        s = s.dropna()
        return [100.0 * (s >= t).mean() for t in thr] if len(s) else [np.nan] * len(thr)

    pose = pd.to_numeric(df.get("overall_pose_confidence"), errors="coerce")
    scale = pd.to_numeric(df.get("scale_confidence"), errors="coerce")
    worst = df[conf_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, survival(pose), label="overall pose confidence", color="#4C72B0")
    ax.plot(thr, survival(scale), label="scale confidence", color="#DD8452")
    ax.plot(thr, survival(worst), label="worst measurement confidence", color="#55A868")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("% of images with confidence >= threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_title("Cumulative retention vs quality cutoff")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, output_dir, "cumulative_confidence.png")


# --------------------------------------------------------------------------- #
# Ground-truth error analysis (needs the YOLO label files)
# --------------------------------------------------------------------------- #
# The labels live next to the images in the dataset:
#     datasets/<dataset>/labels/<split>/<stem>.txt   (same stem as the image)
# Each line is a YOLO-pose instance:
#     class  cx cy w h  x1 y1 [v1]  x2 y2 [v2] ...    (all NORMALISED to [0,1])
# We rebuild the GT measurements in PIXELS (so they compare with the '[px]'
# columns) by de-normalising with the image width/height, then take the SUM of
# segment lengths, exactly like the pipeline.


def _num(x) -> float:
    """Coerce a CSV cell to float; '' / None / bad -> NaN."""
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _spearman(a, b) -> float:
    """Spearman rank correlation, computed without SciPy (rank + Pearson)."""
    a, b = pd.Series(list(a), dtype=float), pd.Series(list(b), dtype=float)
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    ra, rb = a[mask].rank(), b[mask].rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def build_gt_index(datasets_root: Path, splits):
    """Map stem -> label file, stem -> image file, stem -> split (all splits)."""
    labels, images, split_of = {}, {}, {}
    if not datasets_root.is_dir():
        return labels, images, split_of
    for dataset_dir in datasets_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        for split in splits:
            ldir = dataset_dir / "labels" / split
            if ldir.is_dir():
                for f in ldir.glob("*.txt"):
                    labels.setdefault(f.stem, f)
                    split_of.setdefault(f.stem, split)
            idir = dataset_dir / "images" / split
            if idir.is_dir():
                for f in idir.iterdir():
                    if f.is_file() and f.suffix.lower() in IMG_EXTS:
                        images.setdefault(f.stem, f)
    return labels, images, split_of


def parse_label_file(path: Path, num_kp: int):
    """Return the largest-box instance as {'xy':(N,2) normalised, 'vis':(N,) or None}.

    When several insects are annotated we keep the biggest box, which mirrors a
    'largest_box' selection. This can disagree with the instance the pipeline
    actually measured on multi-insect images (a known limitation for those).
    """
    best, best_area = None, -1.0
    try:
        text = path.read_text().splitlines()
    except OSError:
        return None
    for line in text:
        t = line.split()
        if len(t) < 5:
            continue
        try:
            vals = list(map(float, t[1:]))
        except ValueError:
            continue
        w, h = vals[2], vals[3]
        kp = vals[4:]
        if len(kp) == num_kp * 3:
            step = 3
        elif len(kp) == num_kp * 2:
            step = 2
        else:
            continue
        xs, ys = kp[0::step][:num_kp], kp[1::step][:num_kp]
        vis = kp[2::step][:num_kp] if step == 3 else None
        area = w * h
        if area > best_area:
            best_area = area
            best = {"xy": np.column_stack([xs, ys]).astype(float),
                    "vis": (np.array(vis, dtype=float) if vis is not None else None),
                    "area": float(area)}          # normalised bbox area (w*h)
    return best


_DIMS_CACHE: dict = {}


def get_dims(stem: str, row, images_map) -> tuple[int, int] | None:
    """Image (width, height): from the CSV columns if present, else from disk."""
    if "image_width" in row and "image_height" in row:
        w, h = _num(row["image_width"]), _num(row["image_height"])
        if w > 0 and h > 0:
            return int(w), int(h)
    p = images_map.get(stem)
    if p is None:
        return None
    if p in _DIMS_CACHE:
        return _DIMS_CACHE[p]
    try:
        from PIL import Image
        with Image.open(p) as im:
            wh = im.size                       # (width, height), header only
        _DIMS_CACHE[p] = wh
        return wh
    except Exception:                          # noqa: BLE001
        return None


def gt_measurements_px(xy_px, vis, meas_indices) -> dict:
    """GT measurement lengths in pixels (NaN if a keypoint is flagged absent)."""
    out = {}
    for m, idxs in meas_indices.items():
        if vis is not None and any(vis[i] == 0 for i in idxs):
            out[m] = float("nan")
            continue
        total = 0.0
        for a, b in zip(idxs[:-1], idxs[1:]):
            total += math.hypot(xy_px[a, 0] - xy_px[b, 0], xy_px[a, 1] - xy_px[b, 1])
        out[m] = total
    return out


def compute_errors(df, datasets_root: Path, splits, review_threshold,
                   oks_kappa: float = 0.05, pck_alpha: float = 0.10):
    """Match each CSV row to its GT label and accumulate the errors.

    Returns None if no labels were found. Otherwise a dict with, per measurement:
        per_measure[m] = {'rel':[], 'abs':[], 'conf':[]}
    and, when the raw keypoints were exported to the CSV, per keypoint:
        per_kp[kp]     = {'err':[], 'nerr':[], 'conf':[]}   (px, px/scale, conf)
    plus per-image aligned lists (img_mean_rel, img_needs_review, img_split,
    img_oks, img_pck, img_overall_conf) and n_gt.
    """
    labels_map, images_map, split_of = build_gt_index(datasets_root, splits)
    if not labels_map:
        return None

    meas_names = [m for m in DEF_MEASUREMENT_NAMES
                  if m in MEASUREMENT_INDICES and (px_col(m) in df.columns)]
    per = {m: {"rel": [], "abs": [], "conf": [], "minconf": []} for m in meas_names}
    img_mean_rel, img_nr, img_split = [], [], []

    # Keypoint-level setup (only if the raw kp columns are present).
    kp_names = keypoint_names_in(df)
    kp_index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    per_kp = {kp: {"err": [], "nerr": [], "conf": []} for kp in kp_names}
    img_oks, img_pck, img_overall_conf = [], [], []
    n_gt = 0

    for _, row in df.iterrows():
        stem = Path(str(row.get("image_name", ""))).stem
        lp = labels_map.get(stem)
        if lp is None:
            continue
        inst = parse_label_file(lp, NUM_KEYPOINTS)
        if inst is None:
            continue
        dims = get_dims(stem, row, images_map)
        if dims is None:
            continue
        W, H = dims
        xy = inst["xy"].copy()
        xy[:, 0] *= W
        xy[:, 1] *= H
        vis = inst["vis"]
        gt = gt_measurements_px(xy, vis, {m: MEASUREMENT_INDICES[m] for m in meas_names})
        n_gt += 1

        # ---- measurement-level errors ---------------------------------------
        rels = []
        for m in meas_names:
            pred = _num(row.get(px_col(m)))
            g = gt[m]
            conf = _num(row.get(conf_col(m)))
            if np.isnan(pred) or np.isnan(g) or g <= 0:
                continue
            per[m]["rel"].append(abs(pred - g) / g)
            per[m]["abs"].append(abs(pred - g))
            per[m]["conf"].append(conf)
            per[m]["minconf"].append(_min_kp_conf(row, m))
            rels.append(abs(pred - g) / g)

        # ---- keypoint-level errors + OKS + PCK ------------------------------
        area_px = inst["area"] * W * H            # GT object scale s^2
        if kp_names and area_px > 0:
            s = math.sqrt(area_px)
            oks_terms, pck_hits, pck_total = [], 0, 0
            for kp in kp_names:
                idx = kp_index[kp]
                if vis is not None and vis[idx] == 0:      # GT keypoint absent
                    continue
                px = _num(row.get(kp + KP_X_SUFFIX))
                py = _num(row.get(kp + KP_Y_SUFFIX))
                pc = _num(row.get(kp + KP_CONF_SUFFIX))
                if np.isnan(px) or np.isnan(py):           # no prediction
                    continue
                d = math.hypot(px - xy[idx, 0], py - xy[idx, 1])
                per_kp[kp]["err"].append(d)
                per_kp[kp]["nerr"].append(d / s)
                per_kp[kp]["conf"].append(pc)
                oks_terms.append(math.exp(-(d * d) / (2.0 * area_px * oks_kappa ** 2)))
                pck_total += 1
                pck_hits += int(d <= pck_alpha * s)
            if oks_terms:
                img_oks.append(float(np.mean(oks_terms)))
                img_pck.append(pck_hits / pck_total if pck_total else np.nan)
                img_overall_conf.append(_num(row.get("overall_pose_confidence")))

        pose = _num(row.get("overall_pose_confidence"))
        sc = _num(row.get("scale_confidence"))
        needs_review = (np.isnan(pose) or pose < review_threshold
                        or (not np.isnan(sc) and sc < review_threshold))
        if rels:
            img_mean_rel.append(float(np.mean(rels)))
            img_nr.append(bool(needs_review))
            img_split.append(split_of.get(stem, "?"))

    return {"per_measure": per, "meas_names": meas_names,
            "img_mean_rel": np.array(img_mean_rel),
            "img_needs_review": np.array(img_nr, dtype=bool),
            "img_split": np.array(img_split, dtype=object),
            "kp_names": kp_names, "per_kp": per_kp,
            "img_oks": np.array(img_oks),
            "img_pck": np.array(img_pck),
            "img_overall_conf": np.array(img_overall_conf),
            "oks_kappa": oks_kappa, "pck_alpha": pck_alpha,
            "n_gt": n_gt}


# ----- GT figures ---------------------------------------------------------- #
def fig_error_vs_conf_correlation(err, output_dir):
    """Spearman correlation between per-measurement error and confidence.

    A good confidence is NEGATIVELY correlated with the error (higher confidence
    -> smaller error), so useful bars point DOWN.
    """
    names, corrs = [], []
    for m in err["meas_names"]:
        d = err["per_measure"][m]
        names.append(m)
        corrs.append(_spearman(d["conf"], d["rel"]))
    x = np.arange(len(names))
    colors = ["#55A868" if (c is not None and c < 0) else "#C44E52" for c in corrs]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 6))
    ax.bar(x, corrs, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Spearman(confidence, relative error)")
    ax.set_ylim(-1, 1)
    ax.set_title("Error vs confidence per measurement (negative = confidence works)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "error_vs_confidence_correlation.png")
    return dict(zip(names, corrs))


def fig_error_vs_conf_scatter(err, output_dir, n_bins=10):
    """Pooled confidence vs relative error, with a binned-mean calibration line."""
    conf = np.concatenate([err["per_measure"][m]["conf"] for m in err["meas_names"]]) \
        if err["meas_names"] else np.array([])
    rel = np.concatenate([err["per_measure"][m]["rel"] for m in err["meas_names"]]) \
        if err["meas_names"] else np.array([])
    mask = ~np.isnan(conf) & ~np.isnan(rel)
    conf, rel = conf[mask], rel[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    if conf.size:
        ax.scatter(conf, rel, s=5, alpha=0.15, color="#4C72B0")
        # binned mean error per confidence bin (a reliability / calibration line)
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(conf, bins) - 1
        xs, ys = [], []
        for b in range(n_bins):
            sel = idx == b
            if sel.sum():
                xs.append((bins[b] + bins[b + 1]) / 2)
                ys.append(rel[sel].mean())
        ax.plot(xs, ys, "o-", color="#C44E52", label="mean error per confidence bin")
        # clip the y view to the 99th percentile so a few outliers don't flatten it
        ax.set_ylim(0, float(np.percentile(rel, 99)) if rel.size else 1)
        ax.legend()
    ax.set_xlabel("measurement confidence")
    ax.set_ylabel("relative error |pred - gt| / gt")
    ax.set_xlim(0, 1)
    ax.set_title("Confidence vs error (pooled over measurements)")
    ax.grid(alpha=0.3)
    save(fig, output_dir, "error_vs_confidence_scatter.png")


def fig_mean_error_vs_needs_review(err, output_dir):
    """Mean relative error for flagged vs non-flagged images (+/- SEM)."""
    rel = err["img_mean_rel"]
    nr = err["img_needs_review"]
    groups = [("not flagged", rel[~nr]), ("needs review", rel[nr])]
    means = [g.mean() if g.size else np.nan for _, g in groups]
    sems = [g.std() / math.sqrt(g.size) if g.size else 0.0 for _, g in groups]
    labels = [f"{lab}\n(n={g.size})" for lab, g in groups]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([0, 1], means, yerr=sems, capsize=5,
           color=["#4C72B0", "#C44E52"], alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean relative error (per image)")
    ax.set_title("Mean error vs needs_review")
    for i, mval in enumerate(means):
        if not math.isnan(mval):
            ax.text(i, mval, f"{mval:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "mean_error_vs_needs_review.png")
    return means


def fig_rel_error_boxplot(err, output_dir):
    """Distribution of the relative error per measurement (which are hardest)."""
    names = [m for m in err["meas_names"] if err["per_measure"][m]["rel"]]
    data = [np.array(err["per_measure"][m]["rel"]) for m in names]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 6))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("relative error")
    ax.set_title("Relative error per measurement (outliers hidden)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "rel_error_boxplot_per_measurement.png")


def fig_error_by_split(err, output_dir):
    """Mean relative error per split (train / val / test) to check generalisation."""
    rel, split = err["img_mean_rel"], err["img_split"]
    order = [s for s in ["train", "val", "test"] if s in set(split)]
    if not order:
        return
    means = [rel[split == s].mean() for s in order]
    sems = [rel[split == s].std() / math.sqrt(max(1, (split == s).sum())) for s in order]
    counts = [(split == s).sum() for s in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(range(len(order)), means, yerr=sems, capsize=5, color="#8172B3", alpha=0.85)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{s}\n(n={c})" for s, c in zip(order, counts)])
    ax.set_ylabel("mean relative error (per image)")
    ax.set_title("Mean error by split")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "mean_error_by_split.png")


# ----- keypoint-level figures (need EXPORT_KEYPOINTS in the CSV) ------------ #
def fig_oks_vs_overall_conf(err, output_dir):
    """OKS (per image) vs overall pose confidence, with the correlation shown.

    OKS rewards keypoints close to the GT (scaled by object size); a good
    overall confidence should rise WITH OKS (positive correlation).
    """
    oks = err["img_oks"]
    conf = err["img_overall_conf"]
    mask = ~np.isnan(oks) & ~np.isnan(conf)
    oks, conf = oks[mask], conf[mask]
    rho = _spearman(conf, oks)

    fig, ax = plt.subplots(figsize=(7, 6))
    if oks.size:
        ax.scatter(conf, oks, s=6, alpha=0.25, color="#4C72B0")
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(conf, bins) - 1
        xs, ys = [], []
        for b in range(10):
            sel = idx == b
            if sel.sum():
                xs.append((bins[b] + bins[b + 1]) / 2)
                ys.append(oks[sel].mean())
        ax.plot(xs, ys, "o-", color="#C44E52", label="mean OKS per confidence bin")
        ax.legend()
    ax.set_xlabel("overall pose confidence")
    ax.set_ylabel("OKS")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"OKS vs overall confidence  (Spearman = {rho:.3f})")
    ax.grid(alpha=0.3)
    save(fig, output_dir, "oks_vs_overall_confidence.png")
    return rho


def fig_kp_error_conf_correlation(err, output_dir):
    """Per-keypoint Spearman(confidence, normalised error). Negative = good."""
    names = err["kp_names"]
    corrs = [_spearman(err["per_kp"][kp]["conf"], err["per_kp"][kp]["nerr"]) for kp in names]
    x = np.arange(len(names))
    colors = ["#55A868" if (c is not None and c < 0) else "#C44E52" for c in corrs]
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.32), 6))
    ax.bar(x, corrs, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("Spearman(confidence, normalised error)")
    ax.set_ylim(-1, 1)
    ax.set_title("Per-keypoint: error vs confidence (negative = confidence works)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "kp_error_vs_confidence_correlation.png")
    return dict(zip(names, corrs))


def fig_kp_error_conf_heatmap(err, output_dir, n_bins=10):
    """Heatmap of mean normalised error vs confidence, for each keypoint.

    x = keypoint, y = confidence bin, colour = mean error (per keypoint per bin).
    A well-behaved keypoint shows a clear vertical gradient (low error at high
    confidence, i.e. dark at the top).
    """
    names = err["kp_names"]
    bins = np.linspace(0, 1, n_bins + 1)
    matrix = np.full((n_bins, len(names)), np.nan)
    for j, kp in enumerate(names):
        conf = np.asarray(err["per_kp"][kp]["conf"], dtype=float)
        nerr = np.asarray(err["per_kp"][kp]["nerr"], dtype=float)
        ok = ~np.isnan(conf) & ~np.isnan(nerr)
        conf, nerr = conf[ok], nerr[ok]
        if not conf.size:
            continue
        idx = np.digitize(conf, bins) - 1
        for b in range(n_bins):
            sel = idx == b
            if sel.sum():
                matrix[b, j] = nerr[sel].mean()

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.32), 6))
    # Cap the colour scale at the 95th percentile so outliers don't wash it out.
    vmax = np.nanpercentile(matrix, 95) if np.isfinite(matrix).any() else 1.0
    im = ax.imshow(matrix, origin="lower", aspect="auto",
                   extent=[0, len(names), 0, 1], cmap="magma_r", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(names)) + 0.5)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("confidence")
    ax.set_title("Mean normalised error vs confidence, per keypoint")
    fig.colorbar(im, ax=ax, label="mean error (px / object scale)")
    save(fig, output_dir, "kp_error_vs_confidence_heatmap.png")


def fig_oks_histogram(err, output_dir):
    """Distribution of per-image OKS (a standard pose-quality overview)."""
    oks = err["img_oks"]
    oks = oks[~np.isnan(oks)]
    fig, ax = plt.subplots(figsize=(7, 5))
    if oks.size:
        ax.hist(oks, bins=30, range=(0, 1), color="#55A868", alpha=0.85)
        ax.axvline(oks.mean(), color="k", ls="--", lw=1, label=f"mean = {oks.mean():.3f}")
        ax.legend()
    ax.set_xlabel("OKS")
    ax.set_ylabel("frequency (images)")
    ax.set_title(f"OKS distribution (kappa={err['oks_kappa']})")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "oks_histogram.png")


def fig_kp_mean_error(err, output_dir):
    """Mean normalised error per keypoint (which keypoints are hardest)."""
    names = err["kp_names"]
    means = [float(np.mean(err["per_kp"][kp]["nerr"])) if err["per_kp"][kp]["nerr"]
             else np.nan for kp in names]
    order = np.argsort([-(m if not math.isnan(m) else -1) for m in means])
    names_s = [names[i] for i in order]
    means_s = [means[i] for i in order]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.32), 6))
    ax.bar(x, means_s, color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=90, fontsize=6)
    ax.set_ylabel("mean normalised error (px / object scale)")
    ax.set_title("Mean per-keypoint error (worst first)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, output_dir, "kp_mean_error.png")

def _min_kp_conf(row, m) -> float:
    """Confiance agrégée d'une mesure = min des confiances de ses keypoints.

    NaN si aucune colonne '[kp_conf]' n'est présente (EXPORT_KEYPOINTS=False)
    ou si aucun des keypoints de la mesure n'a de confiance exportée.
    """
    cs = [_num(row.get(KEYPOINT_NAMES[i] + KP_CONF_SUFFIX))
          for i in MEASUREMENT_INDICES[m]]
    cs = [c for c in cs if not np.isnan(c)]
    return min(cs) if cs else float("nan")


def _auc(scores, labels) -> tuple[float, int, int]:
    """AUC ROC par la statistique de Mann-Whitney (rangs, ties moyennés).

    labels = True pour un vrai positif (mesure acceptable). Un score plus
    élevé doit indiquer une mesure meilleure. Renvoie (auc, n_pos, n_neg) ;
    auc = NaN si une des deux classes est vide.
    """
    s = pd.Series(list(scores), dtype=float)
    y = pd.Series(list(labels), dtype=bool)
    mask = s.notna()
    s, y = s[mask], y[mask]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    r = s.rank()                                  # rangs croissants, ties moyennés
    auc = (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg

def _roc_curve(scores, labels):
    """Courbe ROC sans sklearn.

    Renvoie (fpr, tpr, n_pos, n_neg) avec les seuils par ordre décroissant de
    score, ou None si une des deux classes est vide. Les ex-aequo sont
    regroupés sur un seul point (un seuil ne peut pas les séparer).
    """
    s = pd.Series(list(scores), dtype=float)
    y = pd.Series(list(labels), dtype=bool)
    mask = s.notna()
    s, y = s[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    keep = np.r_[np.diff(s) != 0, True]          # dernier index de chaque score
    tp = np.cumsum(y)[keep]
    fp = np.cumsum(~y)[keep]
    return (np.r_[0.0, fp / n_neg], np.r_[0.0, tp / n_pos], n_pos, n_neg)

def fig_auc_min_aggregation(err, output_dir, tol=0.05):
    """AUC par mesure du baseline 'min des confiances kp' contre le label
    'erreur relative < tol'.

    C'est le test de faisabilité à faire avant d'investir dans une agrégation
    apprise : si le min des confiances kp sépare mal les mesures bonnes des
    mauvaises (AUC ~ 0.5-0.6), aucune pondération de ces mêmes confiances ne
    fera beaucoup mieux, et il faut d'autres features. La confiance déjà
    stockée dans le CSV est tracée à côté comme point de comparaison.

    La ligne pointillée à 0.5 est le hasard. La classe positive est la mesure
    ACCEPTABLE, donc une AUC > 0.5 signifie que la confiance est informative.
    """
    names, auc_min, auc_csv, rates, ns = [], [], [], [], []
    for m in err["meas_names"]:
        d = err["per_measure"][m]
        rel = np.asarray(d["rel"], dtype=float)
        if rel.size == 0:
            continue
        good = rel < tol                              # label: vrai positif
        a_min, n_pos, n_neg = _auc(d.get("minconf", []), good)
        a_csv, _, _ = _auc(d["conf"], good)
        names.append(m)
        auc_min.append(a_min)
        auc_csv.append(a_csv)
        rates.append(100.0 * good.mean())
        ns.append(n_pos + n_neg)

    if not names:
        print("[auc] aucune mesure avec des erreurs GT -> figure sautee.")
        return None
    if all(math.isnan(a) for a in auc_min):
        print("[auc] pas de colonnes '[kp_conf]' dans le CSV -> agregation min "
              "indisponible (mettre EXPORT_KEYPOINTS=True et relancer).")

    # AUC poolee, toutes mesures confondues
    pooled_s = np.concatenate([np.asarray(err["per_measure"][m].get("minconf", []),
                                          dtype=float) for m in names]) \
        if names else np.array([])
    pooled_y = np.concatenate([np.asarray(err["per_measure"][m]["rel"],
                                          dtype=float) < tol for m in names])
    pooled_auc = (_auc(pooled_s, pooled_y)[0]
                  if pooled_s.size == pooled_y.size and pooled_s.size else float("nan"))

    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.55), 6))
    ax.bar(x - w / 2, auc_min, w, color="#4C72B0", alpha=0.85,
           label="min des confiances kp")
    ax.bar(x + w / 2, auc_csv, w, color="#DD8452", alpha=0.85,
           label="confiance mesure (CSV)")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={n}, {r:.0f}% ok)" for m, n, r in zip(names, ns, rates)],
                       rotation=90, fontsize=6)
    ax.set_ylabel(f"AUC (positif = erreur relative < {tol:.0%})")
    ax.set_ylim(0, 1)
    ax.set_title(f"Pouvoir discriminant de la confiance par mesure "
                 f"(AUC poolee min-kp = {pooled_auc:.3f})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, output_dir, "auc_min_aggregation_per_measurement.png")

    return {"tol": tol, "pooled_auc_min": pooled_auc,
            "per_measure": {m: {"auc_min": a, "auc_csv": c, "n": n, "pos_rate": r}
                            for m, a, c, n, r in zip(names, auc_min, auc_csv, ns, rates)}}

def fig_roc_grid(err, output_dir, tol=0.05, score_key="minconf",
                 targets=(0.001, 0.005, 0.01), n_rows=3, n_cols=3, per_axes=3,
                 min_neg_factor=5.0, order="csv"):
    """Grille de mini-ROC en axe x logarithmique, 3 mesures par sous-figure.

    Positif = mesure acceptable (erreur relative < tol). L'axe x est en log
    pour lire le régime qui vous intéresse (FPR de 0.1 a 1%) : en échelle
    linéaire toute cette zone est écrasée contre l'axe et illisible.

    Résolution : une mesure ayant n_neg négatifs ne peut pas estimer un FPR
    plus fin que 1/n_neg. La portion de courbe sous min_neg_factor/n_neg est
    tracée en pointillé fin -- elle est définie par une poignée d'échantillons
    et ne doit pas être lue comme une performance atteignable.
    """
    score_of = lambda d: d.get(score_key) if d.get(score_key) else d["conf"]

    curves = []
    for m in err["meas_names"]:
        d = err["per_measure"][m]
        rel = np.asarray(d["rel"], dtype=float)
        if rel.size == 0:
            continue
        roc = _roc_curve(score_of(d), rel < tol)
        if roc is None:
            print(f"[roc] {m}: une seule classe presente -> ignoree.")
            continue
        fpr, tpr, n_pos, n_neg = roc
        curves.append({"name": m, "fpr": fpr, "tpr": tpr, "n_pos": n_pos,
                       "n_neg": n_neg, "auc": float(np.trapezoid(tpr, fpr))})

    if not curves:
        print("[roc] aucune mesure exploitable -> figure sautee.")
        return None
    if order == "auc":
        curves.sort(key=lambda c: c["auc"])       # les pires en premier
    cap = n_rows * n_cols * per_axes
    if len(curves) > cap:
        print(f"[roc] {len(curves)} mesures pour {cap} emplacements -> "
              f"{len(curves) - cap} non tracee(s).")
        curves = curves[:cap]

    # borne gauche commune : la resolution de la mesure la mieux fournie
    best_res = min(1.0 / c["n_neg"] for c in curves)
    xmin = max(1e-4, best_res * 0.5)
    palette = ["#4C72B0", "#C44E52", "#55A868"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for k, ax in enumerate(axes):
        group = curves[k * per_axes:(k + 1) * per_axes]
        if not group:
            ax.axis("off")
            continue
        for c, col in zip(group, palette):
            floor = min_neg_factor / c["n_neg"]
            f = np.clip(c["fpr"], xmin, 1.0)
            # courbe complete en pointillé fin, puis zone fiable en trait plein
            ax.plot(f, c["tpr"], drawstyle="steps-post", color=col, lw=0.8,
                    ls=":", alpha=0.7)
            solid = c["fpr"] >= floor
            if solid.any():
                ax.plot(f[solid], c["tpr"][solid], drawstyle="steps-post",
                        color=col, lw=1.6,
                        label=f"{c['name']}  AUC={c['auc']:.2f}  n-={c['n_neg']}")
            else:
                ax.plot([], [], color=col, lw=1.6,
                        label=f"{c['name']}  (n-={c['n_neg']}, trop peu)")
        for t in targets:
            ax.axvline(t, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax.plot([xmin, 1], [xmin, 1], color="k", lw=0.6, alpha=0.4)  # hasard
        ax.set_xscale("log")
        ax.set_xlim(xmin, 1.0)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"ROC par mesure, score = {score_key} "
                 f"(positif : erreur relative < {tol:.0%}) -- "
                 f"pointillé = sous la résolution de l'échantillon")
    fig.supxlabel("FPR (log) -- traits verticaux : " +
                  ", ".join(f"{t:.1%}" for t in targets), fontsize=9)
    fig.supylabel("rappel (TPR)", fontsize=9)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
    save(fig, output_dir, "roc_grid_per_measurement.png")

    # rappel atteignable a chaque FPR cible, pour le summary
    out = {}
    for c in curves:
        rec = {}
        for t in targets:
            ok = c["fpr"] <= t
            rec[t] = (float(c["tpr"][ok].max()) if ok.any() else 0.0) \
                if t >= min_neg_factor / c["n_neg"] else float("nan")
        out[c["name"]] = {"auc": c["auc"], "n_pos": c["n_pos"],
                          "n_neg": c["n_neg"], "recall_at": rec}
    return {"tol": tol, "score_key": score_key, "per_measure": out}

# --------------------------------------------------------------------------- #
# Summary text
# --------------------------------------------------------------------------- #
def write_summary(df, conf_cols, output_dir, review_threshold,
                  needs_review_pct, scale_type_pct,
                  scale_stats=None, missing_rates=None, missing_denom=0,
                  err=None, err_corr=None, scale_cm=None):
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
    lines.append("SCALE METHOD vs MANUAL ANNOTATIONS")
    lines.append("-" * 64)
    if scale_cm:
        lines.append(f"annotated images: {scale_cm['n_annotated']}  "
                     f"matched in the CSV: {scale_cm['n']}")
        lines.append(f"overall accuracy: {scale_cm['accuracy']:.1f}%")
        lines.append("")
        header = "  " + f"{'truth \\ pred':<14}" + "".join(f"{c:>12}" for c in scale_cm["cols"])
        lines.append(header + f"{'support':>10}")
        for i, r in enumerate(scale_cm["rows"]):
            cells = "".join(f"{v:>12}" for v in scale_cm["matrix"][i])
            lines.append(f"  {r:<14}{cells}{scale_cm['support'][i]:>10}")
        lines.append("")
        lines.append("  correctly classified per class:")
        for r in scale_cm["rows"]:
            v = scale_cm["recall"][r]
            lines.append(f"    {r:<12}: " + (f"{v:5.1f}%" if v == v else "  n/a (no annotated image)"))
    else:
        lines.append("  (no confusion matrix - needs 'scale_method' and 'image_name' "
                     "in the CSV plus an annotation JSON; pass --annotations)")
    lines.append("")

    lines.append("-" * 64)
    lines.append("SCALE px/mm DISTRIBUTION")
    lines.append("-" * 64)
    if scale_stats and scale_stats.get("n"):
        lines.append(f"  n={scale_stats['n']}  mean={scale_stats['mean']:.2f}  "
                     f"std={scale_stats['std']:.2f}  median={scale_stats['median']:.2f}")
        lines.append(f"  min={scale_stats['min']:.2f}  max={scale_stats['max']:.2f}  "
                     f"(a second peak in the histogram = a wrong-scale mode)")
    else:
        lines.append("  (no scale values)")
    lines.append("")

    if missing_rates:
        lines.append("-" * 64)
        lines.append(f"MISSING-VALUE RATE PER MEASUREMENT (posed images: {missing_denom})")
        lines.append("-" * 64)
        for m, rate in missing_rates.items():
            lines.append(f"  {m:<30} {rate:5.1f}% missing")
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

    # ----- ground-truth error section (only if labels were found) -----------
    if err is not None:
        lines.append("=" * 64)
        lines.append(f"GROUND-TRUTH ERROR  (images matched to a label: {err['n_gt']})")
        lines.append("=" * 64)
        rel_all = err["img_mean_rel"]
        if rel_all.size:
            lines.append(f"per-image mean relative error: mean={rel_all.mean():.3f}  "
                         f"median={np.median(rel_all):.3f}")
        if err_corr:
            nr = err["img_needs_review"]
            lines.append(f"mean error | needs_review=False : "
                         f"{rel_all[~nr].mean():.3f} (n={int((~nr).sum())})"
                         if (~nr).any() else "mean error | needs_review=False : n/a")
            lines.append(f"mean error | needs_review=True  : "
                         f"{rel_all[nr].mean():.3f} (n={int(nr.sum())})"
                         if nr.any() else "mean error | needs_review=True  : n/a")
        lines.append("")
        lines.append("  per measurement:  MAE[px]  mean_rel_err  spearman(conf,err)  n")
        for m in err["meas_names"]:
            d = err["per_measure"][m]
            n = len(d["rel"])
            if n:
                mae = float(np.mean(d["abs"]))
                mre = float(np.mean(d["rel"]))
                sp = err_corr.get(m, float("nan")) if err_corr else float("nan")
                lines.append(f"  {m:<30} {mae:8.1f}  {mre:11.3f}  {sp:17.3f}  {n}")
            else:
                lines.append(f"  {m:<30} (no matched GT)")
        lines.append("")

        # keypoint-level stats
        if err.get("kp_names") and err["img_oks"].size:
            oks = err["img_oks"][~np.isnan(err["img_oks"])]
            pck = err["img_pck"][~np.isnan(err["img_pck"])]
            lines.append("-" * 64)
            lines.append(f"KEYPOINTS (OKS kappa={err['oks_kappa']}, "
                         f"PCK alpha={err['pck_alpha']})")
            lines.append("-" * 64)
            if oks.size:
                lines.append(f"mean OKS = {oks.mean():.3f}   median OKS = {np.median(oks):.3f}")
            if pck.size:
                lines.append(f"mean PCK@{err['pck_alpha']} = {pck.mean():.3f}")
            lines.append(f"corr(OKS, overall_confidence) spearman = "
                         f"{_spearman(err['img_overall_conf'], err['img_oks']):.3f}")
            lines.append("")
            lines.append("  per keypoint:  mean_norm_err  spearman(conf,err)  n")
            for kp in err["kp_names"]:
                d = err["per_kp"][kp]
                n = len(d["nerr"])
                if n:
                    mne = float(np.mean(d["nerr"]))
                    sp = _spearman(d["conf"], d["nerr"])
                    lines.append(f"  {kp:<24} {mne:13.3f}  {sp:17.3f}  {n}")
                else:
                    lines.append(f"  {kp:<24} (no matched GT)")
            lines.append("")

    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[txt] {path}")


# --------------------------------------------------------------------------- #
# Annotated image copies (--print)
# --------------------------------------------------------------------------- #
# With --print, every image listed in the CSV is copied into the print folder
# (default <output-dir>/annotated_images/) with, drawn on top of it:
#   * the pose keypoints        ('<kp> [kp_x]' / '[kp_y]' / '[kp_conf]' columns)
#   * the measurement segments  same columns + the project definitions
#                               (MEASUREMENT_INDICES); a measurement left empty
#                               in the CSV is not drawn
#   * the scale detection, following 'scale_method':
#       scale_bar -> 'scale_bar_box' (+ 'scale_text_box' with the OCR text)
#       ruler     -> the line 'ruler_line' along 'ruler_orientation'
#     (a missing / unknown method draws whatever geometry is present)
# Keypoints and segments are coloured by their confidence (red = 0 -> green = 1,
# cyan = no confidence). All coordinates are image pixels, like the CSV.
#
# Accepted cell formats: JSON, Python literal or numpy repr ('[ 12.  30.5 ]').
# Values that all lie in [0, 1] are taken as normalised and rescaled.
#   boxes             : x1 y1 x2 y2 [conf], a list of such boxes, or dicts
#                       {"xyxy": [...], "conf": ...}
#   ruler_line        : 1 number  -> row (y) or column (x) index; the axis comes
#                                    from ruler_orientation; full image span
#                       3 numbers -> index, start, end along the line
#                       4 numbers -> segment x1 y1 x2 y2 (orientation not needed)
#                       [[x, y], ...] -> polyline
#   ruler_orientation : horizontal / row / ligne     -> horizontal line y = index
#                       vertical / col / colonne     -> vertical line   x = index
# A value that cannot be read is counted and one example is printed at the end,
# so an unexpected format shows up immediately in the console.
PRINT_COLUMNS = {
    "scale_bar_box": "scale_bar_box",
    "scale_text_box": "scale_text_box",
    "scale_ocr_text": "scale_ocr_text",
    "scale_bar_conf": "scale_bar_confidence",
    "ruler_line": "ruler_line",
    "ruler_orientation": "ruler_orientation",
    "ruler_conf": "ruler_confidence",
}
RULER_AXIS_ALIASES = {
    "horizontal": "row", "horizontale": "row", "h": "row", "row": "row", "ligne": "row",
    "vertical": "col", "verticale": "col", "v": "col", "col": "col", "column": "col",
    "colonne": "col",
}

PRINT_SCALE_COLOR = (255, 0, 255)          # magenta: scale bar box / ruler line
PRINT_TEXT_BOX_COLOR = (255, 150, 0)       # orange: scale text (OCR) box
PRINT_NAN_COLOR = (0, 200, 255)            # cyan: element without a confidence
PRINT_SAVE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _norm_scale_method(value) -> str:
    """'scale_method' cell folded onto none / scale_bar / ruler (like the matrix)."""
    v = "none" if value is None else str(value).strip().lower()
    return SCALE_METHOD_ALIASES.get(v, v)


def _literal(cell):
    """CSV cell -> Python object (number, list, dict...), None when empty.

    Tries JSON, then a Python literal, then falls back to the list of numbers
    found in the text (covers numpy reprs such as '[ 12.  30.5 400.   60. ]').
    """
    if cell is None:
        return None
    if _is_num(cell):
        return None if math.isnan(float(cell)) else float(cell)
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]", "()"):
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    nums = [float(v) for v in _NUM_RE.findall(s)]
    return nums or s                                   # s = unreadable text


def _normalised(values) -> bool:
    return all(0.0 <= v <= 1.0 for v in values)


def _parse_boxes(cell, W, H):
    """Boxes of one CSV cell -> [([x1, y1, x2, y2], conf), ...] in pixels.

    [] when the cell is empty, None when it cannot be read.
    """
    obj = _literal(cell)
    if obj is None:
        return []
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, (list, tuple)):
        return None
    if obj and all(_is_num(v) for v in obj):            # flat list of numbers
        n = len(obj)
        if n in (4, 5):
            obj = [obj]
        elif n % 4 == 0:                                # several boxes, flattened
            obj = [obj[i:i + 4] for i in range(0, n, 4)]
        else:
            return None

    boxes = []
    for b in obj:
        conf = float("nan")
        if isinstance(b, dict):
            conf = _num(b.get("conf", b.get("confidence")))
            b = b.get("xyxy", b.get("box", b.get("bbox")))
        if not isinstance(b, (list, tuple)) or len(b) < 4 or not all(_is_num(v) for v in b[:4]):
            continue
        xyxy = [float(v) for v in b[:4]]
        if any(math.isnan(v) for v in xyxy):
            continue
        if len(b) >= 5 and math.isnan(conf):
            conf = _num(b[4])
        if _normalised(xyxy):
            xyxy = [xyxy[0] * W, xyxy[1] * H, xyxy[2] * W, xyxy[3] * H]
        x1, x2 = sorted((xyxy[0], xyxy[2]))
        y1, y2 = sorted((xyxy[1], xyxy[3]))
        boxes.append(([x1, y1, x2, y2], conf))
    return boxes if boxes else None


def _ruler_axis(cell) -> str | None:
    """ruler_orientation -> 'row' (horizontal line) / 'col' (vertical) / None."""
    if cell is None or (_is_num(cell) and math.isnan(float(cell))):
        return None
    return RULER_AXIS_ALIASES.get(str(cell).strip().lower())


def _parse_ruler(line_cell, orient_cell, W, H):
    """ruler_line (+ ruler_orientation) -> polyline [(x, y), ...] in pixels.

    [] when the cell is empty, None when it cannot be read.
    """
    obj = _literal(line_cell)
    if obj is None:
        return []
    if _is_num(obj):
        obj = [obj]
    if not isinstance(obj, (list, tuple)) or not obj:
        return None

    # list of points [[x, y], ...]
    if all(isinstance(p, (list, tuple)) and len(p) >= 2 and _is_num(p[0]) and _is_num(p[1])
           for p in obj):
        pts = [(float(p[0]), float(p[1])) for p in obj]
        if len(pts) < 2:
            return None
        if _normalised([v for p in pts for v in p]):
            pts = [(x * W, y * H) for x, y in pts]
        return pts

    if not all(_is_num(v) for v in obj):
        return None
    v = [float(x) for x in obj]
    if len(v) == 4:                                     # segment x1 y1 x2 y2
        if _normalised(v):
            v = [v[0] * W, v[1] * H, v[2] * W, v[3] * H]
        return [(v[0], v[1]), (v[2], v[3])]

    axis = _ruler_axis(orient_cell)
    if len(v) in (1, 3) and axis is not None:           # index [+ start, end]
        across, along = (H, W) if axis == "row" else (W, H)
        pos = v[0] * across if 0.0 < v[0] < 1.0 else v[0]
        start, end = (v[1], v[2]) if len(v) == 3 else (0.0, float(along - 1))
        if len(v) == 3 and max(start, end) <= 1.0:
            start, end = start * along, end * along
        if axis == "row":
            return [(start, pos), (end, pos)]
        return [(pos, start), (pos, end)]
    return None


def _conf_rgb(c: float) -> tuple[int, int, int]:
    """Confidence -> RGB on the RdYlGn colour map (cyan when NaN)."""
    if c is None or math.isnan(c):
        return PRINT_NAN_COLOR
    r, g, b, _ = matplotlib.colormaps["RdYlGn"](min(max(float(c), 0.0), 1.0))
    return int(r * 255), int(g * 255), int(b * 255)


@functools.lru_cache(maxsize=8)
def _print_font(size: int):
    """TrueType font of the given size (DejaVu Sans ships with matplotlib)."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:                                   # noqa: BLE001
        try:
            return ImageFont.load_default(size=size)
        except TypeError:                               # Pillow < 10.1
            return ImageFont.load_default()


def _outlined_text(draw, xy, text, size, fill=(255, 255, 255), bounds=None):
    """Text with a black outline, readable on any background.

    bounds=(W, H) shifts the text back inside the image when it would overflow.
    """
    font, stroke = _print_font(size), max(1, size // 8)
    x, y = xy
    if bounds is not None:
        _, _, tw, th = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        x = min(max(0, x), max(0, bounds[0] - tw))
        y = min(max(0, y), max(0, bounds[1] - th))
    draw.text((x, y), text, font=font, fill=fill,
              stroke_width=stroke, stroke_fill=(0, 0, 0))


def _to_rgb(im):
    """Any PIL image -> 8-bit RGB (16-bit / float images are min-max stretched)."""
    from PIL import Image
    if im.mode == "RGB":
        return im
    if im.mode.startswith("I") or im.mode == "F":
        a = np.asarray(im, dtype=np.float64)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(a)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im.convert("RGB")


def _load_image_for_row(path: Path, row):
    """Open an image in the pixel frame of the CSV coordinates.

    Returns (RGB image, sx, sy): multiply a CSV coordinate by (sx, sy) to get
    the pixel in the returned image. The EXIF orientation is applied unless the
    CSV image_width / image_height say the pipeline worked on the raw frame.
    A mismatch that is neither (e.g. a resized copy) is handled by rescaling.
    """
    from PIL import Image, ImageOps
    im = Image.open(path)
    im.load()
    try:
        upright = ImageOps.exif_transpose(im)
    except Exception:                                   # noqa: BLE001
        upright = im

    w, h = _num(row.get("image_width")), _num(row.get("image_height"))
    csv_wh = (int(round(w)), int(round(h))) if (w > 0 and h > 0) else None
    if csv_wh is None or upright.size == csv_wh:
        img = upright
    elif im.size == csv_wh:
        img = im
    else:
        img = upright
    sx = img.width / csv_wh[0] if csv_wh else 1.0
    sy = img.height / csv_wh[1] if csv_wh else 1.0
    return _to_rgb(img), sx, sy


def _print_context(df, labels) -> dict:
    """What can be drawn with this CSV (tells the user once what is missing)."""
    kp_names = keypoint_names_in(df)
    meas = []
    if not kp_names:
        print("[print] no '[kp_x]' / '[kp_y]' / '[kp_conf]' columns in the CSV -> "
              "keypoints and measurement segments not drawn "
              "(set EXPORT_KEYPOINTS=True and re-run process_folder.py).")
    elif not HAVE_PROJECT:
        print("[print] project definitions not importable -> keypoints only "
              "(run from the project root to also draw the measurement segments).")
    else:
        have = set(kp_names)
        for m in DEF_MEASUREMENT_NAMES:
            if m not in MEASUREMENT_INDICES or px_col(m) not in df.columns:
                continue
            kps = [KEYPOINT_NAMES[i] for i in MEASUREMENT_INDICES[m]]
            if len(kps) >= 2 and all(k in have for k in kps):
                meas.append((m, kps))
        print(f"[print] drawing {len(kp_names)} keypoints and "
              f"{len(meas)} measurement segments per image.")

    cols = {k: (c if c in df.columns else None) for k, c in PRINT_COLUMNS.items()}
    for key in ("scale_bar_box", "ruler_line", "ruler_orientation"):
        if cols[key] is None:
            print(f"[print] column '{PRINT_COLUMNS[key]}' not in the CSV -> "
                  f"{'scale-bar box' if key == 'scale_bar_box' else 'ruler line'} "
                  f"may not be drawn.")
    return {"kp_names": kp_names, "meas": meas, "cols": cols,
            "has_method": "scale_method" in df.columns, "labels": labels,
            "stats": Counter(), "bad_examples": {}}


def _get(row, cols, key):
    """Raw cell for a PRINT_COLUMNS key (None when the column is absent)."""
    return row.get(cols[key]) if cols[key] else None


def _record_bad(ctx, key, value):
    ctx["stats"][f"bad_{key}"] += 1
    ctx["bad_examples"].setdefault(key, value)


def annotate_image(img, row, ctx, sx=1.0, sy=1.0):
    """Draw scale detection, measurement segments, keypoints and a header."""
    from PIL import ImageDraw
    W, H = img.size
    Wc, Hc = W / sx, H / sy                             # frame of the CSV values
    base = max(W, H)
    lw = max(2, round(base / 500))                      # line width
    rad = max(3, round(base / 350))                     # keypoint radius
    fs = max(12, round(base / 70))                      # font size
    draw = ImageDraw.Draw(img, "RGBA")
    cols, stats = ctx["cols"], ctx["stats"]
    method = _norm_scale_method(row.get("scale_method")) if ctx["has_method"] else None
    if method is not None:
        stats[f"method_{method}"] += 1

    def box_label(xyxy, text, color, below=False):
        x1, y1, x2, y2 = xyxy[0] * sx, xyxy[1] * sy, xyxy[2] * sx, xyxy[3] * sy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        ty = y2 + 2 * lw if below else y1 - fs - 2 * lw
        _outlined_text(draw, (x1, ty), text, fs, fill=color, bounds=(W, H))

    # ---- scale bar: bar box + text (OCR) box ---------------------------------
    if method not in ("ruler", "none"):
        cell = _get(row, cols, "scale_bar_box")
        boxes = _parse_boxes(cell, Wc, Hc)
        if boxes is None:
            _record_bad(ctx, "scale_bar_box", cell)
        else:
            bar_conf = _num(_get(row, cols, "scale_bar_conf"))
            for xyxy, conf in boxes:
                c = conf if not math.isnan(conf) else bar_conf
                box_label(xyxy, "scale bar" + ("" if math.isnan(c) else f" {c:.2f}"),
                          PRINT_SCALE_COLOR, below=True)   # text box label goes above
            stats["drawn_scale_bar"] += bool(boxes)

        cell = _get(row, cols, "scale_text_box")
        boxes = _parse_boxes(cell, Wc, Hc)
        if boxes is None:
            _record_bad(ctx, "scale_text_box", cell)
        else:
            ocr = _get(row, cols, "scale_ocr_text")
            ocr = "" if ocr is None or (_is_num(ocr) and math.isnan(float(ocr))) else str(ocr).strip()
            for xyxy, _ in boxes:
                box_label(xyxy, f"text '{ocr}'" if ocr else "text", PRINT_TEXT_BOX_COLOR)

    # ---- ruler: detection line / column --------------------------------------
    if method not in ("scale_bar", "none"):
        cell, orient = _get(row, cols, "ruler_line"), _get(row, cols, "ruler_orientation")
        pts = _parse_ruler(cell, orient, Wc, Hc)
        if pts is None:
            _record_bad(ctx, "ruler_line", f"ruler_line={cell!r}, ruler_orientation={orient!r}")
        elif pts:
            pts = [(x * sx, y * sy) for x, y in pts]
            draw.line(pts, fill=(0, 0, 0), width=lw + 2)           # outline
            draw.line(pts, fill=PRINT_SCALE_COLOR, width=lw)
            rc = _num(_get(row, cols, "ruler_conf"))
            label = "ruler" + ("" if math.isnan(rc) else f" {rc:.2f}")
            _outlined_text(draw, (pts[0][0] + 2 * lw, pts[0][1] + 2 * lw), label, fs,
                           fill=PRINT_SCALE_COLOR, bounds=(W, H))
            stats["drawn_ruler"] += 1
        elif method == "ruler":
            stats["ruler_without_line"] += 1

    # ---- measurement segments (only measurements with a value in the CSV) ---
    for m, kps in ctx["meas"]:
        if math.isnan(_num(row.get(px_col(m)))):
            continue
        pts = [(_num(row.get(k + KP_X_SUFFIX)), _num(row.get(k + KP_Y_SUFFIX))) for k in kps]
        if any(math.isnan(x) or math.isnan(y) for x, y in pts):
            continue
        pts = [(x * sx, y * sy) for x, y in pts]
        color = _conf_rgb(_num(row.get(conf_col(m))))
        draw.line(pts, fill=(0, 0, 0), width=lw + 2, joint="curve")   # outline
        draw.line(pts, fill=color, width=lw, joint="curve")
        if ctx["labels"]:
            # label at the middle of the polyline (half its arc length)
            seglen = [math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])]
            half, k = sum(seglen) / 2.0, 0
            while k < len(seglen) - 1 and half > seglen[k]:
                half -= seglen[k]
                k += 1
            t = half / seglen[k] if seglen[k] else 0.0
            mx = pts[k][0] + t * (pts[k + 1][0] - pts[k][0])
            my = pts[k][1] + t * (pts[k + 1][1] - pts[k][1])
            mm = _num(row.get(mm_col(m)))
            val = f"{mm:.2f} mm" if not math.isnan(mm) else f"{_num(row.get(px_col(m))):.0f} px"
            _outlined_text(draw, (mx + rad, my + rad), f"{m}: {val}",
                           max(10, int(fs * 0.7)), fill=color, bounds=(W, H))

    # ---- keypoints -----------------------------------------------------------
    for k in ctx["kp_names"]:
        x, y = _num(row.get(k + KP_X_SUFFIX)), _num(row.get(k + KP_Y_SUFFIX))
        if math.isnan(x) or math.isnan(y):
            continue
        x, y = x * sx, y * sy
        draw.ellipse([x - rad, y - rad, x + rad, y + rad],
                     fill=_conf_rgb(_num(row.get(k + KP_CONF_SUFFIX))),
                     outline=(0, 0, 0), width=max(1, lw // 2))

    # ---- header: image name, confidences, scale ------------------------------
    pose = _num(row.get("overall_pose_confidence"))
    sc = _num(row.get("scale_confidence"))
    ppm = _num(row.get("scale_px_per_mm"))
    scale_txt = "scale:" + (f" {method}" if method else "")
    scale_txt += f"  {ppm:.2f} px/mm" if not math.isnan(ppm) else "  no px/mm"
    scale_txt += f"  (conf {sc:.2f})" if not math.isnan(sc) else ""
    lines = [(str(row.get("image_name", "")), (255, 255, 255)),
             ("pose conf: " + (f"{pose:.2f}" if not math.isnan(pose) else "no pose detected"),
              (255, 255, 255)),
             (scale_txt, (255, 255, 255)),
             ("colour = confidence (red 0 -> green 1)", (200, 200, 200))]

    font = _print_font(fs)
    pad, step = fs // 2, int(fs * 1.25)
    width = max(draw.textbbox((0, 0), t, font=font)[2] for t, _ in lines)
    draw.rectangle([0, 0, width + 2 * pad, len(lines) * step + 2 * pad], fill=(0, 0, 0, 150))
    for i, (t, c) in enumerate(lines):
        _outlined_text(draw, (pad, pad + i * step), t, fs, fill=c)
    return img


def _build_name_index(images_dir: Path, exclude: Path | None) -> dict[str, list[Path]]:
    """{base name (lowercase) -> [image paths]} over images_dir, recursively."""
    index: dict[str, list[Path]] = {}
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if exclude is not None and p.resolve().is_relative_to(exclude):
            continue                                  # our own annotated copies
        index.setdefault(p.name.lower(), []).append(p)
    return index


def _find_image(name: str, images_dir: Path, index) -> Path | None:
    """CSV image_name -> file: absolute path, path under images_dir, base name."""
    p = Path(name.replace("\\", "/"))
    if p.is_absolute() and p.is_file():
        return p
    if (images_dir / p).is_file():
        return images_dir / p
    hits = index.get(p.name.lower())
    return hits[0] if hits else None


def _print_scale_report(ctx):
    """Console summary of what was drawn for the scale (helps spot a bad format)."""
    s = ctx["stats"]
    methods = {k[len("method_"):]: v for k, v in s.items() if k.startswith("method_")}
    if methods:
        print("[print] scale_method: " + ", ".join(f"{k}={v}" for k, v in sorted(methods.items())))
    print(f"[print] drawn: scale-bar box on {s['drawn_scale_bar']} image(s), "
          f"ruler line on {s['drawn_ruler']} image(s)")
    if s["ruler_without_line"]:
        print(f"[print] {s['ruler_without_line']} image(s) with scale_method=ruler but an "
              f"empty '{PRINT_COLUMNS['ruler_line']}' cell.")
    for key, example in ctx["bad_examples"].items():
        print(f"[print] {s[f'bad_{key}']} '{key}' value(s) not understood -> not drawn. "
              f"Example: {str(example)[:200]}")


def print_images(df, images_dir, out_dir, labels=False):
    """Write an annotated copy of every CSV image into out_dir (see above)."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("[print] Pillow is not installed (pip install pillow) -> skipped.")
        return
    if "image_name" not in df.columns:
        print("[print] no 'image_name' column in the CSV -> skipped.")
        return
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    if not images_dir.is_dir():
        print(f"[print] images folder not found: {images_dir} -> skipped.")
        return
    src_root, dst_root = images_dir.resolve(), out_dir.resolve()
    if dst_root == src_root or src_root.is_relative_to(dst_root):
        print(f"[print] the print folder ({out_dir}) must not be the images folder "
              f"or one of its parents -> skipped (originals left untouched).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = _print_context(df, labels)
    index = _build_name_index(images_dir, exclude=dst_root)
    ambiguous = sum(1 for v in index.values() if len(v) > 1)
    if ambiguous:
        print(f"[print] {ambiguous} base name(s) found several times under "
              f"{images_dir} -> the first one (sorted) is used for those.")

    rows = df.to_dict("records")
    written, missing, failed, used = 0, [], [], set()
    for i, row in enumerate(rows, 1):
        name = row.get("image_name")
        if not isinstance(name, str) or not name.strip():
            continue
        src = _find_image(name, images_dir, index)
        if src is None:
            missing.append(name)
            continue
        ext = src.suffix.lower() if src.suffix.lower() in PRINT_SAVE_EXTS else ".png"
        stem, dst = src.stem, out_dir / f"{src.stem}{ext}"
        k = 1
        while dst.name.lower() in used:               # same base name twice
            dst = out_dir / f"{stem}_{k}{ext}"
            k += 1
        used.add(dst.name.lower())
        try:
            img, sx, sy = _load_image_for_row(src, row)
            annotate_image(img, row, ctx, sx, sy)
            params = {"quality": 92} if ext in (".jpg", ".jpeg", ".webp") else {}
            img.save(dst, **params)
            written += 1
        except Exception as exc:                        # noqa: BLE001
            failed.append(f"{name} ({exc})")
        if i % 200 == 0:
            print(f"[print] {i}/{len(rows)} rows processed...")

    print(f"[print] {written} annotated image(s) written to {out_dir}")
    _print_scale_report(ctx)
    if missing:
        print(f"[print] {len(missing)} image(s) of the CSV not found under {images_dir}"
              f" (e.g. {', '.join(missing[:3])})")
    if failed:
        print(f"[print] {len(failed)} image(s) failed: {'; '.join(failed[:3])}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Make figures from a results CSV.")
    p.add_argument("--input", default="results.csv", help="Results CSV path.")
    p.add_argument("--output-dir", default="analysis_output", help="Where to write PNG/TXT.")
    p.add_argument("--review-threshold", type=float, default=0.5,
                   help="Confidence below which a row counts as 'needs review'.")
    p.add_argument("--annotations", default="annotations.json",
                   help="JSON of manual scale-bar annotations (from annotate_gui.py). "
                        "Used for the scale_method confusion matrix.")
    p.add_argument("--datasets-root", default=None,
                   help="Root of the YOLO datasets (default: the project's "
                        "config.DATASETS_ROOT). Used to find the GT label files.")
    p.add_argument("--gt-splits", nargs="*", default=["train", "val", "test"],
                   help="Splits to read GT labels from (default: train val test).")
    p.add_argument("--oks-kappa", type=float, default=0.05,
                   help="OKS falloff constant (uncalibrated; default 0.05).")
    p.add_argument("--pck-alpha", type=float, default=0.10,
                   help="PCK threshold as a fraction of object scale (default 0.10).")
    p.add_argument("--no-gt", action="store_true",
                   help="Skip the ground-truth error analysis entirely.")
    p.add_argument("--print", dest="print_images", action="store_true",
                   help="Also write a copy of every image of the CSV with the pose "
                        "keypoints, the measurement segments and the scale detection "
                        "(scale-bar boxes or ruler row/column) drawn on it.")
    p.add_argument("--images-dir", default=None,
                   help="Folder of the analysed images (the one given to "
                        "process_folder.py, searched recursively). Required with --print.")
    p.add_argument("--print-dir", default=None,
                   help="Where to write the annotated copies "
                        "(default: <output-dir>/annotated_images).")
    p.add_argument("--print-labels", action="store_true",
                   help="With --print, also write each measurement's name and value "
                        "next to its segment.")
    args = p.parse_args()
    if args.print_images and not args.images_dir:
        p.error("--print needs --images-dir (the folder the CSV was computed on).")

    os.makedirs(args.output_dir, exist_ok=True)
    df = read_results(args.input)
    conf_cols = measurement_conf_columns(df)
    print(f"[data] {len(df)} rows, {len(conf_cols)} measurement-confidence columns")

    # Always-available figures.
    if conf_cols:
        fig_boxplot(df, conf_cols, args.output_dir)
        fig_mean_std(df, conf_cols, args.output_dir)
        fig_heatmap(df, conf_cols, args.output_dir)
        fig_conf_correlation(df, conf_cols, args.output_dir)
        fig_cumulative(df, conf_cols, args.output_dir)
    fig_hist(_series(df, "overall_pose_confidence"),
             "Overall pose confidence", "hist_overall_pose_confidence.png", args.output_dir)
    fig_hist(_series(df, "scale_confidence"),
             "Scale confidence", "hist_scale_confidence.png", args.output_dir)
    fig_scatter_pose_scale(df, args.output_dir)
    needs_review_pct = fig_needs_review(df, args.output_dir, args.review_threshold)

    # Extra requested figures.
    scale_stats = fig_scale_distribution(df, args.output_dir)
    missing_rates, missing_denom = fig_missing_rate(df, args.output_dir)
    fig_lr_symmetry(df, args.output_dir)

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
    scale_cm = fig_scale_method_confusion(df, args.output_dir, args.annotations)

    # ----- ground-truth error analysis --------------------------------------
    err, err_corr = None, None
    if args.no_gt:
        print("[gt] skipped (--no-gt).")
    elif not HAVE_PROJECT:
        print("[gt] skipped: could not import the project (run from the project root).")
    else:
        datasets_root = Path(args.datasets_root) if args.datasets_root else proj_config.DATASETS_ROOT
        print(f"[gt] reading labels under {datasets_root} (splits: {args.gt_splits})")
        err = compute_errors(df, datasets_root, args.gt_splits, args.review_threshold,
                             oks_kappa=args.oks_kappa, pck_alpha=args.pck_alpha)
        if err is None:
            print("[gt] no label files found -> GT figures skipped.")
        elif err["n_gt"] == 0:
            print("[gt] labels found but no CSV image matched them -> GT figures skipped.")
            err = None
        else:
            print(f"[gt] matched {err['n_gt']} images to a GT label.")
            err_corr = fig_error_vs_conf_correlation(err, args.output_dir)
            auc_stats = fig_auc_min_aggregation(err, args.output_dir, tol=0.1)
            roc_stats = fig_roc_grid(err, args.output_dir, tol=0.1,
                                     score_key="minconf", order="auc")
            fig_error_vs_conf_scatter(err, args.output_dir)
            fig_mean_error_vs_needs_review(err, args.output_dir)
            fig_rel_error_boxplot(err, args.output_dir)
            fig_error_by_split(err, args.output_dir)
            # keypoint-level figures (only if the raw kp columns are present)
            if err["kp_names"] and err["img_oks"].size:
                oks_corr = fig_oks_vs_overall_conf(err, args.output_dir)
                kp_corr = fig_kp_error_conf_correlation(err, args.output_dir)
                fig_kp_error_conf_heatmap(err, args.output_dir)
                fig_oks_histogram(err, args.output_dir)
                fig_kp_mean_error(err, args.output_dir)
            else:
                print("[gt] no keypoint columns in CSV -> keypoint figures skipped "
                      "(set EXPORT_KEYPOINTS=True and re-run to unlock OKS/kp metrics).")

    write_summary(df, conf_cols, args.output_dir, args.review_threshold,
                  needs_review_pct, scale_type_pct,
                  scale_stats=scale_stats, missing_rates=missing_rates,
                  missing_denom=missing_denom, err=err, err_corr=err_corr,
                  scale_cm=scale_cm)

    if args.print_images:
        print_dir = args.print_dir or os.path.join(args.output_dir, "annotated_images")
        print_images(df, args.images_dir, print_dir, labels=args.print_labels)
    print(f"\nDone. Figures and summary in: {args.output_dir}")


if __name__ == "__main__":
    main()