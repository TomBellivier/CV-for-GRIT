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

A note on "per keypoint"
------------------------
The CSV stores confidences PER MEASUREMENT (one value per measured distance),
not per keypoint, so the box plot / heatmap are per measurement. Getting true
per-keypoint figures would mean adding keypoint-confidence columns to the export
(a small change, but it means re-running the folder).
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    per = {m: {"rel": [], "abs": [], "conf": []} for m in meas_names}
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
    print(f"\nDone. Figures and summary in: {args.output_dir}")


if __name__ == "__main__":
    main()