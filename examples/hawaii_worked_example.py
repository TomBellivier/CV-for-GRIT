"""Worked example of the manuscript (Appendix S7): elytra of held-out Hawaiian beetles.

EXTRAITS is run on the images of the Imageomics Hawaii-beetles dataset that were never
annotated (hence never used for training), and its fore-wing measurements are compared
with the elytra measurements of the dataset, made by hand with TORAS.

Correspondence of the definitions (checked by `defcheck` on the annotated images):
    elytral length  <->  mean of the left and right fore wing lengths
    elytral width   <->  sum of the left and right fore wing widths

The individual images carry no scale: the scale bar is on the group image the specimen
was cropped from, without resizing. The reference scale of the dataset (`px_scalebar`,
1 cm) therefore converts both EXTRAITS and the reference to millimetres, and the
pipeline runs with `--scale none`.

Run from the repository root:

    python examples/hawaii_worked_example.py defcheck
    python examples/hawaii_worked_example.py prepare [--limit N]
    python pipeline/process_folder.py --input "annotated_images/full databases/coleoptera/hawaii_heldout" \
        --output results/worked_example/hawaii_heldout.csv --scale none
    python examples/hawaii_worked_example.py analyse

Outputs go to results/worked_example/.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from kp_infos import MEAS_TO_KP  # noqa: E402

REPO_ID = "imageomics/Hawaii-beetles"
ANNOTATION_CSV = REPO_ROOT / "annotation_data" / "annotation_data.csv"
IMAGE_DIR = REPO_ROOT / "annotated_images" / "full databases" / "coleoptera" / "hawaii_heldout"
OUT_DIR = REPO_ROOT / "results" / "worked_example"
REFERENCE_CSV = OUT_DIR / "hawaii_reference.csv"
PIPELINE_CSV = OUT_DIR / "hawaii_heldout.csv"

# IMG_0093_specimen_1_MECKON_NEON.BET.D20.000001.png -> group image, position, individualID
NAME_PATTERN = re.compile(r"^(IMG_\d+)_specimen_(\d+)_[A-Z]+_(.+)\.png$")

# (trait, EXTRAITS columns, how they are combined, reference column in cm)
TRAITS = [
    ("Elytral length", ["left fore wing length", "right fore wing length"], "mean",
     "cm_elytra_max_length"),
    ("Elytral width", ["left fore wing width", "right fore wing width"], "sum",
     "cm_elytra_max_width"),
]


def image_key(names: pd.Series) -> pd.Series:
    parts = names.str.extract(NAME_PATTERN)
    return parts[0] + "|" + parts[1] + "|" + parts[2]


def load_reference() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    ref = pd.read_csv(hf_hub_download(REPO_ID, "trait_annotations.csv", repo_type="dataset"))
    group_image = ref.groupImageFilePath.str.extract(r"([^/]+)\.png$")[0]
    ref["key"] = group_image + "|" + ref.BeetlePosition.astype(str) + "|" + ref.individualID
    ref["px_per_mm"] = ref.px_scalebar / (10.0 * ref.cm_scalebar)
    return ref


def annotated_hawaii() -> pd.DataFrame:
    ann = pd.read_csv(ANNOTATION_CSV, low_memory=False)
    ann = ann[(ann.group == "coleoptera") & ann.image_name.str.match(NAME_PATTERN)].copy()
    ann["key"] = image_key(ann.image_name)
    return ann


def polyline_length(frame: pd.DataFrame, measure: str) -> pd.Series:
    kps = MEAS_TO_KP[measure]
    total = 0.0
    for p, q in zip(kps[:-1], kps[1:]):
        total = total + np.hypot(frame[f"{p}_x"] - frame[f"{q}_x"],
                                 frame[f"{p}_y"] - frame[f"{q}_y"])
    return total


def combine(frame: pd.DataFrame, columns: list[str], how: str) -> pd.Series:
    values = frame[columns]
    return values.mean(axis=1) if how == "mean" else values.sum(axis=1, min_count=len(columns))


def agreement(pred: pd.Series, ref: pd.Series) -> dict:
    ok = pred.notna() & ref.notna() & (pred > 0) & (ref > 0)
    pred, ref = pred[ok], ref[ok]
    rel = (pred - ref).abs() / ref
    return {
        "N": int(ok.sum()),
        "MAE (mm)": float((pred - ref).abs().mean()),
        "median relative error (%)": float(100 * rel.median()),
        "Pearson r": float(np.corrcoef(pred, ref)[0, 1]) if ok.sum() > 2 else np.nan,
        "bias (mm)": float((pred - ref).mean()),
    }


# --------------------------------------------------------------------------- #
def cmd_defcheck(_args) -> None:
    """Annotated keypoints vs reference, in pixels: do the definitions match?"""
    ref = load_reference()
    ann = annotated_hawaii()
    measures = [m for _, columns, _, _ in TRAITS for m in columns]
    measures += ["abdomen length", "abdomen width", "thorax width"]
    ann = pd.concat([ann, pd.DataFrame({m: polyline_length(ann, m) for m in measures})], axis=1)
    joined = ann.merge(ref, on="key")
    candidates = [
        ("fore wing length, mean L/R", combine(joined, TRAITS[0][1], "mean"), "px_elytra_max_length"),
        ("abdomen length", joined["abdomen length"], "px_elytra_max_length"),
        ("fore wing width, sum L+R", combine(joined, TRAITS[1][1], "sum"), "px_elytra_max_width"),
        ("abdomen width", joined["abdomen width"], "px_elytra_max_width"),
        ("thorax width", joined["thorax width"], "px_basal_pronotum_width"),
    ]
    rows = []
    for name, values, ref_col in candidates:
        ok = values.notna() & (values > 0) & (joined[ref_col] > 0)
        ratio = values[ok] / joined.loc[ok, ref_col]
        rows.append({"EXTRAITS measurement": name, "reference": ref_col, "N": int(ok.sum()),
                     "median ratio": ratio.median(),
                     "Pearson r": np.corrcoef(values[ok], joined.loc[ok, ref_col])[0, 1],
                     "median |ratio - 1|": (ratio - 1).abs().median()})
    print(f"annotated Hawaii images: {len(ann)}, matched to the reference: {len(joined)}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"held-out individuals (never annotated): {len(set(ref.key) - set(ann.key))}")


def cmd_prepare(args) -> None:
    """Download the held-out individual images and write the reference table."""
    from huggingface_hub import HfApi, hf_hub_download

    ref = load_reference()
    annotated = set(annotated_hawaii().key)
    files = [f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
             if f.startswith("individual_specimens/") and f.endswith(".png")]
    names = pd.Series([Path(f).name for f in files])
    table = pd.DataFrame({"repo_path": files, "image_name": names, "key": image_key(names)})
    table = table[table.key.notna() & ~table.key.isin(annotated)]
    table = table.merge(ref[["key", "px_per_mm", "cm_elytra_max_length", "cm_elytra_max_width"]],
                        on="key")
    table = table.sort_values("image_name")
    if args.limit:
        table = table.head(args.limit)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(table.repo_path, 1):
        target = IMAGE_DIR / Path(path).name
        if not target.exists():
            local = hf_hub_download(REPO_ID, path, repo_type="dataset")
            target.write_bytes(Path(local).read_bytes())
        if i % 50 == 0:
            print(f"{i}/{len(table)} images")
    table.drop(columns="repo_path").to_csv(REFERENCE_CSV, index=False)
    print(f"{len(table)} held-out images in {IMAGE_DIR}\nreference table: {REFERENCE_CSV}")


def cmd_analyse(args) -> None:
    """Agreement statistics, output excerpt and Figure 6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pd.read_csv(args.results)
    ref = pd.read_csv(REFERENCE_CSV)
    data = out.merge(ref, on="image_name")
    print(f"pipeline rows: {len(out)}, matched to the reference: {len(data)}")

    rows, points = [], {}
    for trait, columns, how, ref_col in TRAITS:
        px = data[[c + " [px]" for c in columns]]
        valid = data[[c + " [valid_pred]" for c in columns]]
        reference = 10.0 * data[ref_col]
        for label, kept in (("unfiltered", px), ("filtered", px.where(valid.to_numpy() == 1))):
            if how == "sum":   # a width is only defined when both elytra are kept
                pred_px = kept.sum(axis=1, min_count=len(columns))
            else:
                pred_px = kept.mean(axis=1)
            pred = pred_px / data.px_per_mm
            rows.append({"trait": trait, "filter": label, **agreement(pred, reference)})
            points[(trait, label)] = (reference, pred)

    stats = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_DIR / "agreement.csv", index=False)
    print(stats.round(3).to_string(index=False))

    excerpt_cols = ["image_name", "left fore wing length [px]", "right fore wing length [px]",
                    "left fore wing length [valid_pred]", "right fore wing length [valid_pred]",
                    "overall_pose_confidence"]
    print("\nOutput excerpt:")
    print(out[[c for c in excerpt_cols if c in out.columns]].head(5).round(3).to_string(index=False))

    # Figure 6: (a) EXTRAITS vs reference, (b) Bland-Altman, filtered measurements.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {"Elytral length": "#1f77b4", "Elytral width": "#d62728"}
    lim = 0.0
    for trait, _, _, _ in TRAITS:
        ref_mm, pred_mm = points[(trait, "filtered")]
        ok = ref_mm.notna() & pred_mm.notna()
        ref_mm, pred_mm = ref_mm[ok], pred_mm[ok]
        ax_a.scatter(ref_mm, pred_mm, s=10, alpha=0.6, color=colors[trait], label=trait)
        lim = max(lim, ref_mm.max(), pred_mm.max())
        mean, diff = (ref_mm + pred_mm) / 2, pred_mm - ref_mm
        bias, sd = diff.mean(), diff.std()
        ax_b.scatter(mean, diff, s=10, alpha=0.6, color=colors[trait], label=trait)
        for y, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            ax_b.axhline(y, color=colors[trait], linestyle=style, linewidth=1)
    ax_a.plot([0, lim * 1.05], [0, lim * 1.05], color="black", linewidth=1)
    ax_a.set(xlabel="Reference (mm)", ylabel="EXTRAITS (mm)", title="(a)")
    ax_b.axhline(0, color="grey", linewidth=0.5)
    ax_b.set(xlabel="Mean of EXTRAITS and reference (mm)",
             ylabel="EXTRAITS - reference (mm)", title="(b)")
    ax_a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure6_hawaii.png", dpi=300)
    print(f"\nwritten: {OUT_DIR / 'agreement.csv'}, {OUT_DIR / 'figure6_hawaii.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("defcheck", help="check the definitions on the annotated images")
    prepare = sub.add_parser("prepare", help="download the held-out images")
    prepare.add_argument("--limit", type=int, default=None, help="first N images only")
    analyse = sub.add_parser("analyse", help="statistics and Figure 6")
    analyse.add_argument("--results", type=Path, default=PIPELINE_CSV)
    args = parser.parse_args()
    {"defcheck": cmd_defcheck, "prepare": cmd_prepare, "analyse": cmd_analyse}[args.command](args)


if __name__ == "__main__":
    main()
