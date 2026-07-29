#!/usr/bin/env python3
"""
crossval.py
===========

Terminal entry point for a stratified 5-fold cross-validation of a YOLO pose
model on a multi-group insect keypoint dataset (42 keypoints).

Expected dataset layout (per insect group)::

    <data-root>/<Group>/images/<train|val|test>/*.jpg      (or 'image/'  – auto-detected)
    <data-root>/<Group>/labels/<train|val|test>/*.txt      (or 'label/'  – auto-detected)

The train/val/test splits are pooled and re-partitioned into k folds, stratified
by group. A fresh model is trained per fold, then evaluated with native pose mAP
plus custom OKS and PCKh@0.5 (head size = distance between keypoints 1 and 2).

Example
-------
    python crossval.py \
        --data-root /path/to/datasets \
        --hyp data.yaml \
        --output ./cv_results \
        --model yolo26s-pose.yaml \
        --epochs 100 --imgsz 640 --device 0

Outputs (in --output):
    folds_assignment.csv         which image went to which fold
    folds/fold_k/{train,val}.txt image lists per fold (reproducible)
    summary_metrics.csv          per-fold mAP / OKS / PCKh + mean/std
    per_keypoint_pck.csv         PCKh per keypoint per fold + mean/std
    per_keypoint_oks.csv         OKS per keypoint per fold + mean/std
    per_keypoint_metrics.png     single bar plot (PCKh + OKS) with error bars
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from crossval_lib import CVConfig, run_cross_validation

DEFAULT_GROUPS = ["Coleoptera", "Diptera", "Hymenoptera", "Lepidoptera"]

# Training keys that must NOT be forwarded from the hyp file as-is
# (they are controlled by dedicated CLI flags instead).
_RESERVED_HYP_KEYS = {"epochs", "imgsz", "device", "seed", "data", "model",
                      "project", "name", "pretrained", "save"}

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stratified k-fold cross-validation for a YOLO pose model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # paths
    p.add_argument("--data-root", required=True, type=Path,
                   help="Root folder containing the per-group dataset folders.")
    p.add_argument("--hyp", required=True, type=Path,
                   help="YAML file with training hyper-parameters (lr, augmentation...).")
    p.add_argument("--output", default=Path("./cv_results"), type=Path,
                   help="Directory where all reports/plots/manifests are written.")
    p.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                   help="Insect group sub-folder names (used for stratification).")

    # model / training
    p.add_argument("--model", default="yolo26s-pose.yaml",
                   help="Model spec. Use a '.yaml' to train from scratch (no weights).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0",
                   help="'0' for first GPU, or 'cpu' to force CPU.")
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds (k).")
    p.add_argument("--seed", type=int, default=42, help="Seed for fold split + training.")

    # keypoint geometry
    p.add_argument("--n-kpt", type=int, default=42)
    p.add_argument("--kpt-dim", type=int, default=3,
                   help="3 => (x, y, visibility). Visibility 0 keypoints are ignored.")
    p.add_argument("--flip-idx", type=Path, default=None,
                   help="Optional JSON file with the left/right keypoint swap list. "
                        "Required to enable horizontal flip augmentation safely.")

    # metric parameters
    p.add_argument("--head-ids", type=int, nargs=2, default=(1, 2),
                   metavar=("HEAD_LEFT", "HEAD_RIGHT"),
                   help="0-based keypoint ids used to compute the head size for PCKh.")
    p.add_argument("--pck-alpha", type=float, default=0.5,
                   help="PCKh threshold as a fraction of the head size.")
    p.add_argument("--oks-sigma", type=float, default=0.05,
                   help="Constant OKS falloff sigma (no per-keypoint sigmas assumed).")
    p.add_argument("--iou-match-thr", type=float, default=0.5,
                   help="IoU threshold to match predicted and ground-truth instances.")
    return p


def load_hyp(path: Path) -> dict:
    """Load the hyper-parameter yaml and drop keys reserved for CLI flags."""
    with open(path) as f:
        hyp = yaml.safe_load(f) or {}
    return {k: v for k, v in hyp.items() if k not in _RESERVED_HYP_KEYS}


def load_flip_idx(path: Path | None, n_kpt: int) -> list[int] | None:
    """Load and validate the left/right keypoint swap map, if provided."""
    if path is None:
        return None
    flip = json.loads(Path(path).read_text())
    if len(flip) != n_kpt or sorted(flip) != list(range(n_kpt)):
        raise ValueError(
            f"flip_idx must be a permutation of 0..{n_kpt - 1} (got length {len(flip)}).")
    return list(flip)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output / "crossval.log", mode="w"),
        ],
    )
    log = logging.getLogger("crossval")

    hyp = load_hyp(args.hyp)
    flip_idx = load_flip_idx(args.flip_idx, args.n_kpt)

    # Horizontal flip on pose data is only valid with a correct left/right swap map.
    # Without one, force fliplr = 0 to avoid silently scrambling the keypoints.
    if flip_idx is None and float(hyp.get("fliplr", 0.0)) > 0.0:
        log.warning("fliplr=%.2f requested but no --flip-idx provided: forcing fliplr=0. "
                    "Provide a flip_idx JSON (a 0..%d permutation) to enable it.",
                    hyp["fliplr"], args.n_kpt - 1)
        hyp["fliplr"] = 0.0

    cfg = CVConfig(
        data_root=args.data_root,
        output_dir=args.output,
        groups=args.groups,
        hyp=hyp,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=str(args.device),
        n_splits=args.folds,
        seed=args.seed,
        n_kpt=args.n_kpt,
        kpt_dim=args.kpt_dim,
        flip_idx=flip_idx,
        head_ids=tuple(args.head_ids),
        pck_alpha=args.pck_alpha,
        oks_sigma=args.oks_sigma,
        iou_match_thr=args.iou_match_thr,
    )

    log.info("Config: %d-fold, seed=%d, model=%s, epochs=%d, imgsz=%d, device=%s",
             cfg.n_splits, cfg.seed, cfg.model, cfg.epochs, cfg.imgsz, cfg.device)
    run_cross_validation(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())