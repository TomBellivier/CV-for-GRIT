"""
crossval_lib.py
===============

Reusable utilities to evaluate a YOLO *pose* model with a stratified k-fold
cross-validation, for a multi-group insect keypoint dataset (42 keypoints).

Design notes
------------
* A **single global model** is trained per fold on the union of the other folds,
  then evaluated on the held-out fold.  Folds are **stratified by insect group**
  (Coleoptera / Diptera / Hymenoptera / Lepidoptera) so every fold keeps the same
  group proportions.
* The original train/val/test splits of every group are **pooled** together, then
  re-partitioned into `n_splits` folds (this is a true k-fold).
* Metrics reported: native Ultralytics **mAP** (pose, OKS-based), a custom
  **OKS** (mean + per-keypoint), and **PCKh@0.5** where the head size is the
  distance between keypoints "head-left" (id 1) and "head-right" (id 2).
* Keypoints flagged as not-labelled (visibility == 0) are **ignored** in the
  OKS and PCK computations.

Heavy dependencies (`ultralytics`, `torch`) are imported lazily inside the
functions that need them, so the metric/plotting helpers can be imported and
unit-tested with numpy only.
"""

from __future__ import annotations

import csv
import gc
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

logger = logging.getLogger("crossval")


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------
@dataclass
class CVConfig:
    """All parameters needed to run the cross-validation."""

    data_root: Path                              # root holding the group folders
    output_dir: Path                             # where reports / plots / manifests go
    groups: Sequence[str]                        # insect group sub-folder names
    hyp: Dict                                     # training hyper-parameters (from the yaml)

    # model / training
    model: str = "yolo26s-pose.yaml"             # ".yaml" => built from scratch (no pretrained weights)
    epochs: int = 100
    imgsz: int = 640
    device: str = "0"                            # "0" = first GPU, "cpu" = force CPU
    n_splits: int = 5
    seed: int = 42

    # keypoint geometry
    n_kpt: int = 42
    kpt_dim: int = 3                             # 3 => (x, y, visibility)
    names: Dict[int, str] = field(default_factory=lambda: {0: "insect"})
    flip_idx: Optional[List[int]] = None         # left/right swap map; None => flip disabled

    # metric parameters
    head_ids: Tuple[int, int] = (1, 2)           # 0-based ids of head-left / head-right
    pck_alpha: float = 0.5                        # PCKh threshold = alpha * head_size
    oks_sigma: float = 0.05                       # constant OKS falloff (no per-kpt sigmas known)
    iou_match_thr: float = 0.5                    # IoU threshold for GT<->pred instance matching

    def sigmas(self) -> np.ndarray:
        """Per-keypoint OKS sigma vector (constant by default)."""
        return np.full(self.n_kpt, self.oks_sigma, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
@dataclass
class Item:
    """A single (image, label) sample together with its stratification group."""

    image: Path
    label: Path
    group: str
    split: str                                   # original split it came from (train/val/test)
    pooled_image: Optional[Path] = None          # symlink path when a pool is built


_STD_LAYOUT = ("images", "labels")               # canonical Ultralytics folder names
_ALT_LAYOUT = ("image", "label")                 # the singular naming used in this dataset
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _detect_layout(group_root: Path) -> Tuple[str, str, bool]:
    """
    Return (image_dir_name, label_dir_name, is_standard) for one group folder.

    `is_standard` is True when the folders are literally ``images``/``labels``
    (Ultralytics can then resolve labels by itself and no symlinks are needed).
    """
    for img_name, lbl_name in (_STD_LAYOUT, _ALT_LAYOUT):
        if (group_root / img_name).is_dir() and (group_root / lbl_name).is_dir():
            return img_name, lbl_name, (img_name, lbl_name) == _STD_LAYOUT
    raise FileNotFoundError(
        f"No image/label folders found under {group_root}. "
        f"Expected one of {_STD_LAYOUT} or {_ALT_LAYOUT}."
    )


def discover_items(cfg: CVConfig) -> Tuple[List[Item], bool]:
    """
    Walk every group folder, pool train/val/test, and return the list of samples.

    Returns
    -------
    items : list[Item]
    all_standard : bool
        True when *all* groups use the canonical ``images``/``labels`` layout, in
        which case no symlink pool is required.
    """
    items: List[Item] = []
    all_standard = True

    for group in cfg.groups:
        group_root = cfg.data_root / group
        if not group_root.is_dir():
            raise FileNotFoundError(f"Group folder missing: {group_root}")

        img_name, lbl_name, is_std = _detect_layout(group_root)
        all_standard = all_standard and is_std
        logger.info("Group %-13s layout: %s/%s (standard=%s)",
                    group, img_name, lbl_name, is_std)

        img_root = group_root / img_name
        lbl_root = group_root / lbl_name

        # Pool all original splits together (true k-fold => splits are merged).
        for split in ("train", "val", "test"):
            split_dir = img_root / split
            if not split_dir.is_dir():
                continue
            for img_path in sorted(split_dir.iterdir()):
                if img_path.suffix.lower() not in _IMG_EXT:
                    continue
                label_path = lbl_root / split / (img_path.stem + ".txt")
                if not label_path.exists():
                    logger.warning("No label for %s -> skipped", img_path)
                    continue
                items.append(Item(image=img_path, label=label_path,
                                  group=group, split=split))

    if not items:
        raise RuntimeError("No (image, label) pairs discovered. Check --data-root and layout.")

    logger.info("Discovered %d samples across %d groups.", len(items), len(cfg.groups))
    return items, all_standard


# ---------------------------------------------------------------------------
# Stratified fold assignment
# ---------------------------------------------------------------------------
def assign_folds(items: List[Item], cfg: CVConfig) -> np.ndarray:
    """
    Assign every item to a fold using StratifiedKFold on the insect group.

    Returns an int array `fold_of[i]` in [0, n_splits).
    """
    from sklearn.model_selection import StratifiedKFold

    groups = np.array([it.group for it in items])
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_of = np.full(len(items), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(items)), groups)):
        fold_of[val_idx] = fold_idx  # an item's fold == the fold where it is the validation set

    assert (fold_of >= 0).all(), "Some items were not assigned to a fold."
    return fold_of


# ---------------------------------------------------------------------------
# Building the data given to Ultralytics (txt lists + per-fold yaml)
# ---------------------------------------------------------------------------
def build_symlink_pool(items: List[Item], pool_dir: Path) -> None:
    """
    Create a symlink pool with canonical ``images``/``labels`` naming.

    Only needed when the source dataset uses non-standard folder names, because
    Ultralytics resolves a label path by replacing ``/images/`` with ``/labels/``.
    Symlinks are *pointers* (a few bytes each): no image data is duplicated.
    Filenames are prefixed by ``group__split__`` to avoid any collision.
    """
    img_pool = pool_dir / "images"
    lbl_pool = pool_dir / "labels"
    img_pool.mkdir(parents=True, exist_ok=True)
    lbl_pool.mkdir(parents=True, exist_ok=True)

    for it in items:
        stem = f"{it.group}__{it.split}__{it.image.stem}"
        img_link = img_pool / (stem + it.image.suffix)
        lbl_link = lbl_pool / (stem + ".txt")
        for link, target in ((img_link, it.image), (lbl_link, it.label)):
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.resolve())
        it.pooled_image = img_link

    logger.info("Symlink pool built at %s (%d images).", pool_dir, len(items))


def write_fold_yaml(cfg: CVConfig, fold_idx: int, fold_of: np.ndarray,
                    items: List[Item], use_pool: bool) -> Tuple[Path, List[Item]]:
    """
    Write the train/val image lists and the dataset yaml for a given fold.

    The image paths used are the pooled symlinks when `use_pool` is True, else the
    original absolute image paths.  Returns (yaml_path, validation_items).
    """
    fold_dir = cfg.output_dir / "folds" / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    def path_of(it: Item) -> str:
        p = it.pooled_image if use_pool else it.image
        return str(Path(p).resolve())

    train_items = [it for i, it in enumerate(items) if fold_of[i] != fold_idx]
    val_items = [it for i, it in enumerate(items) if fold_of[i] == fold_idx]

    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    train_txt.write_text("\n".join(path_of(it) for it in train_items) + "\n")
    val_txt.write_text("\n".join(path_of(it) for it in val_items) + "\n")

    data = {
        "path": str(fold_dir.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "kpt_shape": [cfg.n_kpt, cfg.kpt_dim],
        "names": {int(k): v for k, v in cfg.names.items()},
    }
    if cfg.flip_idx is not None:
        data["flip_idx"] = list(cfg.flip_idx)

    yaml_path = fold_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    logger.info("Fold %d: %d train / %d val -> %s",
                fold_idx, len(train_items), len(val_items), yaml_path)
    return yaml_path, val_items


def detect_class_names(items: List[Item], cfg: CVConfig) -> Dict[int, str]:
    """Scan label files to build a {class_id: name} mapping (best effort)."""
    class_ids = set()
    for it in items[: min(len(items), 500)]:  # sampling is enough to find the classes
        for line in it.label.read_text().splitlines():
            parts = line.split()
            if parts:
                class_ids.add(int(float(parts[0])))
    if not class_ids:
        return dict(cfg.names)
    if class_ids == {0}:
        return {0: "insect"}
    return {cid: f"class_{cid}" for cid in sorted(class_ids)}


# ---------------------------------------------------------------------------
# Label / prediction parsing
# ---------------------------------------------------------------------------
def image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) without decoding the whole image when possible."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        import cv2  # noqa: WPS433 (fallback only)
        arr = cv2.imread(str(path))
        h, w = arr.shape[:2]
        return w, h


def parse_label_file(label_path: Path, n_kpt: int, kpt_dim: int,
                     w: int, h: int) -> List[Dict]:
    """
    Parse a YOLO-pose label file into a list of ground-truth instances.

    Each line: cls cx cy bw bh (px1 py1 [v1]) ... (normalized 0..1).
    Coordinates are de-normalized to pixels here.
    """
    instances: List[Dict] = []
    for line in label_path.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        cls = int(float(t[0]))
        cx, cy, bw, bh = (float(v) for v in t[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h

        kvals = t[5:]
        kpts = np.zeros((n_kpt, 3), dtype=np.float64)
        for i in range(n_kpt):
            base = i * kpt_dim
            if base + 1 >= len(kvals):
                break
            kx = float(kvals[base]) * w
            ky = float(kvals[base + 1]) * h
            vis = float(kvals[base + 2]) if kpt_dim == 3 else 2.0
            kpts[i] = (kx, ky, vis)

        area = max(bw * w * bh * h, 1e-6)  # box area in pixels, used as OKS scale s^2
        instances.append({
            "cls": cls,
            "box": np.array([x1, y1, x2, y2], dtype=np.float64),
            "kpts": kpts,
            "area": area,
        })
    return instances


def extract_predictions(result) -> List[Dict]:
    """Turn one Ultralytics Results object into a list of predicted instances."""
    preds: List[Dict] = []
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return preds
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    kdata = result.keypoints.data.cpu().numpy()  # (N, K, 3) -> x, y, conf
    for i in range(len(boxes)):
        preds.append({"box": boxes[i], "kpts": kdata[i], "conf": float(confs[i])})
    return preds


# ---------------------------------------------------------------------------
# Instance matching + metrics
# ---------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gts: List[Dict], preds: List[Dict], iou_thr: float) -> Dict[int, int]:
    """Greedy IoU matching. Returns {gt_index: pred_index} for matched pairs."""
    if not preds or not gts:
        return {}
    iou = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            iou[i, j] = box_iou(g["box"], p["box"])

    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_g, used_p, matched = set(), set(), {}
    for i, j in order:
        i, j = int(i), int(j)
        if i in used_g or j in used_p or iou[i, j] < iou_thr:
            continue
        matched[i] = j
        used_g.add(i)
        used_p.add(j)
    return matched


class MetricAccumulator:
    """Accumulate OKS / PCKh over a whole validation fold, per keypoint."""

    def __init__(self, cfg: CVConfig):
        self.cfg = cfg
        k = cfg.n_kpt
        self.image_oks: List[float] = []          # one OKS per GT instance
        self.oks_sum = np.zeros(k)                 # per-kpt OKS accumulator
        self.oks_cnt = np.zeros(k)
        self.pck_hit = np.zeros(k)                 # per-kpt correct count
        self.pck_cnt = np.zeros(k)

    def add_instance(self, gt: Dict, pred: Optional[Dict]) -> None:
        cfg = self.cfg
        gk = gt["kpts"]
        vis = gk[:, 2] > 0                          # visibility flag => ignore v==0
        if vis.sum() == 0:
            return

        if pred is None:
            # Missed detection: OKS = 0, and every visible kpt counts as a PCK miss.
            self.image_oks.append(0.0)
            h0, h1 = cfg.head_ids
            if gk[h0, 2] > 0 and gk[h1, 2] > 0:
                head = np.linalg.norm(gk[h0, :2] - gk[h1, :2])
                if head > 0:
                    self.pck_cnt[vis] += 1          # hits stay 0
            return

        pk = pred["kpts"]
        d2 = (gk[:, 0] - pk[:, 0]) ** 2 + (gk[:, 1] - pk[:, 1]) ** 2
        d = np.sqrt(d2)

        # ---- OKS ----
        kappa = 2.0 * cfg.sigmas()                  # k_i = 2 * sigma_i
        e = d2 / (2.0 * gt["area"] * kappa ** 2 + 1e-12)
        oks_per_kpt = np.exp(-e)
        self.image_oks.append(float(oks_per_kpt[vis].mean()))
        self.oks_sum[vis] += oks_per_kpt[vis]
        self.oks_cnt[vis] += 1

        # ---- PCKh@alpha (head size reference) ----
        h0, h1 = cfg.head_ids
        if gk[h0, 2] > 0 and gk[h1, 2] > 0:
            head = np.linalg.norm(gk[h0, :2] - gk[h1, :2])
            if head > 0:
                thr = cfg.pck_alpha * head
                correct = (d <= thr).astype(float)
                self.pck_hit[vis] += correct[vis]
                self.pck_cnt[vis] += 1

    def finalize(self) -> Dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            per_kpt_oks = np.where(self.oks_cnt > 0, self.oks_sum / self.oks_cnt, np.nan)
            per_kpt_pck = np.where(self.pck_cnt > 0, self.pck_hit / self.pck_cnt, np.nan)
        return {
            "mean_oks": float(np.mean(self.image_oks)) if self.image_oks else float("nan"),
            "mean_pck": float(self.pck_hit.sum() / self.pck_cnt.sum())
            if self.pck_cnt.sum() > 0 else float("nan"),
            "per_kpt_oks": per_kpt_oks,
            "per_kpt_pck": per_kpt_pck,
        }


def evaluate_predictions(gts_list: List[List[Dict]],
                         preds_list: List[List[Dict]],
                         cfg: CVConfig) -> Dict:
    """Compute OKS / PCKh over a fold given per-image GT and predictions."""
    acc = MetricAccumulator(cfg)
    for gts, preds in zip(gts_list, preds_list):
        matched = greedy_match(gts, preds, cfg.iou_match_thr)
        for i, gt in enumerate(gts):
            j = matched.get(i)
            acc.add_instance(gt, preds[j] if j is not None else None)
    return acc.finalize()


# ---------------------------------------------------------------------------
# Training / native validation / custom evaluation for one fold
# ---------------------------------------------------------------------------
def train_fold(cfg: CVConfig, fold_idx: int, data_yaml: Path):
    """Train a fresh (from-scratch) model on a fold and return the trained model."""
    from ultralytics import YOLO  # lazy import (heavy)

    model = YOLO(cfg.model)  # a ".yaml" => random init, no pretrained weights
    train_args = {
        "data": str(data_yaml),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "device": cfg.device,
        "seed": cfg.seed,
        "project": str((cfg.output_dir / "runs").resolve()),
        "name": f"fold_{fold_idx}",
        "exist_ok": True,
        "pretrained": False,   # train from scratch
        "save": False,         # do not persist model weights (per requirement)
        "plots": False,
        "verbose": True,
        "batch": 16
    }
    train_args.update(cfg.hyp)  # optimizer + augmentation overrides from the hyp yaml
    logger.info("Fold %d: training %s for %d epochs on device=%s",
                fold_idx, cfg.model, cfg.epochs, cfg.device)
    model.train(**train_args)
    return model


def validate_native(model, cfg: CVConfig, data_yaml: Path) -> Dict:
    """Run Ultralytics' own pose validation to obtain mAP (OKS-based)."""
    metrics = model.val(data=str(data_yaml), imgsz=cfg.imgsz, device=cfg.device,
                        split="val", plots=False, verbose=False, save_json=False)
    out = {"map50": float("nan"), "map50_95": float("nan")}
    try:
        out["map50"] = float(metrics.pose.map50)
        out["map50_95"] = float(metrics.pose.map)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read native pose mAP: %s", exc)
    return out


def custom_evaluate(model, val_items: List[Item], cfg: CVConfig) -> Dict:
    """Run predictions on the validation fold and compute OKS / PCKh ourselves."""
    gts_list, preds_list = [], []
    for it in val_items:
        w, h = image_size(it.image)
        gts_list.append(parse_label_file(it.label, cfg.n_kpt, cfg.kpt_dim, w, h))
        res = model.predict(source=str(it.image), imgsz=cfg.imgsz,
                            device=cfg.device, verbose=False, batch=16)[0]
        preds_list.append(extract_predictions(res))
    return evaluate_predictions(gts_list, preds_list, cfg)


def free_model(model) -> None:
    """Release GPU/CPU memory held by a fold's model."""
    try:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def save_fold_manifest(items: List[Item], fold_of: np.ndarray, path: Path) -> None:
    """Save which image went to which fold (reproducibility / audit)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "group", "split", "image", "label"])
        for it, fold in zip(items, fold_of):
            writer.writerow([int(fold), it.group, it.split, str(it.image), str(it.label)])
    logger.info("Fold manifest written to %s", path)


def save_summary_csv(fold_metrics: List[Dict], path: Path) -> None:
    """Per-fold scalar metrics + mean/std rows."""
    keys = ["map50", "map50_95", "mean_oks", "mean_pck"]
    rows = [{"fold": i, **{k: fm[k] for k in keys}} for i, fm in enumerate(fold_metrics)]
    arr = {k: np.array([fm[k] for fm in fold_metrics], dtype=float) for k in keys}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", *keys])
        for r in rows:
            writer.writerow([r["fold"], *[f"{r[k]:.6f}" for k in keys]])
        writer.writerow(["mean", *[f"{np.nanmean(arr[k]):.6f}" for k in keys]])
        writer.writerow(["std", *[f"{np.nanstd(arr[k]):.6f}" for k in keys]])
    logger.info("Summary metrics written to %s", path)


def save_per_keypoint_csv(fold_metrics: List[Dict], key: str, path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Save a per-keypoint metric across folds and return (mean, std) arrays.

    `key` is 'per_kpt_oks' or 'per_kpt_pck'.
    """
    stack = np.vstack([fm[key] for fm in fold_metrics])  # (n_folds, n_kpt)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["keypoint"] + [f"fold_{i}" for i in range(stack.shape[0])] + ["mean", "std"]
        writer.writerow(header)
        for k in range(stack.shape[1]):
            writer.writerow([k, *[f"{v:.6f}" for v in stack[:, k]],
                             f"{mean[k]:.6f}", f"{std[k]:.6f}"])
    logger.info("Per-keypoint %s written to %s", key, path)
    return mean, std


def plot_per_keypoint(pck_mean: np.ndarray, pck_std: np.ndarray,
                      oks_mean: np.ndarray, oks_std: np.ndarray,
                      path: Path) -> None:
    """Single figure: two bar sub-plots (PCKh and OKS) per keypoint with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(pck_mean)
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 0.28), 8), sharex=True)

    for ax, mean, std, title in (
        (axes[0], pck_mean, pck_std, "PCKh@0.5 per keypoint (mean ± std over folds)"),
        (axes[1], oks_mean, oks_std, "OKS per keypoint (mean ± std over folds)"),
    ):
        ax.bar(x, mean, yerr=std, capsize=2, color="#4C78A8", edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_ylabel("score")

    axes[-1].set_xlabel("keypoint id")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Per-keypoint bar plot written to %s", path)

# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run_cross_validation(cfg: CVConfig) -> None:
    """Run the full stratified k-fold cross-validation and write all reports."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) discover + fold assignment
    items, all_standard = discover_items(cfg)
    cfg.names = detect_class_names(items, cfg)
    logger.info("Detected classes: %s", cfg.names)

    fold_of = assign_folds(items, cfg)
    save_fold_manifest(items, fold_of, cfg.output_dir / "folds_assignment.csv")

    # 2) symlink pool only if the layout is non-standard (image/label)
    use_pool = not all_standard
    if use_pool:
        logger.info("Non-standard layout detected -> building a symlink pool.")
        build_symlink_pool(items, cfg.output_dir / "pool")
    else:
        logger.info("Standard images/labels layout -> no symlink pool needed.")

    # 3) per-fold train + evaluate
    fold_metrics: List[Dict] = []
    for fold_idx in range(cfg.n_splits):
        logger.info("=" * 70)
        logger.info("FOLD %d / %d", fold_idx + 1, cfg.n_splits)
        logger.info("=" * 70)

        data_yaml, val_items = write_fold_yaml(cfg, fold_idx, fold_of, items, use_pool)
        model = train_fold(cfg, fold_idx, data_yaml)

        native = validate_native(model, cfg, data_yaml)
        custom = custom_evaluate(model, val_items, cfg)
        fold_metrics.append({**native, **custom})

        logger.info("Fold %d: mAP50=%.4f mAP50-95=%.4f OKS=%.4f PCKh@0.5=%.4f",
                    fold_idx, native["map50"], native["map50_95"],
                    custom["mean_oks"], custom["mean_pck"])
        free_model(model)

    # 4) reports + plot
    save_summary_csv(fold_metrics, cfg.output_dir / "summary_metrics.csv")
    pck_mean, pck_std = save_per_keypoint_csv(
        fold_metrics, "per_kpt_pck", cfg.output_dir / "per_keypoint_pck.csv")
    oks_mean, oks_std = save_per_keypoint_csv(
        fold_metrics, "per_kpt_oks", cfg.output_dir / "per_keypoint_oks.csv")
    plot_per_keypoint(pck_mean, pck_std, oks_mean, oks_std,
                      cfg.output_dir / "per_keypoint_metrics.png")

    logger.info("Cross-validation finished. Reports in %s", cfg.output_dir)