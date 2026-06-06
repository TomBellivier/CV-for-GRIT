"""
Train and evaluate a single YOLO-pose model configuration across several
insect groups.

Run this script once per (model, hyper-parameter) combination. For every insect
group it instantiates a dedicated model, trains it on that group's YOLO-pose
dataset, evaluates it with a keypoint pipeline (OKS-based mAP from the native
validator, plus per-keypoint MPJPE / PCK / mean OKS computed here), and stores
every result in one self-contained Excel workbook.

The workbook is consumed later by compare_pose_results.py.

Example
-------
python train_eval_pose.py \
    --model yolo26n-pose.pt \
    --data-config groups.yaml \
    --epochs 100 --batch 16 --imgsz 640 \
    --lr0 0.01 --pose 12.0 --kobj 1.0 \
    --out-dir pose_results
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

# Default group -> data.yaml mapping, used when no --data-config file is given.
# Replace the paths with the actual location of each group's dataset.
DEFAULT_GROUPS = {
    "Coleoptera": "models/datasets/Coleoptera/yolo-config.yaml",
    "Hymenoptera": "models/datasets/Hymenoptera/yolo-config.yaml",
    "Lepidoptera": "models/datasets/Lepidoptera/yolo-config.yaml"
}

# PCK thresholds, expressed as a fraction of the ground-truth bbox diagonal.
PCK_THRESHOLDS = [0.05, 0.1]

# Constant per-keypoint sigma used for the custom OKS estimate. Without an
# established sigma table for these keypoints, a single value is a reasonable
# starting point and can be tuned later.
OKS_SIGMA = 0.05

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Model weights or config, e.g. yolo26n-pose.pt")
    parser.add_argument("--data-config", default=None,
                        help="YAML/JSON file mapping group names to data.yaml "
                             "paths. Falls back to DEFAULT_GROUPS if omitted.")
    parser.add_argument("--out-dir", default="pose_results",
                        help="Directory where the result workbook is written.")
    parser.add_argument("--runs-dir", default="runs_pose",
                        help="Ultralytics project directory for training runs.")

    # Training hyper-parameters.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--pose", type=float, default=12.0,
                        help="Pose (keypoint location) loss gain.")
    parser.add_argument("--kobj", type=float, default=1.0,
                        help="Keypoint objectness loss gain.")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default=None,
                        help="CUDA device id(s) or 'cpu'. Auto if omitted.")

    # Evaluation parameters.
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for the prediction loop.")
    parser.add_argument("--iou-match", type=float, default=0.5,
                        help="IoU threshold to match predictions to GT boxes.")
    return parser.parse_args()


def load_group_mapping(path):
    """Return a {group_name: data_yaml_path} dict from a YAML/JSON file."""
    if path is None:
        return dict(DEFAULT_GROUPS)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yml", ".yaml"}:
            content = yaml.safe_load(handle)
        else:
            content = json.load(handle)
    groups = content.get("groups", content)
    return {str(name): str(value) for name, value in groups.items()}


def read_data_yaml(data_yaml_path):
    """Parse a YOLO data.yaml and return resolved paths and keypoint shape."""
    data_path = Path(data_yaml_path).resolve()
    with data_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    val_field = data.get("val", "images/val")
    val_path = Path(val_field)
    if not val_path.is_absolute():
        val_path = (root / val_path).resolve()

    kpt_shape = data.get("kpt_shape", [None, 3])
    n_kpts, kpt_dim = int(kpt_shape[0]), int(kpt_shape[1])
    names = data.get("kpt_names", {})[0]
    return {
        "val_path": val_path,
        "n_kpts": n_kpts,
        "kpt_dim": kpt_dim,
        "names": names,
    }


def list_val_images(val_path):
    """Resolve the validation split into a list of image file paths."""
    if val_path.is_dir():
        return sorted(p for p in val_path.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTENSIONS)
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        with val_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        return [Path(line) for line in lines]
    return []


def label_path_for_image(image_path):
    """Map an image path to its YOLO label file path."""
    parts = list(image_path.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_label_file(label_path, img_w, img_h, n_kpts, kpt_dim):
    """Read one YOLO-pose label file into a list of GT instances (pixels)."""
    instances = []
    if not label_path.exists():
        return instances

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if len(values) < 5 + n_kpts * kpt_dim:
                continue
            values = [float(v) for v in values]
            cx, cy, bw, bh = values[1:5]
            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h

            raw_kpts = np.array(values[5:5 + n_kpts * kpt_dim], dtype=float)
            raw_kpts = raw_kpts.reshape(n_kpts, kpt_dim)
            kpts = np.zeros((n_kpts, 3), dtype=float)
            kpts[:, 0] = raw_kpts[:, 0] * img_w
            kpts[:, 1] = raw_kpts[:, 1] * img_h
            kpts[:, 2] = raw_kpts[:, 2] if kpt_dim == 3 else 2.0

            instances.append({"box": np.array([x1, y1, x2, y2]), "kpts": kpts})
    return instances


def box_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_instances(gt_boxes, pred_boxes, iou_thr):
    """Greedy GT-to-prediction matching by IoU. Returns (gt_idx, pred_idx)."""
    pairs = []
    used = set()
    for gi, gt_box in enumerate(gt_boxes):
        best_pi, best_iou = -1, iou_thr
        for pi, pred_box in enumerate(pred_boxes):
            if pi in used:
                continue
            iou = box_iou(gt_box, pred_box)
            if iou >= best_iou:
                best_iou, best_pi = iou, pi
        if best_pi >= 0:
            pairs.append((gi, best_pi))
            used.add(best_pi)
    return pairs


class KeypointAccumulator:
    """Aggregate per-keypoint distances, PCK hits and OKS over all matches."""

    def __init__(self, n_kpts, thresholds):
        self.n_kpts = n_kpts
        self.thresholds = thresholds
        self.dist_px = np.zeros(n_kpts)
        self.dist_norm = np.zeros(n_kpts)
        self.count = np.zeros(n_kpts)
        self.pck_hits = {thr: np.zeros(n_kpts) for thr in thresholds}
        self.oks_values = []
        self.n_matched = 0

    def add(self, gt_kpts, pred_kpts, gt_box):
        bw = gt_box[2] - gt_box[0]
        bh = gt_box[3] - gt_box[1]
        diag = math.hypot(bw, bh)
        area = max(bw * bh, 1e-6)
        if diag <= 0:
            return
        self.n_matched += 1

        visible = gt_kpts[:, 2] > 0
        deltas = gt_kpts[:, :2] - pred_kpts[:, :2]
        dists = np.linalg.norm(deltas, axis=1)

        oks_terms = []
        for k in range(self.n_kpts):
            if not visible[k]:
                continue
            d = dists[k]
            self.dist_px[k] += d
            self.dist_norm[k] += d / diag
            self.count[k] += 1
            for thr in self.thresholds:
                if d / diag <= thr:
                    self.pck_hits[thr][k] += 1
            oks_terms.append(math.exp(-(d ** 2) / (2 * area * OKS_SIGMA ** 2)))

        if oks_terms:
            self.oks_values.append(float(np.mean(oks_terms)))

    def per_keypoint_frame(self, names):
        rows = []
        for k in range(self.n_kpts):
            n = self.count[k]
            row = {
                "kpt_index": k,
                "kpt_name": names.get(k, str(k)) if isinstance(names, dict)
                else (names[k] if k < len(names) else str(k)),
                "n_obs": int(n),
                "mpjpe_px": self.dist_px[k] / n if n > 0 else np.nan,
                "nmpjpe": self.dist_norm[k] / n if n > 0 else np.nan,
            }
            for thr in self.thresholds:
                row[f"pck_{thr}"] = self.pck_hits[thr][k] / n if n > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self):
        total = self.count.sum()
        out = {
            "num_matched": self.n_matched,
            "mpjpe_px": self.dist_px.sum() / total if total > 0 else np.nan,
            "nmpjpe": self.dist_norm.sum() / total if total > 0 else np.nan,
            "mean_oks": float(np.mean(self.oks_values)) if self.oks_values
            else np.nan,
        }
        for thr in self.thresholds:
            hits = self.pck_hits[thr].sum()
            out[f"pck_{thr}"] = hits / total if total > 0 else np.nan
        return out


def run_keypoint_pipeline(model, info, conf, iou_match):
    """Predict on the validation split and compute per-keypoint metrics."""
    images = list_val_images(info["val_path"])
    accumulator = KeypointAccumulator(info["n_kpts"], PCK_THRESHOLDS)
    n_images = 0

    for image_path in images:
        if not image_path.exists():
            continue
        n_images += 1
        result = model.predict(str(image_path), conf=conf, verbose=False)[0]
        img_h, img_w = result.orig_shape

        gt = parse_label_file(label_path_for_image(image_path),
                              img_w, img_h, info["n_kpts"], info["kpt_dim"])
        if not gt:
            continue

        if result.boxes is None or result.keypoints is None \
                or len(result.boxes) == 0:
            continue
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_kpts = result.keypoints.data.cpu().numpy()

        gt_boxes = [item["box"] for item in gt]
        for gi, pi in match_instances(gt_boxes, pred_boxes, iou_match):
            accumulator.add(gt[gi]["kpts"], pred_kpts[pi], gt[gi]["box"])

    return accumulator, n_images


def build_run_tag(args):
    """Build a filesystem-safe identifier for this run configuration."""
    stem = Path(args.model).stem
    tag = f"{stem}_lr{args.lr0}_lrf{args.lrf}_pose{args.pose}" \
          f"_kobj{args.kobj}_e{args.epochs}_b{args.batch}_imgsz{args.imgsz}"
    return tag.replace("/", "-").replace("\\", "-")


def train_one_group(args, group_name, data_yaml, run_tag):
    """Train a model on one group and return (best_model, save_dir)."""
    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        pose=args.pose,
        kobj=args.kobj,
        patience=args.patience,
        device=args.device,
        project=args.runs_dir,
        name=f"{run_tag}__{group_name}",
        exist_ok=True,
        verbose=False,
    )
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_weights)) if best_weights.exists() else model
    return best_model, save_dir


def read_learning_curve(save_dir, group_name):
    """Read the per-epoch results.csv produced during training."""
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    frame.columns = [c.strip() for c in frame.columns]
    frame.insert(0, "group", group_name)
    return frame


def evaluate_one_group(best_model, data_yaml, info, args):
    """Run native validation and the custom keypoint pipeline for one group."""
    metrics = best_model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    accumulator, n_images = run_keypoint_pipeline(
        best_model, info, args.conf, args.iou_match)

    summary = {
        "num_val_images": n_images,
        "pose_map": float(metrics.pose.map),
        "pose_map50": float(metrics.pose.map50),
        "pose_map75": float(metrics.pose.map75),
        "box_map": float(metrics.box.map),
        "box_map50": float(metrics.box.map50),
    }
    summary.update(accumulator.summary())
    print(info.keys())
    per_keypoint = accumulator.per_keypoint_frame(info["names"])
    return summary, per_keypoint


def main():
    args = parse_args()
    groups = load_group_mapping(args.data_config)
    run_tag = build_run_tag(args)

    summary_rows = []
    per_keypoint_frames = []
    curve_frames = []

    for group_name, data_yaml in groups.items():
        print(f"[{run_tag}] processing group '{group_name}'")
        info = read_data_yaml(data_yaml)

        time0 = time.time()

        best_model, save_dir = train_one_group(
            args, group_name, data_yaml, run_tag)
        
        time1 = time.time()


        curve = read_learning_curve(save_dir, group_name)
        if not curve.empty:
            curve_frames.append(curve)

        summary, per_keypoint = evaluate_one_group(
            best_model, data_yaml, info, args)
        summary["group"] = group_name
        summary["training_time_sec"] = round(time1 - time0, 2)
        summary_rows.append(summary)

        per_keypoint.insert(0, "group", group_name)
        per_keypoint_frames.append(per_keypoint)

    metadata = {
        "run_tag": run_tag,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "pose": args.pose,
        "kobj": args.kobj,
        "patience": args.patience,
        "device": str(args.device),
        "conf": args.conf,
        "iou_match": args.iou_match,
        "oks_sigma": OKS_SIGMA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "all_args": json.dumps(vars(args)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{run_tag}.xlsx"

    summary_df = pd.DataFrame(summary_rows)
    front = ["group"] + [c for c in summary_df.columns if c != "group"]
    summary_df = summary_df[front]
    per_keypoint_df = pd.concat(per_keypoint_frames, ignore_index=True)
    curves_df = pd.concat(curve_frames, ignore_index=True) \
        if curve_frames else pd.DataFrame()
    metadata_df = pd.DataFrame(
        {"field": list(metadata.keys()), "value": list(metadata.values())})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_keypoint_df.to_excel(writer, sheet_name="per_keypoint", index=False)
        curves_df.to_excel(writer, sheet_name="learning_curves", index=False)

    print(f"[{run_tag}] results written to {out_path}")


if __name__ == "__main__":
    main()
