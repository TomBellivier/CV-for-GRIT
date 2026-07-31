"""
Build the two datasets the top-down pipeline needs.

From the four per-group YOLO-pose datasets, this produces:

1. ``det/``        -- full images, detection labels only, one class per insect
                      group. The detector therefore also predicts the taxon,
                      which the pose stage never has to be told.
2. ``pose_crops/`` -- one image per annotated insect, cropped around its box
                      with a margin, plus a pose label expressed in crop
                      coordinates. A single class, all keypoints kept.

Splits are preserved: a crop taken from a train image lands in the crop train
split, and likewise for val. Nothing leaks.

Example
-------
python prepare_two_stage_dataset.py \
    --data-config groups.yaml \
    --out-dir two_stage_data \
    --margin 1.25 --min-size 32
"""

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_utils import (IMAGE_EXTENSIONS, load_group_mapping,  # noqa: E402
                           read_data_yaml, resolve_split_dirs)

SPLITS = ("train", "val")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--out-dir", default="two_stage_data")
    parser.add_argument("--margin", type=float, default=1.25,
                        help="Box expansion factor for the crops. Must match "
                             "the value used at inference time.")
    parser.add_argument("--min-size", type=int, default=32,
                        help="Skip crops smaller than this, in pixels.")
    parser.add_argument("--link", action="store_true",
                        help="Symlink the detection images instead of copying "
                             "them (saves disk, needs a filesystem that allows "
                             "symlinks).")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def iter_split_files(data_yaml, split):
    """Yield (image_path, label_path) for one split of one group dataset."""
    try:
        from ultralytics.data.utils import img2label_paths
    except Exception:  # pragma: no cover - Ultralytics always ships this
        img2label_paths = None

    for directory in resolve_split_dirs(data_yaml, split):
        if not directory.exists():
            continue
        images = sorted(p for p in directory.rglob("*")
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
        if img2label_paths is not None:
            labels = [Path(p) for p in img2label_paths([str(p) for p in images])]
        else:
            labels = [Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
                      for p in images]
        yield from zip(images, labels)


def read_pose_label(label_path, n_kpts):
    """Parse a YOLO-pose label file, keeping normalised coordinates."""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) < 5 + n_kpts * 2:
            continue
        values = [float(v) for v in tokens]
        n_fields = len(values) - 5
        dim = 3 if n_fields == n_kpts * 3 else 2
        if n_fields < n_kpts * dim:
            continue
        kpts = np.array(values[5:5 + n_kpts * dim], dtype=float).reshape(n_kpts, dim)
        if dim == 2:
            kpts = np.concatenate([kpts, np.full((n_kpts, 1), 2.0)], axis=1)
        rows.append({"cls": int(values[0]),
                     "box": np.array(values[1:5], dtype=float),
                     "kpts": kpts})
    return rows


def make_dirs(root):
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_detection_entry(det_root, split, image_path, instances, class_index,
                          stem, link, stats):
    """Copy/link one full image and write its detection label."""
    dst_img = det_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
    if not dst_img.exists():
        if link:
            try:
                dst_img.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, dst_img)
        else:
            shutil.copy2(image_path, dst_img)

    lines = []
    for item in instances:
        cx, cy, bw, bh = item["box"]
        lines.append(f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (det_root / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    stats[f"det_{split}"] += 1


def write_crops(pose_root, split, image_path, instances, n_kpts, margin,
                min_size, stem, quality, stats, crop_sizes):
    """Crop every annotated insect and write its pose label in crop space."""
    image = cv2.imread(str(image_path))
    if image is None:
        stats["unreadable"] += 1
        return
    img_h, img_w = image.shape[:2]

    for index, item in enumerate(instances):
        cx, cy, bw, bh = item["box"]
        cx, bw = cx * img_w, bw * img_w
        cy, bh = cy * img_h, bh * img_h

        half_w, half_h = bw * margin / 2.0, bh * margin / 2.0
        x1 = int(max(0, round(cx - half_w)))
        y1 = int(max(0, round(cy - half_h)))
        x2 = int(min(img_w, round(cx + half_w)))
        y2 = int(min(img_h, round(cy + half_h)))

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < min_size or crop_h < min_size:
            stats["too_small"] += 1
            continue

        crop = image[y1:y2, x1:x2]
        crop_stem = f"{stem}_{index:02d}"
        cv2.imwrite(str(pose_root / "images" / split / f"{crop_stem}.jpg"), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        crop_sizes.append((crop_w, crop_h))

        # Box in crop coordinates, clamped to the crop.
        bx1 = max(0.0, cx - bw / 2.0 - x1)
        by1 = max(0.0, cy - bh / 2.0 - y1)
        bx2 = min(float(crop_w), cx + bw / 2.0 - x1)
        by2 = min(float(crop_h), cy + bh / 2.0 - y1)
        n_cx = ((bx1 + bx2) / 2.0) / crop_w
        n_cy = ((by1 + by2) / 2.0) / crop_h
        n_bw = (bx2 - bx1) / crop_w
        n_bh = (by2 - by1) / crop_h

        kpts = item["kpts"].copy()
        kx = kpts[:, 0] * img_w - x1
        ky = kpts[:, 1] * img_h - y1
        visible = kpts[:, 2].copy()
        # A keypoint pushed outside the crop by the margin is no longer
        # observable in this image; mark it invisible rather than clamp it to
        # an edge, which would teach the model a wrong location.
        outside = (kx < 0) | (kx > crop_w) | (ky < 0) | (ky > crop_h)
        visible[outside] = 0
        kx = np.clip(kx, 0, crop_w) / crop_w
        ky = np.clip(ky, 0, crop_h) / crop_h
        stats["kpts_dropped"] += int(outside.sum())

        fields = [f"0 {n_cx:.6f} {n_cy:.6f} {n_bw:.6f} {n_bh:.6f}"]
        for k in range(n_kpts):
            fields.append(f"{kx[k]:.6f} {ky[k]:.6f} {int(visible[k])}")
        (pose_root / "labels" / split / f"{crop_stem}.txt").write_text(
            " ".join(fields) + "\n", encoding="utf-8")
        stats[f"crops_{split}"] += 1


def main():
    args = parse_args()
    groups = load_group_mapping(args.data_config)
    out_dir = Path(args.out_dir).resolve()
    det_root = out_dir / "det"
    pose_root = out_dir / "pose_crops"
    make_dirs(det_root)
    make_dirs(pose_root)

    reference = read_data_yaml(next(iter(groups.values())))
    n_kpts, kpt_dim = reference["n_kpts"], reference["kpt_dim"]
    print(f"[prepare] keypoint layout: {n_kpts} x {kpt_dim}")

    stats = Counter()
    crop_sizes = []
    per_group = defaultdict(Counter)
    class_names = {}

    for class_index, (group, data_yaml) in enumerate(groups.items()):
        class_names[class_index] = group
        info = read_data_yaml(data_yaml)
        if info["n_kpts"] != n_kpts:
            raise ValueError(
                f"group '{group}' declares {info['n_kpts']} keypoints, "
                f"expected {n_kpts}. Harmonise the datasets first.")

        for split in SPLITS:
            for image_path, label_path in iter_split_files(data_yaml, split):
                instances = read_pose_label(label_path, n_kpts)
                if not instances:
                    stats["no_label"] += 1
                    continue
                stem = f"{group}__{image_path.stem}"
                write_detection_entry(det_root, split, image_path, instances,
                                      class_index, stem, args.link, stats)
                write_crops(pose_root, split, image_path, instances, n_kpts,
                            args.margin, args.min_size, stem,
                            args.jpeg_quality, stats, crop_sizes)
                per_group[group][split] += len(instances)

    det_yaml = {
        "path": str(det_root),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    (det_root / "det.yaml").write_text(
        yaml.safe_dump(det_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    pose_yaml = {
        "path": str(pose_root),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [n_kpts, 3],
        "names": {0: "insect"},
    }
    source_raw = reference["raw"]
    if source_raw.get("flip_idx"):
        pose_yaml["flip_idx"] = list(source_raw["flip_idx"])
    if source_raw.get("kpt_names"):
        pose_yaml["kpt_names"] = source_raw["kpt_names"]
    (pose_root / "pose.yaml").write_text(
        yaml.safe_dump(pose_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    print("\n[prepare] detection dataset:", det_root / "det.yaml")
    print(f"           train images {stats['det_train']}, "
          f"val images {stats['det_val']}")
    print("[prepare] pose crop dataset:", pose_root / "pose.yaml")
    print(f"           train crops {stats['crops_train']}, "
          f"val crops {stats['crops_val']}")
    for group in groups:
        print(f"           {group:<14} train {per_group[group]['train']:>5} "
              f"val {per_group[group]['val']:>5}")
    if stats["too_small"]:
        print(f"[prepare] {stats['too_small']} crop(s) skipped, below "
              f"--min-size {args.min_size}")
    if stats["kpts_dropped"]:
        print(f"[prepare] {stats['kpts_dropped']} keypoint(s) fell outside "
              f"their crop and were marked invisible. If that number is large, "
              f"raise --margin.")
    if stats["no_label"]:
        print(f"[prepare] {stats['no_label']} image(s) had no usable label")

    if crop_sizes:
        sizes = np.array(crop_sizes)
        longest = sizes.max(axis=1)
        suggestion = int(np.ceil(np.percentile(longest, 90) / 32) * 32)
        print(f"\n[prepare] crop size: median {int(np.median(longest))} px, "
              f"p90 {int(np.percentile(longest, 90))} px, "
              f"max {int(longest.max())} px")
        print(f"[prepare] suggested --pose-imgsz: {min(suggestion, 640)} "
              f"(a multiple of 32 covering ~90% of crops without upscaling)")
    print(f"\n[prepare] remember: --margin {args.margin} must be passed "
          f"unchanged to eval_two_stage.py.")


if __name__ == "__main__":
    main()
