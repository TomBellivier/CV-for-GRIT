"""
Generic evaluation driver.

Takes any object implementing the ``BasePredictor`` interface and produces the
*exact same workbook* as the original ``train_eval_pose.py``:

    metadata         field / value
    summary          group, num_val_images, pose_map, pose_map50, pose_map75,
                     box_map, box_map50, num_matched, mean_kpt_conf, mpjpe_px,
                     nmpjpe, mean_oks, pck_0.05, pck_0.1, training_time_sec
    per_keypoint     group, kpt_index, kpt_name, n_obs, kpt_conf, mpjpe_px,
                     nmpjpe, pck_0.05, pck_0.1
    learning_curves  group + the columns of Ultralytics' results.csv

Column names and order are frozen. Approach-specific information goes into the
metadata sheet as extra field/value rows, which does not disturb comparison.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataset_utils import labels_for_images, resolve_val_images
from map_eval import MapCollector
from pose_metrics import (PCK_THRESHOLDS, KeypointAccumulator, match_instances,
                          parse_label_file)

# Frozen column order of the summary sheet.
SUMMARY_COLUMNS = [
    "group", "num_val_images", "pose_map", "pose_map50", "pose_map75",
    "box_map", "box_map50", "num_matched", "mean_kpt_conf", "mpjpe_px",
    "nmpjpe", "mean_oks", "pck_0.05", "pck_0.1", "training_time_sec",
]

NAN_MAP = {"pose_map": float("nan"), "pose_map50": float("nan"),
           "pose_map75": float("nan"), "box_map": float("nan"),
           "box_map50": float("nan")}


def run_keypoint_pipeline(predictor, data_yaml, info, conf, iou_match,
                          collect_map=False, kpt_sigma=None):
    """Predict on the validation split and compute per-keypoint metrics.

    Mirrors the original loop exactly (same image list, same label parsing, same
    greedy IoU matching, same accumulator), with one addition: when
    ``collect_map`` is set, ground truth and predictions are also fed to a
    MapCollector so that mAP can be computed without the native validator.
    """
    images = resolve_val_images(data_yaml, info)
    labels = labels_for_images(images)
    accumulator = KeypointAccumulator(info["n_kpts"], PCK_THRESHOLDS)
    collector = MapCollector(info["n_kpts"], kpt_sigma) if collect_map else None

    if not images:
        print("  WARNING: no validation images resolved; "
              "PCK/MPJPE will be empty.")
        return accumulator, 0, collector

    n_images = 0
    n_with_labels = 0
    for image_path, label_path in zip(images, labels):
        if not image_path.exists():
            continue
        n_images += 1

        pred_boxes, pred_kpts, pred_scores, (img_h, img_w) = \
            predictor.predict(image_path)

        gt = parse_label_file(Path(label_path), img_w, img_h,
                              info["n_kpts"], info["kpt_dim"])

        if collector is not None:
            collector.add(
                [item["box"] for item in gt],
                [item["kpts"] for item in gt],
                pred_boxes, pred_kpts, pred_scores)

        if not gt:
            continue
        n_with_labels += 1

        if len(pred_boxes) == 0:
            continue

        gt_boxes = [item["box"] for item in gt]
        for gi, pi in match_instances(gt_boxes, pred_boxes, iou_match):
            accumulator.add(gt[gi]["kpts"], pred_kpts[pi], gt[gi]["box"])

    if n_with_labels == 0:
        print("  WARNING: no label files found next to the validation images; "
              "check the images/labels folder layout.")
    elif accumulator.n_matched == 0:
        print(f"  WARNING: no GT-prediction pairs matched at IoU >= {iou_match}; "
              "PCK/MPJPE will be empty.")
    elif accumulator.count.sum() == 0:
        print("  WARNING: pairs matched but every GT keypoint has visibility 0; "
              "PCK/MPJPE will be empty. Check the visibility flags in the labels.")
    return accumulator, n_images, collector


def evaluate_one_group(predictor, data_yaml, info, args, map_source="auto"):
    """Score one group and return (summary_dict, per_keypoint_frame, source)."""
    native = None
    if map_source in ("auto", "native"):
        try:
            native = predictor.native_val(data_yaml, args.imgsz, args.device)
        except Exception as exc:  # noqa: BLE001 - never lose a run over metrics
            print(f"  WARNING: native validation failed ({exc}); "
                  "falling back to the custom mAP implementation.")
            native = None

    need_custom = native is None
    if map_source == "custom":
        need_custom = True

    accumulator, n_images, collector = run_keypoint_pipeline(
        predictor, data_yaml, info, args.conf, args.iou_match,
        collect_map=need_custom, kpt_sigma=getattr(args, "map_kpt_sigma", None))

    if native is not None and map_source != "custom":
        map_values, source = native, "native"
    elif collector is not None:
        map_values, source = collector.results(), "custom"
    else:
        map_values, source = dict(NAN_MAP), "none"

    summary = {"num_val_images": n_images}
    summary.update(map_values)
    summary.update(accumulator.summary())
    per_keypoint = accumulator.per_keypoint_frame(info["kpt_names"])
    return summary, per_keypoint, source


def read_learning_curve(save_dir, group_name):
    """Read the per-epoch results.csv produced during training."""
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    frame.columns = [c.strip() for c in frame.columns]
    frame.insert(0, "group", group_name)
    return frame


def base_metadata(args, extra=None):
    """Assemble the metadata rows, keeping the original fields first."""
    metadata = {
        "run_tag": getattr(args, "run_tag", ""),
        "model": getattr(args, "model", ""),
        "epochs": getattr(args, "epochs", ""),
        "batch": getattr(args, "batch", ""),
        "imgsz": getattr(args, "imgsz", ""),
        "lr0": getattr(args, "lr0", ""),
        "lrf": getattr(args, "lrf", ""),
        "pose": getattr(args, "pose", ""),
        "kobj": getattr(args, "kobj", ""),
        "patience": getattr(args, "patience", ""),
        "device": str(getattr(args, "device", None)),
        "conf": getattr(args, "conf", ""),
        "iou_match": getattr(args, "iou_match", ""),
        "oks_sigma": 0.05,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "all_args": json.dumps(vars(args), default=str),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_workbook(out_dir, run_tag, metadata, summary_rows,
                   per_keypoint_frames, curve_frames):
    """Write the four-sheet workbook, with the frozen summary column order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{run_tag}.xlsx"

    summary_df = pd.DataFrame(summary_rows)
    ordered = [c for c in SUMMARY_COLUMNS if c in summary_df.columns]
    extras = [c for c in summary_df.columns if c not in SUMMARY_COLUMNS]
    if extras:
        # Should not happen; guards against silently reordering the sheet.
        print(f"[warn] unexpected summary columns appended: {extras}")
    summary_df = summary_df[ordered + extras]

    per_keypoint_df = pd.concat(per_keypoint_frames, ignore_index=True) \
        if per_keypoint_frames else pd.DataFrame()
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
    return out_path
