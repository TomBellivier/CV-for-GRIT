"""
Evaluate the top-down pipeline and write the standard workbook.

Crucially, evaluation runs on the *original* per-group validation images and
against the *original* labels -- not on the crops. Scoring the pose model on its
own crops would compare it to nothing the other approaches ever saw, and would
quietly hide every detection failure. Here a missed detection costs keypoints,
exactly as it should.

Because no single Ultralytics model represents the pipeline, box/pose mAP come
from the reimplementation in ``common/map_eval.py``. The ``map_source`` row of
the metadata sheet records this; see the README for what it means for
comparability.

Example
-------
python eval_two_stage.py \
    --manifest two_stage_weights/two_stage_manifest.json \
    --data-config groups.yaml \
    --out-dir pose_results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_utils import load_group_mapping, read_data_yaml  # noqa: E402
from evaluate import (base_metadata, evaluate_one_group, read_learning_curve,  # noqa: E402
                      write_workbook)
from predictors import TwoStagePredictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="two_stage_manifest.json from train_two_stage.py")
    parser.add_argument("--data-config", default=None,
                        help="Original per-group pose data.yaml mapping.")
    parser.add_argument("--out-dir", default="pose_results")
    parser.add_argument("--run-tag", default=None)

    parser.add_argument("--margin", type=float, default=None,
                        help="Crop expansion. Defaults to the manifest value; "
                             "it must match the one used at preparation time.")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.01,
                        help="Low on purpose: each crop holds exactly one "
                             "insect, and the best instance is kept.")
    parser.add_argument("--det-imgsz", type=int, default=None)
    parser.add_argument("--pose-imgsz", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--filter-by-class", action="store_true",
                        help="Keep only detections whose predicted class is the "
                             "group under evaluation. Off by default, so the "
                             "pipeline is scored without being told the taxon.")

    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Kept for metadata parity; the detector uses "
                             "--det-conf.")
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--map-kpt-sigma", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--pose", type=float, default=None)
    parser.add_argument("--kobj", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    det_entry = manifest["detector"]
    pose_entry = manifest["pose"]
    margin = args.margin if args.margin is not None else manifest.get("margin", 1.25)
    det_imgsz = args.det_imgsz or det_entry.get("imgsz", 640)
    pose_imgsz = args.pose_imgsz or pose_entry.get("imgsz", 256)

    # The evaluation driver reads args.imgsz for the metadata sheet; the two
    # stages carry their own sizes.
    args.imgsz = det_imgsz
    args.model = f"{det_entry['weights']} + {pose_entry['weights']}"
    run_tag = args.run_tag or \
        f"twostage_m{margin}_det{det_imgsz}_pose{pose_imgsz}"
    args.run_tag = run_tag

    groups = load_group_mapping(args.data_config)
    reference = read_data_yaml(next(iter(groups.values())))

    predictor = TwoStagePredictor(
        det_model=det_entry["weights"],
        pose_model=pose_entry["weights"],
        n_kpts=reference["n_kpts"],
        margin=margin,
        det_conf=args.det_conf,
        pose_conf=args.pose_conf,
        det_imgsz=det_imgsz,
        pose_imgsz=pose_imgsz,
        device=args.device,
        max_instances=args.max_instances,
    )
    if args.filter_by_class:
        predictor.group_class_map = {name: index
                                     for index, name in enumerate(groups)}

    summary_rows, per_keypoint_frames, curve_frames = [], [], []
    total_train_time = (det_entry.get("training_time_sec", 0) or 0) + \
        (pose_entry.get("training_time_sec", 0) or 0)

    for group, data_yaml in groups.items():
        print(f"[{run_tag}] evaluating group '{group}'")
        info = read_data_yaml(data_yaml)
        predictor.n_kpts = info["n_kpts"]
        predictor.set_group(group)

        summary, per_keypoint, source = evaluate_one_group(
            predictor, data_yaml, info, args, map_source="custom")

        summary["group"] = group
        # The two stages are trained once for all groups, so the per-group cell
        # holds the shared total rather than a group-specific figure.
        summary["training_time_sec"] = round(total_train_time, 2)
        summary_rows.append(summary)

        per_keypoint.insert(0, "group", group)
        per_keypoint_frames.append(per_keypoint)

    for name, entry in (("detector", det_entry), ("pose", pose_entry)):
        curve = read_learning_curve(entry.get("run_dir", ""), name)
        if not curve.empty:
            curve_frames.append(curve)

    metadata = base_metadata(args, extra={
        "approach": "two_stage_topdown",
        "detector_weights": det_entry["weights"],
        "pose_weights": pose_entry["weights"],
        "crop_margin": margin,
        "det_imgsz": det_imgsz,
        "pose_imgsz": pose_imgsz,
        "det_conf": args.det_conf,
        "pose_conf": args.pose_conf,
        "filter_by_class": args.filter_by_class,
        "det_training_time_sec": det_entry.get("training_time_sec"),
        "pose_training_time_sec": pose_entry.get("training_time_sec"),
        "map_source": "custom",
        "map_kpt_sigma": args.map_kpt_sigma
        if args.map_kpt_sigma is not None else "1/n_kpts",
    })

    write_workbook(args.out_dir, run_tag, metadata, summary_rows,
                   per_keypoint_frames, curve_frames)


if __name__ == "__main__":
    main()
