"""
Baseline evaluation: one plain YOLO-pose model, scored on every group.

This is the reference the three approaches have to beat, and it is also the way
to get a *like-for-like* comparison against the two-stage pipeline: run it with
``--map-source custom`` and the baseline is scored by the same reimplemented
mAP as the pipeline.

The model can be a single generalist (the base model from ``train_base.py``) or
a per-group model, via ``--weights-config``.

Example
-------
python eval_single.py \
    --weights base_model/weights/best.pt \
    --data-config groups.yaml \
    --out-dir pose_results --run-tag baseline_shared
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from dataset_utils import load_group_mapping, read_data_yaml
from evaluate import (base_metadata, evaluate_one_group, read_learning_curve,
                      write_workbook)
from predictors import SingleModelPredictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None,
                        help="One model applied to every group.")
    parser.add_argument("--weights-config", default=None,
                        help="JSON mapping group names to weight paths, when "
                             "each group has its own fully trained model.")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--out-dir", default="pose_results")
    parser.add_argument("--run-tag", default="baseline")
    parser.add_argument("--runs-config", default=None,
                        help="Optional JSON mapping group names to Ultralytics "
                             "run directories, to fill the learning_curves sheet.")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--map-source", default="auto",
                        choices=["auto", "native", "custom"])
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
    if not args.weights and not args.weights_config:
        raise SystemExit("provide --weights or --weights-config")

    groups = load_group_mapping(args.data_config)
    per_group_weights = {}
    if args.weights_config:
        per_group_weights = json.loads(
            Path(args.weights_config).read_text(encoding="utf-8"))
    run_dirs = {}
    if args.runs_config:
        run_dirs = json.loads(Path(args.runs_config).read_text(encoding="utf-8"))

    args.model = args.weights or args.weights_config
    args.run_tag = args.run_tag

    shared = YOLO(args.weights) if args.weights and not per_group_weights else None

    summary_rows, per_keypoint_frames, curve_frames = [], [], []
    map_sources = set()

    for group, data_yaml in groups.items():
        print(f"[{args.run_tag}] evaluating group '{group}'")
        info = read_data_yaml(data_yaml)

        model = shared if shared is not None else \
            YOLO(per_group_weights.get(group, args.weights))
        predictor = SingleModelPredictor(model, n_kpts=info["n_kpts"],
                                         conf=args.conf, imgsz=args.imgsz,
                                         device=args.device)

        summary, per_keypoint, source = evaluate_one_group(
            predictor, data_yaml, info, args, map_source=args.map_source)
        map_sources.add(source)

        summary["group"] = group
        summary["training_time_sec"] = None
        summary_rows.append(summary)

        per_keypoint.insert(0, "group", group)
        per_keypoint_frames.append(per_keypoint)

        if group in run_dirs:
            curve = read_learning_curve(run_dirs[group], group)
            if not curve.empty:
                curve_frames.append(curve)

    metadata = base_metadata(args, extra={
        "approach": "single_model_baseline",
        "weights": args.weights or "(per group)",
        "weights_config": args.weights_config or "",
        "map_source": "/".join(sorted(map_sources)),
        "map_kpt_sigma": args.map_kpt_sigma
        if args.map_kpt_sigma is not None else "1/n_kpts",
    })

    write_workbook(args.out_dir, args.run_tag, metadata, summary_rows,
                   per_keypoint_frames, curve_frames)


if __name__ == "__main__":
    main()
