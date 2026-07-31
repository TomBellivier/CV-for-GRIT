"""
Evaluate the group-BatchNorm approach and write the standard workbook.

The base model is loaded once, its BatchNorm layers are turned into four-way
banks, each group's tensors are installed in its own bank, and every group is
then scored with its bank active.

Example
-------
python eval_group_bn.py \
    --manifest gbn_weights/gbn_manifest.json \
    --data-config groups.yaml \
    --out-dir pose_results
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_utils import load_group_mapping, read_data_yaml  # noqa: E402
from evaluate import (base_metadata, evaluate_one_group, read_learning_curve,  # noqa: E402
                      write_workbook)
from group_bn import (convert_to_group_bn, load_group_bn_state,  # noqa: E402
                      set_active_group)
from predictors import VariantPredictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="gbn_manifest.json produced by train_group_bn.py")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--out-dir", default="pose_results")
    parser.add_argument("--run-tag", default=None)

    parser.add_argument("--imgsz", type=int, default=None)
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


def build_multi_group_model(manifest, device=None):
    groups = list(manifest["groups"].keys())
    model = YOLO(manifest["base_weights"])
    convert_to_group_bn(model.model, groups)

    for group, entry in manifest["groups"].items():
        state = torch.load(entry["weights"], map_location="cpu",
                           weights_only=True)
        load_group_bn_state(model.model, state, group)

    if device is not None:
        model.model.to(device)
    return model, groups


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    if args.imgsz is None:
        args.imgsz = manifest.get("imgsz", 640)
    run_tag = args.run_tag or manifest.get("run_tag", "groupbn")
    args.run_tag = run_tag
    args.model = manifest["base_weights"]

    groups = load_group_mapping(args.data_config) if args.data_config else \
        {name: entry["data"] for name, entry in manifest["groups"].items()}

    model, _ = build_multi_group_model(manifest, args.device)

    summary_rows, per_keypoint_frames, curve_frames = [], [], []
    map_sources = set()

    for group, data_yaml in groups.items():
        print(f"[{run_tag}] evaluating group '{group}'")
        info = read_data_yaml(data_yaml)

        predictor = VariantPredictor(
            model,
            switch_fn=lambda name: set_active_group(model.model, name),
            n_kpts=info["n_kpts"], conf=args.conf, imgsz=args.imgsz,
            device=args.device)
        predictor.set_group(group)

        summary, per_keypoint, source = evaluate_one_group(
            predictor, data_yaml, info, args, map_source=args.map_source)
        map_sources.add(source)

        summary["group"] = group
        summary["training_time_sec"] = \
            manifest.get("training_time_sec", {}).get(group)
        summary_rows.append(summary)

        per_keypoint.insert(0, "group", group)
        per_keypoint_frames.append(per_keypoint)

        run_dir = manifest["groups"].get(group, {}).get("run_dir")
        if run_dir:
            curve = read_learning_curve(run_dir, group)
            if not curve.empty:
                curve_frames.append(curve)

    metadata = base_metadata(args, extra={
        "approach": "group_batchnorm",
        "base_weights": manifest["base_weights"],
        "gbn_also_train": ",".join(manifest.get("also_train", [])),
        "shared_weights_verified": manifest.get("shared_weights_verified", "n/a"),
        "map_source": "/".join(sorted(map_sources)),
        "map_kpt_sigma": args.map_kpt_sigma
        if args.map_kpt_sigma is not None else "1/n_kpts",
    })

    write_workbook(args.out_dir, run_tag, metadata, summary_rows,
                   per_keypoint_frames, curve_frames)


if __name__ == "__main__":
    main()
