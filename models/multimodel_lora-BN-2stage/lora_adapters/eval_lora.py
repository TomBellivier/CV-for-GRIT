"""
Evaluate the LoRA approach and write the standard workbook.

One shared backbone is loaded, all four adapter banks are installed on it, and
each group is scored with its own bank active. This is exactly the deployment
story the approach is meant to demonstrate: a single set of base weights in
memory, four behaviours one string assignment apart.

Example
-------
python eval_lora.py \
    --manifest lora_weights/lora_manifest.json \
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
from lora import LORA_KEY, inject_lora, set_active_group  # noqa: E402
from predictors import VariantPredictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="lora_manifest.json produced by train_lora.py")
    parser.add_argument("--data-config", default=None,
                        help="Overrides the data paths stored in the manifest.")
    parser.add_argument("--out-dir", default="pose_results")
    parser.add_argument("--run-tag", default=None,
                        help="Overrides the tag used in the output filename.")

    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--map-source", default="auto",
                        choices=["auto", "native", "custom"],
                        help="'auto'/'native' use Ultralytics' validator, which "
                             "is what your earlier runs used. 'custom' forces "
                             "the reimplementation, for like-for-like comparison "
                             "with the two-stage pipeline.")
    parser.add_argument("--map-kpt-sigma", type=float, default=None,
                        help="Per-keypoint sigma for the custom mAP. Defaults "
                             "to 1/n_kpts, matching Ultralytics.")

    # Recorded in the metadata sheet for traceability.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--pose", type=float, default=None)
    parser.add_argument("--kobj", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def build_multi_group_model(manifest, device=None):
    """Load the base model and install every group's adapter bank on it."""
    groups = list(manifest["groups"].keys())
    model = YOLO(manifest["base_weights"])

    inject_lora(model.model, groups,
                rank=manifest["rank"], alpha=manifest.get("alpha"),
                targets=manifest["targets"],
                skip_grouped=not manifest.get("include_grouped", False))

    for group, entry in manifest["groups"].items():
        state = torch.load(entry["weights"], map_location="cpu",
                           weights_only=True)
        # Each checkpoint holds a single bank, saved under the key of the group
        # it was trained on; it already matches the slot we want.
        missing, unexpected = model.model.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if LORA_KEY in k]
        if unexpected:
            raise RuntimeError(
                f"adapter tensors for '{group}' do not fit the injected model "
                f"(first mismatch: {unexpected[0]}). Check that --rank and "
                f"--targets match the values used at training time.")
        print(f"[lora] bank '{group}' installed ({len(state)} tensors)")

    if device is not None:
        model.model.to(device)
    return model, groups


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    if args.imgsz is None:
        args.imgsz = manifest.get("imgsz", 640)
    run_tag = args.run_tag or manifest.get("run_tag", "lora")
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
        "approach": "lora_adapters",
        "base_weights": manifest["base_weights"],
        "lora_rank": manifest["rank"],
        "lora_alpha": manifest.get("alpha"),
        "lora_targets": manifest["targets"],
        "lora_include_grouped": manifest.get("include_grouped", False),
        "lora_also_train": ",".join(manifest.get("also_train", [])),
        "map_source": "/".join(sorted(map_sources)),
        "map_kpt_sigma": args.map_kpt_sigma
        if args.map_kpt_sigma is not None else "1/n_kpts",
    })

    write_workbook(args.out_dir, run_tag, metadata, summary_rows,
                   per_keypoint_frames, curve_frames)


if __name__ == "__main__":
    main()
