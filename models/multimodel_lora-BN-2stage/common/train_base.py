"""
Stage 1 -- train the shared base model on all groups at once.

Both the LoRA approach and the group-BatchNorm approach need a single
generalist YOLO-pose model trained on the union of the four datasets. This
script produces it, and is meant to be run once; the resulting ``best.pt`` is
then passed to ``train_lora.py`` or ``train_group_bn.py`` via ``--base-weights``.

Example
-------
python train_base.py \
    --model yolo26n-pose.pt \
    --data-config groups.yaml \
    --epochs 150 --batch 16 --imgsz 640 \
    --degrees 180 --mosaic 0.5 \
    --out-dir base_model
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from dataset_utils import build_combined_yaml, load_group_mapping


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n-pose.pt",
                        help="Starting weights, e.g. yolo26n-pose.pt")
    parser.add_argument("--data-config", default=None,
                        help="YAML/JSON mapping group names to data.yaml paths.")
    parser.add_argument("--combined-yaml", default=None,
                        help="Reuse an existing merged data.yaml instead of "
                             "building one from --data-config.")
    parser.add_argument("--out-dir", default="base_model")
    parser.add_argument("--runs-dir", default="runs_base")
    parser.add_argument("--name", default="base")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default=None)

    # Augmentation. Dorsal views have no canonical orientation, so full
    # in-plane rotation is free data; mosaic is toned down because it wrecks
    # the scale statistics of macro shots.
    parser.add_argument("--degrees", type=float, default=180.0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--fliplr", type=float, default=0.0,
                        help="Leave at 0 unless every group's data.yaml "
                             "declares a correct flip_idx.")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    return parser.parse_args()


def train_base(args):
    if args.combined_yaml:
        combined = Path(args.combined_yaml)
    else:
        groups = load_group_mapping(args.data_config)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = build_combined_yaml(groups, out_dir / "combined_pose.yaml")

    if args.fliplr > 0:
        print("[base] fliplr is enabled -- make sure flip_idx is defined for "
              "all 42 keypoints, otherwise left and right legs get swapped.")

    model = YOLO(args.model)
    model.train(
        data=str(combined),
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
        name=args.name,
        exist_ok=True,
        degrees=args.degrees,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        fliplr=args.fliplr,
        scale=args.scale,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        verbose=False,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"[base] shared base model: {best}")
    return best, save_dir, combined


if __name__ == "__main__":
    train_base(parse_args())
