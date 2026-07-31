"""
Train the two stages of the top-down pipeline.

Stage A -- a 4-class detector on full images. Locating one large insect on a
macro shot is an easy task; this converges quickly and does not need a big model.

Stage B -- a pose model on the crops. Because every crop is centred and
normalised in scale, this model no longer has to learn scale invariance, which
is where the accuracy gain over the single-stage approach comes from. It can
also run at a much smaller input size than 640.

Example
-------
python train_two_stage.py \
    --det-data two_stage_data/det/det.yaml \
    --pose-data two_stage_data/pose_crops/pose.yaml \
    --det-model yolo26n.pt --pose-model yolo26n-pose.pt \
    --det-epochs 100 --pose-epochs 150 --pose-imgsz 256 \
    --out-dir two_stage_weights
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-data", required=True,
                        help="det.yaml from prepare_two_stage_dataset.py")
    parser.add_argument("--pose-data", required=True,
                        help="pose.yaml from prepare_two_stage_dataset.py")
    parser.add_argument("--det-model", default="yolo26n.pt")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--out-dir", default="two_stage_weights")
    parser.add_argument("--runs-dir", default="runs_two_stage")
    parser.add_argument("--margin", type=float, default=1.25,
                        help="Recorded in the manifest so that evaluation uses "
                             "the same crop geometry as training.")
    parser.add_argument("--skip-det", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")

    parser.add_argument("--det-epochs", type=int, default=100)
    parser.add_argument("--det-batch", type=int, default=16)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--det-lr0", type=float, default=0.01)

    parser.add_argument("--pose-epochs", type=int, default=150)
    parser.add_argument("--pose-batch", type=int, default=32)
    parser.add_argument("--pose-imgsz", type=int, default=256)
    parser.add_argument("--pose-lr0", type=float, default=0.01)
    parser.add_argument("--pose-gain", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--degrees", type=float, default=180.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    return parser.parse_args()


def train_detector(args):
    model = YOLO(args.det_model)
    t0 = time.time()
    model.train(
        data=args.det_data,
        epochs=args.det_epochs,
        batch=args.det_batch,
        imgsz=args.det_imgsz,
        lr0=args.det_lr0,
        patience=args.patience,
        device=args.device,
        project=args.runs_dir,
        name="detector",
        exist_ok=True,
        degrees=args.degrees,
        fliplr=0.5,  # safe here: detection boxes have no left/right semantics
        mosaic=0.5,
        close_mosaic=20,
        verbose=False,
    )
    save_dir = Path(model.trainer.save_dir)
    return save_dir / "weights" / "best.pt", save_dir, round(time.time() - t0, 2)


def train_pose(args):
    model = YOLO(args.pose_model)
    t0 = time.time()
    model.train(
        data=args.pose_data,
        epochs=args.pose_epochs,
        batch=args.pose_batch,
        imgsz=args.pose_imgsz,
        lr0=args.pose_lr0,
        pose=args.pose_gain,
        kobj=args.kobj,
        patience=args.patience,
        device=args.device,
        project=args.runs_dir,
        name="pose_crops",
        exist_ok=True,
        degrees=args.degrees,
        fliplr=args.fliplr,
        # Crops already frame the subject; mosaic and large scale jitter would
        # undo the normalisation this stage exists to provide.
        mosaic=0.0,
        scale=0.25,
        translate=0.05,
        verbose=False,
    )
    save_dir = Path(model.trainer.save_dir)
    return save_dir / "weights" / "best.pt", save_dir, round(time.time() - t0, 2)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "two_stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {}

    if not args.skip_det:
        print("[two-stage] training the detector")
        weights, save_dir, elapsed = train_detector(args)
        manifest["detector"] = {
            "weights": str(weights.resolve()),
            "run_dir": str(save_dir.resolve()),
            "imgsz": args.det_imgsz,
            "data": str(Path(args.det_data).resolve()),
            "training_time_sec": elapsed,
        }
        print(f"  detector: {weights} ({elapsed}s)")

    if not args.skip_pose:
        print("[two-stage] training the pose model on crops")
        weights, save_dir, elapsed = train_pose(args)
        manifest["pose"] = {
            "weights": str(weights.resolve()),
            "run_dir": str(save_dir.resolve()),
            "imgsz": args.pose_imgsz,
            "data": str(Path(args.pose_data).resolve()),
            "training_time_sec": elapsed,
        }
        print(f"  pose model: {weights} ({elapsed}s)")

    manifest["margin"] = args.margin
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[two-stage] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
