"""
Stage 2 of the group-BatchNorm approach -- re-estimate the normalisation of the
shared base model on each group.

For every group: load the base model, freeze everything except the BatchNorm
layers, fine-tune on that group's data, then keep only the BatchNorm tensors.
Convolution weights cannot move, so the four results are guaranteed to sit on
one common set of filters.

Example
-------
python train_group_bn.py \
    --base-weights base_model/weights/best.pt \
    --data-config groups.yaml \
    --epochs 40 --lr0 0.002 \
    --out-dir gbn_weights
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils.torch_utils import de_parallel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_utils import load_group_mapping  # noqa: E402
from group_bn import (bn_state_dict, check_shared_weights,  # noqa: E402
                      extract_bn_from_checkpoint, freeze_except_bn)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-weights", required=True,
                        help="Shared base model produced by train_base.py")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--out-dir", default="gbn_weights")
    parser.add_argument("--runs-dir", default="runs_gbn")
    parser.add_argument("--also-train", default="",
                        help="Comma-separated parameter-name substrings to keep "
                             "trainable alongside the BatchNorm layers.")
    parser.add_argument("--verify-shared", action="store_true",
                        help="After training, check that non-BN weights are "
                             "bit-identical across the four checkpoints.")

    # BatchNorm-only fine-tuning converges fast and needs few epochs: there are
    # only a few thousand free parameters per group.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--degrees", type=float, default=180.0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--fliplr", type=float, default=0.0)
    return parser.parse_args()


def make_trainer_class(also_train):
    """PoseTrainer that isolates the BatchNorm layers before the optimiser.

    The hook has to sit in ``build_optimizer`` because Ultralytics' setup loop
    re-enables ``requires_grad`` on every float parameter outside its own
    ``freeze`` list, undoing anything set earlier.
    """

    class GroupBNPoseTrainer(PoseTrainer):
        def build_optimizer(self, model, *rest, **kwargs):
            freeze_except_bn(model, also_train=also_train)
            return super().build_optimizer(model, *rest, **kwargs)

    return GroupBNPoseTrainer


def train_one_group(args, group, data_yaml, also_train, run_tag):
    model = YOLO(args.base_weights)
    model.add_callback(
        "on_train_start",
        lambda trainer: freeze_except_bn(de_parallel(trainer.model),
                                         also_train=also_train, verbose=False))

    model.train(
        trainer=make_trainer_class(also_train),
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
        name=f"{run_tag}__{group}",
        exist_ok=True,
        degrees=args.degrees,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        fliplr=args.fliplr,
        verbose=False,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    try:
        state = extract_bn_from_checkpoint(best)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read {best} ({exc}); using the in-memory model.")
        state = bn_state_dict(de_parallel(model.model))

    if not state:
        raise RuntimeError(f"no BatchNorm tensors recovered for '{group}'")
    return state, save_dir, best


def main():
    args = parse_args()
    groups = load_group_mapping(args.data_config)
    also_train = tuple(t for t in args.also_train.split(",") if t.strip())
    run_tag = f"groupbn_e{args.epochs}_lr{args.lr0}_b{args.batch}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "base_weights": str(Path(args.base_weights).resolve()),
        "also_train": list(also_train),
        "imgsz": args.imgsz,
        "run_tag": run_tag,
        "groups": {},
        "training_time_sec": {},
    }
    checkpoints = []

    for group, data_yaml in groups.items():
        print(f"[{run_tag}] re-estimating BatchNorm for '{group}'")
        t0 = time.time()
        state, save_dir, best = train_one_group(args, group, data_yaml,
                                                also_train, run_tag)
        elapsed = round(time.time() - t0, 2)
        checkpoints.append(best)

        weight_path = out_dir / f"bn_{group}.pt"
        torch.save(state, weight_path)
        manifest["groups"][group] = {
            "weights": str(weight_path.resolve()),
            "run_dir": str(save_dir.resolve()),
            "data": str(data_yaml),
        }
        manifest["training_time_sec"][group] = elapsed
        size_mb = weight_path.stat().st_size / 1e6
        print(f"  {len(state)} BN tensors saved to {weight_path} "
              f"({size_mb:.2f} MB, {elapsed}s)")

    if args.verify_shared and len(checkpoints) > 1:
        ok, message = check_shared_weights(checkpoints)
        manifest["shared_weights_verified"] = bool(ok)
        manifest["shared_weights_message"] = message
        print(f"[{run_tag}] {'OK' if ok else 'FAILED'}: {message}")
        if not ok:
            print("  The freezing did not hold. Every group now carries its own "
                  "convolution weights, so this is no longer a shared-backbone "
                  "experiment. Check the Ultralytics version before comparing.")

    manifest_path = out_dir / "gbn_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{run_tag}] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
