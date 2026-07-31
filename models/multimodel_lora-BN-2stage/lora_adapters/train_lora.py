"""
Stage 2 of the LoRA approach -- train one adapter bank per insect group.

The shared base model (from ``train_base.py``) is loaded once per group, its
convolutions are wrapped with low-rank adapters, every base weight is frozen,
and only the adapters are optimised on that group's dataset. The result is one
small ``lora_<group>.pt`` file per group, holding a few hundred thousand
parameters instead of a full model.

Because the base weights are frozen, all four adapter banks are guaranteed to
sit on top of *the same* backbone -- which is what makes them swappable at
inference time.

Example
-------
python train_lora.py \
    --base-weights base_model/weights/best.pt \
    --data-config groups.yaml \
    --rank 8 --targets neck_head \
    --epochs 80 --batch 16 \
    --out-dir lora_weights
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
from lora import (extract_lora_from_checkpoint, freeze_base, inject_lora,  # noqa: E402
                  lora_state_dict, set_active_group)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-weights", required=True,
                        help="Shared base model produced by train_base.py")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--out-dir", default="lora_weights")
    parser.add_argument("--runs-dir", default="runs_lora")

    # LoRA hyper-parameters.
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Adapter scaling; defaults to rank (scale = 1).")
    parser.add_argument("--targets", default="neck_head",
                        help="'all', 'neck', 'neck_head', 'head', or an "
                             "explicit list such as '13,16,19,22,23'.")
    parser.add_argument("--include-grouped", action="store_true",
                        help="Also adapt depthwise convolutions (rarely useful).")
    parser.add_argument("--also-train", default="",
                        help="Comma-separated parameter-name substrings to keep "
                             "trainable on top of the adapters, e.g. 'model.23.'")

    # Training hyper-parameters. LoRA tolerates -- and usually wants -- a higher
    # learning rate than full fine-tuning, since far fewer parameters move.
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--degrees", type=float, default=180.0)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--fliplr", type=float, default=0.0)
    return parser.parse_args()


def make_trainer_class(group, args, also_train):
    """Build a PoseTrainer subclass that injects and isolates the adapters.

    Two hooks are needed because Ultralytics rebuilds the model from its yaml
    inside ``YOLO.train()``:

    * ``get_model``      -- inject the adapters right after the model is built,
      before the optimiser sees it;
    * ``build_optimizer`` -- freeze the base weights just before the optimiser
      is constructed, which is after Ultralytics' own loop that re-enables
      ``requires_grad`` on everything outside its ``freeze`` list.
    """

    class LoRAPoseTrainer(PoseTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            inject_lora(model, [group], rank=args.rank, alpha=args.alpha,
                        targets=args.targets,
                        skip_grouped=not args.include_grouped)
            set_active_group(model, group)
            return model

        def build_optimizer(self, model, *rest, **kwargs):
            freeze_base(model, also_train=also_train)
            return super().build_optimizer(model, *rest, **kwargs)

    return LoRAPoseTrainer


def train_one_group(args, group, data_yaml, also_train, run_tag):
    model = YOLO(args.base_weights)

    # Safety net: if the installed Ultralytics version changes the point at
    # which build_optimizer is called, this callback still isolates the
    # adapters before the first backward pass.
    model.add_callback(
        "on_train_start",
        lambda trainer: freeze_base(de_parallel(trainer.model),
                                    also_train=also_train, verbose=False))

    model.train(
        trainer=make_trainer_class(group, args, also_train),
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
        state = extract_lora_from_checkpoint(best)
        source = "best.pt"
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read adapters from {best} ({exc}); "
              "falling back to the in-memory model (last epoch).")
        state = lora_state_dict(de_parallel(model.model))
        source = "last epoch (in memory)"

    if not state:
        raise RuntimeError(
            f"no adapter tensors recovered for group '{group}'. The injection "
            f"most likely did not survive Ultralytics' model rebuild.")

    print(f"  recovered {len(state)} adapter tensors from {source}")
    return state, save_dir


def main():
    args = parse_args()
    groups = load_group_mapping(args.data_config)
    also_train = tuple(t for t in args.also_train.split(",") if t.strip())
    run_tag = f"lora_r{args.rank}_{args.targets}_e{args.epochs}_b{args.batch}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "base_weights": str(Path(args.base_weights).resolve()),
        "rank": args.rank,
        "alpha": args.alpha,
        "targets": args.targets,
        "include_grouped": args.include_grouped,
        "also_train": list(also_train),
        "imgsz": args.imgsz,
        "run_tag": run_tag,
        "groups": {},
        "training_time_sec": {},
    }

    for group, data_yaml in groups.items():
        print(f"[{run_tag}] training adapters for '{group}'")
        t0 = time.time()
        state, save_dir = train_one_group(args, group, data_yaml,
                                          also_train, run_tag)
        elapsed = round(time.time() - t0, 2)

        weight_path = out_dir / f"lora_{group}.pt"
        torch.save(state, weight_path)
        manifest["groups"][group] = {
            "weights": str(weight_path.resolve()),
            "run_dir": str(save_dir.resolve()),
            "data": str(data_yaml),
        }
        manifest["training_time_sec"][group] = elapsed
        size_mb = weight_path.stat().st_size / 1e6
        print(f"  adapters saved to {weight_path} ({size_mb:.2f} MB, {elapsed}s)")

    manifest_path = out_dir / "lora_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{run_tag}] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
