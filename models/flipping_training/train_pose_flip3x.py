"""
Train a YOLO26 pose-estimation model on 4 insect datasets (Coleoptera,
Hymenoptera, Diptera, Lepidoptera) with an on-the-fly 3x flip augmentation.

For every TRAINING image, the dataset yields three views:
    - the original,
    - a horizontal flip,
    - a vertical flip,
never both flips at once. The training volume is therefore tripled.
The flipped views are generated in memory: nothing is written to disk.

------------------------------------------------------------------------------
Two YAML files, do not mix them up
------------------------------------------------------------------------------
--data : the DATASET yaml. It lists the 4 datasets' train/val paths and MUST
         contain `kpt_shape` and `flip_idx`. See dataset_combined.yaml.
--hyp  : the HYPERPARAMETERS yaml (lr0, box, pose, hsv, mosaic, ...). This is
         the file you attached. Its `fliplr`/`flipud` values are overridden to
         0.0 here, because the flips are done deterministically below.

------------------------------------------------------------------------------
Key correctness point (this is what usually breaks pose symmetry)
------------------------------------------------------------------------------
Flipping the image is not enough. For a symmetric keypoint layout, a flip must
(1) mirror the coordinates AND (2) permute the left/right keypoint indices via
`flip_idx`. Ultralytics applies `flip_idx` on its native *horizontal* flip
only, NOT on the vertical one, so an unguarded `flipud` teaches the model wrong
labels and produces exactly the output asymmetry you are seeing.

Since the images are top-down (dorsal) views of bilaterally symmetric insects,
BOTH flips need the SAME `flip_idx` permutation. Proof:
        vertical_flip = horizontal_flip . rotate_180
A 180 degree rotation keeps every keypoint identity (identity permutation) and
the horizontal flip contributes `flip_idx`; composing them leaves `flip_idx` as
the permutation for the vertical flip as well.

`flip_idx` is read from the dataset yaml (single source of truth). It is
validated at startup: right length and valid permutation, otherwise abort.

python train_pose_flip3x.py \
  --data dataset_combined.yaml \
  --hyp data.yaml \
  --epochs 100 --batch 16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.pose import PoseTrainer


# --------------------------------------------------------------------------- #
# Dataset: triples every training sample (original / hflip / vflip) in memory  #
# --------------------------------------------------------------------------- #
class TripleFlipDataset(YOLODataset):
    """A YOLODataset whose length is 3x the base length.

    Index layout over the epoch:
        [        0 .. n-1 ]  -> original
        [    n .. 2n-1    ]  -> horizontal flip
        [  2n .. 3n-1     ]  -> vertical flip

    `n_base` and `flip_idx` are injected by the trainer after construction
    (see TripleFlipPoseTrainer.build_dataset), so no custom __init__ is needed
    and we stay independent of the exact build_yolo_dataset signature.
    """

    n_base: int
    flip_idx: list[int] | None

    def __len__(self) -> int:
        return self.n_base * 3

    def get_image_and_label(self, index: int) -> dict:
        mode = index // self.n_base          # 0 = original, 1 = hflip, 2 = vflip
        base = index % self.n_base           # real image index in [0, n_base)
        label = super().get_image_and_label(base)
        if mode == 1:
            self._flip(label, "horizontal")
        elif mode == 2:
            self._flip(label, "vertical")
        return label

    def _flip(self, label: dict, direction: str) -> None:
        """Mirror the image + instances, then remap symmetric keypoints.

        Instances are still normalized (0..1) at this stage, so we pass a
        width/height of 1.0 to the Instances flip helpers.
        """
        instances = label["instances"]
        if direction == "horizontal":
            label["img"] = np.ascontiguousarray(np.fliplr(label["img"]))
            instances.fliplr(1.0)            # mirrors bbox / segments / keypoint x
        else:  # vertical
            label["img"] = np.ascontiguousarray(np.flipud(label["img"]))
            instances.flipud(1.0)            # mirrors bbox / segments / keypoint y

        # Permute left/right keypoints. Same flip_idx for both directions
        # (see module docstring for why the vertical flip uses it too).
        if self.flip_idx is not None and instances.keypoints is not None:
            instances.keypoints = np.ascontiguousarray(
                instances.keypoints[:, self.flip_idx, :]
            )


# --------------------------------------------------------------------------- #
# Trainer: swap the train dataset for the tripled version                      #
# --------------------------------------------------------------------------- #
class TripleFlipPoseTrainer(PoseTrainer):
    """PoseTrainer that returns a TripleFlipDataset for the training split."""

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode != "train":
            return dataset  # val/test stay untouched -> no data leakage

        flip_idx = self.data.get("flip_idx")
        _validate_flip_idx(flip_idx, self.data.get("kpt_shape"))

        # Re-tag the already-configured dataset instance as our subclass and
        # attach the two extra attributes. This reuses whatever the base
        # build_dataset produced, so it is robust across Ultralytics versions.
        dataset.__class__ = TripleFlipDataset
        dataset.n_base = len(dataset.labels)
        dataset.flip_idx = list(flip_idx)
        return dataset


def _validate_flip_idx(flip_idx, kpt_shape) -> None:
    """Fail early and clearly if the symmetry mapping is missing or wrong."""
    if flip_idx is None:
        raise ValueError(
            "`flip_idx` is missing from the dataset yaml. Make sure --data "
            "points to the DATASET yaml (with kpt_shape + flip_idx), not the "
            "hyperparameters yaml. `flip_idx[i]` is the index that keypoint i "
            "becomes under a left/right mirror (a self-index on the body axis)."
        )
    if kpt_shape is not None and len(flip_idx) != kpt_shape[0]:
        raise ValueError(
            f"`flip_idx` has {len(flip_idx)} entries but the model expects "
            f"{kpt_shape[0]} keypoints. They must match."
        )
    if sorted(flip_idx) != list(range(len(flip_idx))):
        raise ValueError(
            f"`flip_idx` = {flip_idx} is not a valid permutation of "
            f"0..{len(flip_idx) - 1}. Every keypoint index must appear once."
        )


# --------------------------------------------------------------------------- #
# Hyperparameters handling                                                     #
# --------------------------------------------------------------------------- #
def load_hyperparameters(hyp_path: str | None) -> dict:
    """Load the hyperparameter yaml and keep only keys Ultralytics recognizes.

    Unknown keys (e.g. a loss gain absent from the installed Ultralytics
    version) are dropped with a warning instead of crashing training.
    """
    if hyp_path is None:
        return {}
    raw = yaml.safe_load(Path(hyp_path).read_text()) or {}
    valid = {k: v for k, v in raw.items() if k in DEFAULT_CFG_DICT}
    dropped = sorted(set(raw) - set(valid))
    if dropped:
        print(f"Warning: ignoring unrecognized hyperparameters: {dropped}")
    return valid


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True,
                   help="DATASET yaml: 4 datasets' train/val paths + kpt_shape "
                        "+ flip_idx (see dataset_combined.yaml).")
    p.add_argument("--hyp", default=None,
                   help="HYPERPARAMETERS yaml (the file you attached).")
    p.add_argument("--model", default="yolo26s-pose.pt",
                   help="Default pretrained weights to start from.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 / 0,1 / cpu")
    p.add_argument("--project", default="runs/pose")
    p.add_argument("--name", default="insects_flip3x")
    p.add_argument("--output", default="best_model.pt",
                   help="Where the best checkpoint is copied after training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Start from the hyperparameters file, then apply the structural args and
    # the forced flip overrides. Later keys win, so the flips stay disabled
    # even though the hyp file sets fliplr: 0.5.
    train_kwargs = load_hyperparameters(args.hyp)
    train_kwargs.update(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,        # overwrite the same run folder, no run2/run3 pile-up
        # Deterministic 3x flip is handled in the dataset -> disable native flips
        # (native flipud is also keypoint-unaware, which is the bug to avoid).
        fliplr=0.0,
        flipud=0.0,
        # Keep the best checkpoint; no periodic per-epoch weights.
        save=True,
        save_period=-1,
    )

    model = YOLO(args.model)  # loads default pretrained pose weights
    model.train(trainer=TripleFlipPoseTrainer, **train_kwargs)

    # Keep only the best model, at a stable path of your choosing.
    best = Path(model.trainer.best)  # .../weights/best.pt
    if best.is_file():
        shutil.copy2(best, args.output)
        print(f"Best model saved to: {Path(args.output).resolve()}")
    else:
        print(f"Warning: best checkpoint not found at {best}")


if __name__ == "__main__":
    main()