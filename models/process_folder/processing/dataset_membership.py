"""
dataset_membership.py
=====================

Tell whether an image belongs to the training or validation split of the YOLO
datasets, by EXACT file name (as requested).

Expected dataset layout (standard YOLO):
    datasets/<dataset_name>/images/<split>/<image files>

All datasets found under DATASETS_ROOT are scanned, so an image counts as
"train" if its file name appears in ANY dataset's images/train folder (same for
val). Matching is done on the full file name by default; switch to stems via
config.MATCH_ON = "stem".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config


def _key(path: Path) -> str:
    """Turn a path into the comparison key (file name or stem)."""
    return path.name if config.MATCH_ON == "name" else path.stem


def _collect_split_names(datasets_root: Path, split_dirname: str) -> set[str]:
    """Collect the comparison keys of every image in `images/<split>` folders."""
    names: set[str] = set()
    if not datasets_root.is_dir():
        return names

    for dataset_dir in datasets_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        split_dir = dataset_dir / config.IMAGES_SUBDIR / split_dirname
        if not split_dir.is_dir():
            continue
        for f in split_dir.iterdir():
            if f.is_file() and f.suffix.lower() in config.IMG_EXTENSIONS:
                names.add(_key(f))
    return names


@dataclass
class DatasetMembership:
    """Holds the train/val name sets and answers membership questions."""
    train_names: set[str] = field(default_factory=set)
    val_names: set[str] = field(default_factory=set)

    def in_train(self, image_path: str | Path) -> bool:
        return _key(Path(image_path)) in self.train_names

    def in_val(self, image_path: str | Path) -> bool:
        return _key(Path(image_path)) in self.val_names


def build_membership(datasets_root=None) -> DatasetMembership:
    """Scan the datasets once and return a ready-to-query membership object."""
    root = Path(datasets_root or config.DATASETS_ROOT)
    train = _collect_split_names(root, config.TRAIN_SPLIT_DIRNAME)
    val = _collect_split_names(root, config.VAL_SPLIT_DIRNAME)
    print(f"[dataset] {len(train)} train names, {len(val)} val names "
          f"found under {root}")
    return DatasetMembership(train_names=train, val_names=val)
