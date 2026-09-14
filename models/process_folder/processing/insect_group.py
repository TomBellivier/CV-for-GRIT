"""
insect_group.py
================

Maps an image's file name to its taxonomic group (coleoptera / diptera /
hymenoptera / lepidoptera). Images are expected under
    config.DATABASE_DIR / <group> / ...
which is the same layout the measurement-validity classifiers were trained
from (see ../conf_classifier/dataset.py -- index_image_database).

The group is used both as a one-hot feature for those classifiers and as its
own '<group>_one_hot' columns in the output CSV.
"""

from __future__ import annotations

from pathlib import Path

from . import config

INSECT_GROUPS = ["coleoptera", "diptera", "hymenoptera", "lepidoptera"]


def _key(path: Path) -> str:
    """Turn a path into the comparison key (file name or stem), like dataset_membership."""
    return path.name if config.MATCH_ON == "name" else path.stem


class GroupIndex:
    """Holds the file-name -> group map and answers per-image lookups."""

    def __init__(self, index: dict[str, str]):
        self._index = index

    def group_of(self, image_name: str) -> str | None:
        return self._index.get(_key(Path(image_name)))

    def one_hot(self, image_name: str) -> dict[str, float]:
        group = self.group_of(image_name)
        return {f"{g}_one_hot": (1.0 if g == group else 0.0) for g in INSECT_GROUPS}


def build_group_index(database_dir=None) -> GroupIndex:
    """Scan DATABASE_DIR/<group>/... once and return a ready-to-query index."""
    root = Path(database_dir or config.DATABASE_DIR)
    index: dict[str, str] = {}
    if not root.is_dir():
        print(f"[group] database dir not found: {root} -> insect group unavailable "
              f"(one-hot columns will be all zero).")
        return GroupIndex(index)

    for group in INSECT_GROUPS:
        group_dir = root / group
        if not group_dir.is_dir():
            continue
        n = 0
        for f in group_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in config.IMG_EXTENSIONS:
                index[_key(f)] = group
                n += 1
        print(f"[group] {group}: {n} image(s) indexed")
    return GroupIndex(index)
