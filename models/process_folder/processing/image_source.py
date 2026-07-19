"""
image_source.py
===============

Unifies the two ways of feeding images into the pipeline:

    * a LOCAL folder   -> images read from disk with cv2,
    * a Hugging Face DATASET repo -> images streamed into RAM (nothing is
      written to disk), inspired by your test_process_hf.py.

Both are exposed through the same tiny contract so the rest of the pipeline
does not care where an image comes from:

    build_source(...) -> (items, load_fn)
        items   : list of (key, image_name)
                  - key        : what load_fn needs to fetch the image
                  - image_name : the plain file name, used for the CSV and for
                                 train/val membership (matched by exact name)
        load_fn : key -> BGR numpy array (or raises on failure)

The heavy parallelism (many downloads / reads in flight) is applied later by
parallel.py; here we only describe *what* to load and *how* to load one item.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np

from . import config

# Extensions without the leading dot, for HF globbing.
_HF_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp")


# --------------------------------------------------------------------------- #
# Helpers shared by both sources
# --------------------------------------------------------------------------- #
def _pil_to_bgr(pil_image) -> np.ndarray:
    """Convert a PIL image to a BGR uint8 array (the convention cv2/YOLO use)."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# LOCAL folder source
# --------------------------------------------------------------------------- #
def _list_local_images(folder: Path) -> list[tuple[Path, str]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    items = [
        (p, p.name)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in config.IMG_EXTENSIONS
    ]
    return items


def _load_local(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"cv2 could not read image: {path}")
    return img_bgr


def build_local_source(folder) -> tuple[list[tuple[Path, str]], callable]:
    """Return (items, load_fn) for a local folder of images."""
    folder = Path(folder)
    items = _list_local_images(folder)
    return items, _load_local


# --------------------------------------------------------------------------- #
# Hugging Face dataset source
# --------------------------------------------------------------------------- #
def _list_hf_images(fs, repo: str, folders: list[str] | None) -> list[tuple[str, str]]:
    """List image files in a HF dataset repo, optionally restricted to folders.

    Assumes the dataset was uploaded as raw files (imagefolder layout), so the
    files live under 'datasets/<repo>/...'. If you pushed parquet shards instead
    (push_to_hub), you would need the streaming API rather than HfFileSystem.
    """
    root = f"datasets/{repo}"
    # Where to look: the whole repo, or only the requested sub-folders.
    bases = [f"{root}/{sub}" for sub in folders] if folders else [root]

    keys: list[str] = []
    for base in bases:
        for ext in _HF_EXTS:
            for e in (ext, ext.upper()):
                # '**/*.ext' catches nested files; '*.ext' catches files that
                # sit directly in `base` (robust whatever the fsspec version's
                # handling of a zero-depth '**'). Duplicates are removed below.
                keys.extend(fs.glob(f"{base}/**/*.{e}"))
                keys.extend(fs.glob(f"{base}/*.{e}"))
    keys = sorted(set(keys))
    # image_name = the plain file name (used for membership + CSV).
    return [(k, os.path.basename(k)) for k in keys]


def _make_hf_loader(fs) -> callable:
    """Return a load_fn(key) that downloads+decodes one HF file to a BGR array."""
    from PIL import Image

    def load_fn(key: str) -> np.ndarray:
        with fs.open(key, "rb") as f:
            data = f.read()                 # bytes into RAM
        with Image.open(io.BytesIO(data)) as im:
            im.load()                       # force decode before the buffer closes
            return _pil_to_bgr(im)
    return load_fn


def build_hf_source(repo: str, folders: list[str] | None,
                    token: str | None) -> tuple[list[tuple[str, str]], callable]:
    """Return (items, load_fn) for a Hugging Face dataset repo."""
    from huggingface_hub import HfFileSystem

    token = token or os.environ.get("HF_TOKEN")
    fs = HfFileSystem(token=token)
    items = _list_hf_images(fs, repo, folders)
    load_fn = _make_hf_loader(fs)
    return items, load_fn


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def build_source(source_kind: str, *, folder=None, repo=None,
                 hf_folders=None, hf_token=None):
    """Return (items, load_fn) for the requested source.

    source_kind : "folder" | "hf"
    """
    if source_kind == "folder":
        return build_local_source(folder or config.INPUT_FOLDER)
    if source_kind == "hf":
        if not repo:
            raise ValueError("A Hugging Face dataset id is required (--dataset).")
        return build_hf_source(repo, hf_folders, hf_token)
    raise ValueError(f"Unknown source kind: {source_kind!r} (use 'folder' or 'hf').")
