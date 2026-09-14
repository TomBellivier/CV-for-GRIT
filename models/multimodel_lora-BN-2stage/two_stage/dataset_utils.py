"""
Dataset plumbing shared by the three approaches.

The image/label resolution helpers are ported from the original
``train_eval_pose.py`` so that the evaluation loop iterates over exactly the
same validation files as before.

``build_combined_yaml`` is new: it merges the four per-group datasets into a
single data.yaml, which the LoRA and group-BatchNorm approaches need for their
shared base-model training stage.
"""

import json
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default group -> data.yaml mapping, used when no --data-config file is given.
DEFAULT_GROUPS = {
    "Coleoptera": "../../datasets/Coleoptera/yolo-config.yaml",
    "Diptera": "../../datasets/Diptera/yolo-config.yaml",
    "Hymenoptera": "../../datasets/Hymenoptera/yolo-config.yaml",
    "Lepidoptera": "../../datasets/Lepidoptera/yolo-config.yaml",
}


def load_group_mapping(path):
    """Return a {group_name: data_yaml_path} dict from a YAML/JSON file."""
    if path is None:
        return dict(DEFAULT_GROUPS)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yml", ".yaml"}:
            content = yaml.safe_load(handle)
        else:
            content = json.load(handle)
    groups = content.get("groups", content)
    return {str(name): str(value) for name, value in groups.items()}


def read_data_yaml(data_yaml_path):
    """Parse a YOLO data.yaml and return resolved paths and keypoint shape."""
    data_path = Path(data_yaml_path).resolve()
    with data_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    val_path = data_path.parent / "images" / "val"

    kpt_shape = data.get("kpt_shape", [None, 3])
    n_kpts, kpt_dim = int(kpt_shape[0]), int(kpt_shape[1])

    # "names" holds object class names; keypoint labels live under "kpt_names",
    # keyed by class index, each value being a list of keypoint names.
    kpt_names_field = data.get("kpt_names", {})
    if isinstance(kpt_names_field, dict) and kpt_names_field:
        first_class = sorted(kpt_names_field.keys())[0]
        kpt_names = list(kpt_names_field.get(first_class, []))
    elif isinstance(kpt_names_field, list):
        kpt_names = list(kpt_names_field)
    else:
        kpt_names = []

    return {
        "val_path": val_path,
        "n_kpts": n_kpts,
        "kpt_dim": kpt_dim,
        "kpt_names": kpt_names,
        "raw": data,
        "yaml_path": data_path,
    }


def list_val_images(val_path):
    """Resolve the validation split into a list of image file paths."""
    if val_path.is_dir():
        return sorted(p for p in val_path.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTENSIONS)
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        with val_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        return [Path(line) for line in lines]
    return []


def label_path_for_image(image_path):
    """Map an image path to its YOLO label file path (manual fallback)."""
    parts = list(image_path.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def resolve_val_images(data_yaml, info):
    """Resolve validation images the same way model.val() does."""
    try:
        from ultralytics.data.utils import check_det_dataset
        data = check_det_dataset(data_yaml)
        val = data.get("val")
        root = Path(data.get("path", Path(data_yaml).parent))
        sources = list(val) if isinstance(val, (list, tuple)) else [val]

        images = []
        for source in sources:
            source = Path(source)
            if source.is_dir():
                images += [p for p in source.rglob("*")
                           if p.suffix.lower() in IMAGE_EXTENSIONS]
            elif source.suffix.lower() == ".txt" and source.exists():
                for line in source.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    path = Path(line)
                    if not path.is_absolute():
                        path = (root / line.lstrip("./")).resolve()
                    images.append(path)
            elif source.exists():
                images.append(source)
        return sorted({p for p in images if p.exists()})
    except Exception:
        return list_val_images(info["val_path"])


def labels_for_images(image_paths):
    """Map image paths to label paths using Ultralytics' own helper."""
    try:
        from ultralytics.data.utils import img2label_paths
        return [Path(p) for p in img2label_paths([str(p) for p in image_paths])]
    except Exception:
        return [label_path_for_image(p) for p in image_paths]


def resolve_split_dirs(data_yaml_path, split):
    """Return the absolute image directories of one split of a data.yaml."""
    data_path = Path(data_yaml_path).resolve()
    with data_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    if not root.exists():
        root = data_path.parent

    entries = data.get(split, f"images/{split}")
    entries = entries if isinstance(entries, (list, tuple)) else [entries]

    resolved = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            candidates = [(root / entry).resolve(),
                          (data_path.parent / entry).resolve()]
            path = next((c for c in candidates if c.exists()), candidates[0])
        resolved.append(path)
    return resolved


def build_combined_yaml(groups, out_path, verbose=True):
    """Merge several per-group pose datasets into one data.yaml.

    YOLO accepts a list of directories for ``train``/``val``, so no file is
    copied or duplicated on disk: the merged yaml simply points at all four
    source folders at once.

    The keypoint layout (kpt_shape, flip_idx) is taken from the first group and
    checked against the others; a mismatch raises, because silently training on
    inconsistent keypoint definitions is the single most damaging thing that can
    happen to this pipeline.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reference = None
    train_dirs, val_dirs = [], []

    for name, data_yaml in groups.items():
        with Path(data_yaml).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        signature = (tuple(data.get("kpt_shape", [])),
                     tuple(data.get("flip_idx", []) or []))
        if reference is None:
            reference = (signature, data)
        elif signature != reference[0]:
            raise ValueError(
                f"Group '{name}' has kpt_shape/flip_idx {signature}, which "
                f"differs from {reference[0]}. All groups must share the same "
                f"keypoint definition to train a shared backbone.")

        train_dirs += [str(p) for p in resolve_split_dirs(data_yaml, "train")]
        val_dirs += [str(p) for p in resolve_split_dirs(data_yaml, "val")]

    base = reference[1]
    merged = {
        "path": str(out_path.parent),
        "train": train_dirs,
        "val": val_dirs,
        "kpt_shape": list(base.get("kpt_shape", [])),
        "names": base.get("names", {0: "insect"}),
    }
    if base.get("flip_idx"):
        merged["flip_idx"] = list(base["flip_idx"])
    if base.get("kpt_names"):
        merged["kpt_names"] = base["kpt_names"]

    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False, allow_unicode=True)

    if verbose:
        print(f"[dataset] combined yaml written to {out_path}")
        print(f"[dataset]   {len(train_dirs)} train source(s), "
              f"{len(val_dirs)} val source(s)")
        if not base.get("flip_idx"):
            print("[dataset]   WARNING: no flip_idx found. Do NOT enable "
                  "fliplr augmentation without one -- bilateral keypoints "
                  "would be swapped silently.")
    return out_path


# Backwards-compatible alias.
_resolve_split_dirs = resolve_split_dirs
