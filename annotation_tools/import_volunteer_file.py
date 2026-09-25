#!/usr/bin/env python3
"""
import_volunteer_file.py
========================

Rewrite the image paths of a Label Studio export so it can be re-imported
locally.

A volunteer sends back a JSON export in which every task points at the images of
THEIR Label Studio instance (/data/upload/... or a local path that does not
exist here). This script looks for each image inside the given folders and
rewrites data.img as a local-storage URL, so that importing the produced JSON in
a local Label Studio project finds every image again.

Usage
-----
    python import_volunteer_file.py --json_file export.json \\
                                    --image_folders path/to/images path/to/more/images ...

    python import_volunteer_file.py --json_file export.json \\
                                    --image_folders D:/images \\
                                    --output checked.json

Any number of image folders can be given; each one is searched recursively.
Images are matched on their file name, ignoring the 8-character hash Label
Studio prepends on upload (e.g. "95c6e06b-IMG_0093.png" matches "IMG_0093.png").

Before importing the result in Label Studio, add a Local storage source pointing
at the image folder (Settings > Cloud Storage > Add Source Storage > Local).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_LOCAL_FILES_PREFIX = "/data/local-files/?d="
_HASH_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8}-")


def index_images(image_folders) -> dict:
    """Map every image file name found in the given folders to its path.

    Names are indexed both as-is and without the Label Studio upload hash. The
    first folder given wins when the same name appears twice.
    """
    index = {}
    for folder in image_folders:
        folder = Path(folder)
        if not folder.is_dir():
            print(f"/!\\ ignored (not a folder): {folder}")
            continue
        count = 0
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            count += 1
            for key in {path.name, _HASH_PREFIX_RE.sub("", path.name)}:
                if key in index and index[key] != path:
                    print(f"/!\\ duplicated image name '{key}': {index[key]} kept, {path} ignored")
                else:
                    index.setdefault(key, path)
        print(f"{count} image(s) found in {folder}")
    return index


def local_files_url(image_path: Path, document_root) -> str:
    """Encode an image path as a Label Studio local-storage URL.

    Label Studio serves local files relative to LOCAL_FILES_DOCUMENT_ROOT, which
    defaults to the root of the filesystem: the drive letter is therefore
    dropped unless another root is given with --document_root.
    """
    image_path = image_path.resolve()
    root = Path(document_root).resolve() if document_root else Path(image_path.anchor)
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        print(f"/!\\ {image_path} is not inside {root}, full path used instead")
        relative = image_path
    return _LOCAL_FILES_PREFIX + quote(str(relative), safe="")


def image_name(data_img: str) -> str:
    """File name of a data.img field, whatever the separator it was encoded with."""
    name = data_img.replace("%5C", "/").replace("\\", "/").rsplit("/", 1)[-1]
    return _HASH_PREFIX_RE.sub("", name)


def main():
    parser = argparse.ArgumentParser(
        description="Point the tasks of a Label Studio export at local image folders.")
    parser.add_argument("--json_file", required=True,
                        help="Label Studio JSON export to rewrite.")
    parser.add_argument("--image_folders", nargs="+", required=True,
                        help="One or more folders holding the images (searched recursively).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <json_file>_local.json).")
    parser.add_argument("--document_root", default=None,
                        help="Value of LOCAL_FILES_DOCUMENT_ROOT used by Label Studio "
                             "(default: the root of the drive holding the images).")
    args = parser.parse_args()

    json_file = Path(args.json_file)
    if not json_file.is_file():
        parser.error(f"JSON file not found: {json_file}")

    output = Path(args.output) if args.output else json_file.with_name(json_file.stem + "_local.json")

    index = index_images(args.image_folders)
    if not index:
        parser.error("no image found in --image_folders")

    with open(json_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    matched = 0
    missing = []
    for task in tasks:
        name = image_name(task.get("data", {}).get("img", ""))
        image_path = index.get(name)
        if image_path is None:
            missing.append(name)
            continue
        task.setdefault("data", {})["img"] = local_files_url(image_path, args.document_root)
        matched += 1

    with open(output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n{matched}/{len(tasks)} task(s) pointed at a local image -> {output}")
    if missing:
        print(f"{len(missing)} image(s) not found in the given folders:")
        for name in missing[:10]:
            print(f"  {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
