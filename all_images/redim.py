#!/usr/bin/env python3
"""
Copie un dossier d'images en redimensionnant chaque image en 640x640
(étirement/aplatissement, sans conservation du ratio).

L'arborescence d'origine est préservée. Les fichiers non-image
(ex: labels .txt YOLO) sont recopiés tels quels.

Usage:
    python redim.py /chemin/vers/dataset
    python redim.py /chemin/vers/dataset --size 640 --quality 90
"""

import argparse
import shutil
import sys
from pathlib import Path
import tqdm

from PIL import Image

# Extensions considérées comme images
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def process(src_root: Path, size: int, quality: int) -> None:
    dst_root = src_root.with_name(src_root.name + "_redim")

    if dst_root.exists():
        print(f"[!] Le dossier de sortie existe déjà : {dst_root}", file=sys.stderr)
        sys.exit(1)

    n_img, n_copy, n_err = 0, 0, 0

    src_files = list(src_root.rglob("*"))
    for src in tqdm.tqdm(src_files, desc="Traitement des fichiers"):
        if src.is_dir():
            continue

        # Chemin miroir dans le dossier de sortie
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.suffix.lower() in IMG_EXTS:
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")  # évite les soucis de mode (P, RGBA, L...)
                    im = im.resize((size, size), Image.LANCZOS)
                    # On sauvegarde dans le format d'origine quand c'est possible
                    if src.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
                        im.save(dst, quality=quality)
                    else:
                        im.save(dst)
                n_img += 1
            except Exception as e:
                print(f"[!] Erreur sur {src}: {e}", file=sys.stderr)
                n_err += 1
        else:
            # Fichier non-image (labels, .yaml, etc.) : copie conforme
            shutil.copy2(src, dst)
            n_copy += 1

    print(f"[✓] Terminé -> {dst_root}")
    print(f"    Images redimensionnées : {n_img}")
    print(f"    Fichiers copiés        : {n_copy}")
    if n_err:
        print(f"    Erreurs                : {n_err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copie un dossier en redimensionnant les images en NxN."
    )
    parser.add_argument("path", type=Path, help="Dossier source à traiter")
    parser.add_argument(
        "--size", type=int, default=640, help="Taille cible (défaut: 640)"
    )
    parser.add_argument(
        "--quality", type=int, default=90, help="Qualité JPEG/WebP (défaut: 90)"
    )
    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"[!] Dossier introuvable : {args.path}", file=sys.stderr)
        sys.exit(1)

    process(args.path.resolve(), args.size, args.quality)


if __name__ == "__main__":
    main()
