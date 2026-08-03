"""Compare tous les modeles entrainees et produit les heatmaps.

Ce script n'est qu'une interface : toute la logique vit dans
`insectpose.reporting.compare`, donc elle est testee et reutilisable ailleurs
(notebook, autre script). Il exige que `cli report` ait deja tourne.

Exemples
--------
python scripts/compare_models.py
python scripts/compare_models.py --approaches yolo_pooled lora --metrics oks_ap
python scripts/compare_models.py --tags pooled_yolo-tuned --label-by approach tag fold
"""

from __future__ import annotations

import argparse
from pathlib import Path

from insectpose.paths import ProjectPaths
from insectpose.reporting.compare import CompareFilter, write_comparison
from insectpose.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="Racine du projet.")
    parser.add_argument("--out-dir", default=None,
                        help="Repertoire de sortie (defaut : results/comparison).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--approaches", nargs="*", default=[],
                        help="Ne garder que ces approches.")
    parser.add_argument("--tags", nargs="*", default=[], help="Ne garder que ces etiquettes.")
    parser.add_argument("--data-scopes", nargs="*", default=[],
                        help="Ne garder que ces perimetres de donnees (pooled, coleoptera...).")
    parser.add_argument("--split-ids", nargs="*", default=[],
                        help="Ne garder que ces decoupages.")
    parser.add_argument("--run-ids", nargs="*", default=[], help="Ne garder que ces runs.")
    parser.add_argument("--exclude-keypoints", nargs="*", default=[],
                        help="Motifs de keypoints a exclure des heatmaps par point "
                             "(ex. leg hindwing). Ajoute une ligne MEAN (retained).")
    parser.add_argument("--metrics", nargs="*", default=[],
                        help="Ne tracer que ces metriques (defaut : toutes).")
    parser.add_argument("--label-by", nargs="*", default=["approach", "tag"],
                        help="Champs composant l'etiquette de chaque modele.")
    parser.add_argument("--no-per-dataset-keypoints", action="store_true",
                        help="Une seule heatmap keypoints, tous datasets confondus.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    paths = ProjectPaths.default(args.root)
    selection = CompareFilter(
        approaches=tuple(args.approaches), tags=tuple(args.tags),
        data_scopes=tuple(args.data_scopes), split_ids=tuple(args.split_ids),
        run_ids=tuple(args.run_ids), split=args.split,
        label_by=tuple(args.label_by), metrics=tuple(args.metrics),
        exclude_keypoints=tuple(args.exclude_keypoints),
    )
    figures = write_comparison(
        paths, selection, Path(args.out_dir) if args.out_dir else None,
        dpi=args.dpi, per_dataset_keypoints=not args.no_per_dataset_keypoints,
    )
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()