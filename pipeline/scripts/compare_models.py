"""Compare tous les modeles entraines et produit les heatmaps et figures de cout.

Ce script n'est qu'une interface : toute la logique vit dans
`insectpose.reporting.compare` et `insectpose.reporting.figures`, donc elle est testee
et reutilisable ailleurs (notebook, autre script). Il exige que `cli report` ait deja
tourne, puisqu'il lit `results/master.parquet` sans le regenerer.

Exemples
--------
python scripts/compare_models.py
python scripts/compare_models.py --tags surface_f0 --out-dir results/comparison_surface
python scripts/compare_models.py --approaches yolo_pooled lora --metrics oks_ap
python scripts/compare_models.py --exclude-keypoints leg hindwing
python scripts/compare_models.py --no-cost-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from insectpose.paths import ProjectPaths
from insectpose.reporting.compare import CompareFilter, load_master, write_comparison
from insectpose.utils.logging import get_logger, setup_logging

log = get_logger("compare_models")


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
    parser.add_argument("--cost-metric", default="oks_ap",
                        help="Metrique portee en ordonnee des figures de cout.")
    parser.add_argument("--no-cost-figures", action="store_true",
                        help="Ne pas produire les figures performance vs cout.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def write_cost_figures(paths: ProjectPaths, selection: CompareFilter, out_dir: Path,
                       metric: str, dpi: int) -> list[Path]:
    """Figures performance vs cout, sur le sous-ensemble filtre.

    Deux axes, car aucun ne suffit seul : le temps d'entrainement separe reellement les
    approches, tandis que le nombre de parametres specialises isole le cout de la
    specialisation par groupe. Le temps d'INFERENCE, lui, ne discrimine pas — un modele
    LoRA fusionne fait le meme forward qu'un modele entraine entierement.

    Effet de bord : ecrit dans `out_dir`.
    """
    from insectpose.reporting.figures import (
        fig_performance_vs_specialisation,
        fig_performance_vs_training_cost,
    )

    data = selection.apply(load_master(paths))
    if data.empty:
        return []
    produced = [
        fig_performance_vs_training_cost(paths, data, out_dir, metric, selection.split, dpi),
        fig_performance_vs_specialisation(paths, data, out_dir, metric, selection.split, dpi),
    ]
    return [p for p in produced if p is not None]


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
    out_dir = Path(args.out_dir) if args.out_dir else (paths.results / "comparison")

    figures = write_comparison(
        paths, selection, out_dir,
        dpi=args.dpi, per_dataset_keypoints=not args.no_per_dataset_keypoints,
    )

    if not args.no_cost_figures:
        # `--metrics` filtre les heatmaps ; la metrique de cout est independante, sinon
        # restreindre les heatmaps supprimerait aussi les figures de cout.
        cost = write_cost_figures(paths, selection, out_dir, args.cost_metric, args.dpi)
        if not cost:
            log.info("Figures de cout non produites : metrique '%s' absente des runs "
                     "selectionnes, ou manifestes manquants.", args.cost_metric)
        figures.extend(cost)

    for path in figures:
        print(path)


if __name__ == "__main__":
    main()