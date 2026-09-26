"""Compare every trained model and produce the heatmaps and cost figures.

This script is only an interface: all the logic lives in `insectpose.reporting.compare`
and `insectpose.reporting.figures`, so it is tested and reusable elsewhere (notebook,
other script). It requires `cli report` to have run already, since it reads
`results/master.parquet` without regenerating it.

Examples
--------
python scripts/compare_models.py
python scripts/compare_models.py --tags surface_f0 --out-dir results/comparison_surface
python scripts/compare_models.py --approaches yolo_pooled lora --metrics oks_ap
python scripts/compare_models.py --exclude-keypoints leg hindwing
python scripts/compare_models.py --no-cost-figures
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from insectpose.paths import POSE_RESULTS_DIR, ProjectPaths
from insectpose.reporting.compare import CompareFilter, load_master, write_comparison
from insectpose.utils.logging import get_logger, setup_logging

log = get_logger("compare_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--results", default=str(POSE_RESULTS_DIR),
                        help="Analysis folder of the module, where `report` wrote "
                             "master.parquet (default: <repo>/results/pose).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: results/comparison).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--approaches", nargs="*", default=[],
                        help="Keep only these approaches.")
    parser.add_argument("--tags", nargs="*", default=[], help="Keep only these tags.")
    parser.add_argument("--data-scopes", nargs="*", default=[],
                        help="Keep only these data scopes (pooled, coleoptera...).")
    parser.add_argument("--split-ids", nargs="*", default=[],
                        help="Keep only these splits.")
    parser.add_argument("--run-ids", nargs="*", default=[], help="Keep only these runs.")
    parser.add_argument("--exclude-keypoints", nargs="*", default=[],
                        help="Keypoint patterns to exclude from the per-point heatmaps "
                             "(e.g. leg hindwing). Adds a MEAN (retained) row.")
    parser.add_argument("--metrics", nargs="*", default=[],
                        help="Plot only these metrics (default: all).")
    parser.add_argument("--label-by", nargs="*", default=["approach", "tag"],
                        help="Fields making up the label of each model.")
    parser.add_argument("--no-per-dataset-keypoints", action="store_true",
                        help="A single keypoint heatmap, all datasets together.")
    parser.add_argument("--cost-metric", default="oks_ap",
                        help="Metric on the y axis of the cost figures.")
    parser.add_argument("--no-cost-figures", action="store_true",
                        help="Do not produce the performance vs cost figures.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def write_cost_figures(paths: ProjectPaths, selection: CompareFilter, out_dir: Path,
                       metric: str, dpi: int) -> list[Path]:
    """Performance vs cost figures, on the filtered subset.

    Two axes, because none is enough on its own: the training time really separates the
    approaches, while the number of specialised parameters isolates the cost of the
    per-group specialisation. The INFERENCE time does not discriminate — a merged LoRA
    model runs the same forward pass as a fully trained model.

    Side effect: writes into `out_dir`.
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
    paths = dataclasses.replace(ProjectPaths.default(args.root), results=Path(args.results))
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
        # `--metrics` filters the heatmaps; the cost metric is independent, otherwise
        # restricting the heatmaps would also remove the cost figures.
        cost = write_cost_figures(paths, selection, out_dir, args.cost_metric, args.dpi)
        if not cost:
            log.info("Cost figures not produced: metric '%s' missing from the selected "
                     "runs, or missing manifests.", args.cost_metric)
        figures.extend(cost)

    for path in figures:
        print(path)


if __name__ == "__main__":
    main()