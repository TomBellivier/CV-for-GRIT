# results

Data analyses of every module, in one place. Nothing here is read back by a
model or by the pipeline: these are tables, figures and reports for people.
Every script that produces an analysis writes into its sub-folder by default
(an `--output-dir` / `--out-dir` option still overrides it).

```
results/
├── annotation_check/             <- annotation_tools/check_annotations.py
│   ├── issues.csv                ... every problem found, one row per image
│   └── presence.csv              ... one row per image, one column per annotation source
├── run_logs/                     <- run_all.py: every command and its output, one log per run
├── pose/                         <- modules/architectures (pose models)
│   ├── master.parquet            ... every evaluated run, one row per metric (`cli report`)
│   ├── summary_*.parquet         ... report tables (`cli report`)
│   ├── figures/, runs/<run_id>/  ... report figures, global and per run (`cli report`)
│   ├── comparison/               ... heatmaps and cost figures (scripts/compare_models.py)
│   ├── optuna/<study>/           ... HPO diagnostics (plot_optuna.py)
│   └── reports/                  ... paths.reports
├── meas_classifier/              <- modules/meas_classifier (measurement validity)
│   ├── training/                 ... metrics, PR curves, confusion, importances (train_measure_validity.py)
│   └── comparison/               ... the 8 approaches compared (compare_measure_validity_approaches.py)
├── pipeline/                     <- pipeline/analyze_results.py
│   └── <results CSV name>/       ... confidence, scale, ensemble spread, GT error figures + summary.txt
└── ruler_detection/              <- modules/ruler_detection/ruler_detection_evaluation.py
    ├── evaluation.json           ... read by ruler_detection_evaluation.ipynb
    └── problems.json
```

## Commands

| Analysis | Command (from the folder of the script) |
|----------|------------------------------------------|
| Everything below, in one run | `python run_all.py` (at the root; settings in `run_config.yaml`) |
| Annotation check | `python annotation_tools/check_annotations.py` (at the root) |
| Pose models: report | `python -m insectpose.cli report` (in `modules/architectures`) |
| Pose models: comparison | `python scripts/compare_models.py` (in `modules/architectures`) |
| Pose models: HPO | `python plot_optuna.py` (in `modules/architectures`) |
| Measurement validity | `python train_measure_validity.py` / `python compare_measure_validity_approaches.py` (in `modules/meas_classifier`) |
| Pipeline output | `python analyze_results.py [--input <results CSV>]` (in `pipeline`) |
| Ruler detection | `python ruler_detection_evaluation.py` (in `modules/ruler_detection`) |

The pipeline's own output (one row per image, with the mean and standard
deviation of every keypoint over the pose ensemble) is `config.OUTPUT_CSV`
(`images_to_process/results.csv` by default), next to the images it measures;
`analyze_results.py` reads it by default.
