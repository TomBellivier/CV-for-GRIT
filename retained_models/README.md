# retained_models

Trained artifacts that the **pipeline** runs on. Everything here is produced by
a module under `modules/`; nothing is trained in this folder and nothing is
edited by hand.

```
retained_models/
├── pose/                              <- ENSEMBLE of YOLO-pose models (modules/architectures)
│   ├── <run_id>/
│   │   ├── best.pt                    ... one model of the ensemble
│   │   └── model_card.json            ... run_id, approach, fold, metrics, keypoint schema
│   ├── <run_id>/ ...                  ... one folder per model (5 after a `tune`)
│   └── ensemble.json                  ... after `tune`: members + cross-validation estimate
├── scale_bar/
│   └── best.pt                        <- YOLO scale-bar detector
└── measurement_validity/              <- measurement-validity classifiers
    ├── rf_related_<measure>.joblib    ... one random forest per measurement
    └── metrics.csv                    ... their per-measurement decision thresholds
```

## Who writes what

| Folder | Produced by | How |
|--------|-------------|-----|
| `pose/` | `modules/architectures` | `train` / `evaluate run_id=...`: the folder is **replaced** by that single model |
| `pose/` | `modules/architectures` | `tune`: the folder is **replaced** by one model per outer fold (5 by default) |
| `measurement_validity/` | `modules/meas_classifier` | written directly by `python train_measure_validity.py` |
| `scale_bar/` | trained outside this repo | drop the `best.pt` file in by hand |

Only `yolo_pooled` runs are exported (`retain.approaches` in
`modules/architectures/configs/config.yaml`): one YOLO-pose model on the full
42-keypoint schema of `kp_infos.yaml`, which the pipeline loads like any other
member of the ensemble.

## Who reads it

`pipeline/processing/config.py` resolves every model path from this folder:

```python
RETAINED_MODELS_DIR = REPO_ROOT / "retained_models"
POSE_MODELS_DIR                    = RETAINED_MODELS_DIR / "pose"      # every *.pt under it
SCALE_BAR_MODEL_PATH               = RETAINED_MODELS_DIR / "scale_bar" / "best.pt"
MEASUREMENT_CLASSIFIER_DIR         = RETAINED_MODELS_DIR / "measurement_validity"
MEASUREMENT_CLASSIFIER_METRICS_CSV = MEASUREMENT_CLASSIFIER_DIR / "metrics.csv"
```

Nothing is picked by hand: the pipeline runs **every** `*.pt` under `pose/` on
each image, matches the same insect across the models, and writes the mean and
the standard deviation of every keypoint coordinate (0 with a single model).
Measurements, confidences and the measurement-validity classifiers all work on
the mean keypoints. `process_folder.py --models <folder or .pt>` points it at
another ensemble.

Each model's `model_card.json` says which run produced it and what it scored on
its held-out fold. After a `tune`, `ensemble.json` holds `cv_estimate`: the mean
and standard deviation of the primary metric over the outer folds.

Model files (`*.pt`, `*.joblib`) are ignored by git — they are rebuilt by
re-running the module that produced them.
