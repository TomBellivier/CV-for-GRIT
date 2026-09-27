# Insect morphometry pipeline

Runs an **ensemble of trained YOLO-pose models** over a folder of images and
writes one CSV row per image: the mean and standard deviation of every keypoint
coordinate over the models, every measurement in pixels and millimetres (from the
mean keypoints), a confidence value per measurement, an overall pose confidence,
and the detected scale (px/mm) with its own confidence. No training happens here
— models are only loaded.

## Expected folder layout

```
repo_root/
├── kp_infos.yaml, kp_infos.py    # keypoints / measurements (single definition of the repo)
├── pipeline/
│   ├── process_folder.py        # run this
│   └── processing/               # the package (see below)
├── modules/
│   ├── ruler_detection/          # ruler / Fourier detector (single copy, imported by scale.py)
│   └── scale_bar_detection/      # scale-bar detector + OCR (single copy, imported by scale.py)
├── retained_models/              # EVERY model this pipeline loads
│   ├── pose/
│   │   └── <run_id>/best.pt      # the pose ensemble: EVERY *.pt here is used
│   ├── scale_bar/
│   │   └── best.pt               # the YOLO scale-bar detector
│   └── measurement_validity/
│       ├── rf_related_*.joblib   # measurement-validity classifiers
│       └── metrics.csv           # their thresholds
├── images_to_process/            # the images you want to measure
└── annotated_images/full databases/    # image database, one folder per insect group:
    └── <group>/...               #   the group of each image is looked up here
```

`retained_models/` is filled by the modules under `modules/` (a pose training run
exports itself there, `train_measure_validity.py` writes there directly): the
pipeline never reads a model from inside `modules/`. See
`retained_models/README.md`.

## Quick start

1. Edit **`processing/config.py`** if needed (`INPUT_FOLDER`, thresholds...).
   The pose ensemble is whatever `modules/architectures` retained in
   `retained_models/pose/`: nothing to pick by hand.
2. Run — pick a source with `--source`:

```bash
# LOCAL folder (default)
python process_folder.py --source folder --input images_to_process

# HUGGING FACE dataset, streamed into RAM (workers sized to the machine)
python process_folder.py --source hf --dataset TomBellivier/all_images

# only sub-folders 1 and 2 of the dataset
python process_folder.py --source hf --dataset TomBellivier/all_images --hf-folders 1 2
```

Common options: `--output results.csv`, `--models <folder or .pt>` (another ensemble),
`--workers N`, `--buffer N`, `--torch-threads N` (force the automatic sizing, see below),
`--hf-token` (or env `HF_TOKEN`) for private repos.

Two steps are optional:

- `--scale auto|scale_bar|ruler|none` — scale extraction: scale bar then ruler (`auto`),
  one of them only, or none. With `none`, no detector runs, the millimetre columns stay
  empty, the scale confidence is empty ("not computed") and `scale_method` is
  `disabled`; the scale then flags no image for review. Default: `config.USE_SCALE_BAR`
  / `config.USE_RULER_FALLBACK`.
- `--no-measurement-classifier` — no measurement-validity classifier and no
  `<group>_one_hot` / validity columns. Default: `config.RUN_MEASUREMENT_CLASSIFIER`.

```bash
# pose and pixel measurements only
python process_folder.py --input images_to_process --scale none --no-measurement-classifier
```

Dependencies: `ultralytics`, `opencv-python`, `easyocr`, `numpy`, `scipy`,
`pillow`, and `huggingface_hub` (only for `--source hf`).

## Sources and parallelism

The input source and the parallel engine are decoupled, so both sources share
the exact same measurement code:

- **`image_source.py`** turns either a local folder or a HF dataset into a list
  of `(key, image_name)` plus a `load_fn(key) -> BGR array`. HF images are
  downloaded straight into RAM (nothing is written to disk), following the
  approach of your `test_process_hf.py`.
- **`parallel.py`** runs the whole task (load + decode + full pipeline) on a
  pool of `--workers` threads, keeping at most `--buffer` in flight and yielding
  results as they complete. This is the sliding-window idea of your script,
  extended so the inference is parallel too. Threads work well here because
  NumPy/SciPy/PyTorch release the GIL during heavy compute, and a thread blocked
  on a download lets another thread compute.
- **`worker.py`** gives each thread its own model copies (a single Ultralytics
  model is not safe to call from several threads at once), on the device the
  plan hands it.
- **`hardware.py`** sizes the run to the machine, once, at start-up (the plan is
  printed on the `Hardware` line):
  - **device**: every CUDA GPU (workers take them in turn), else the Apple GPU,
    else the CPU (`config.DEVICE`); FP16 on CUDA (`config.HALF_PRECISION_ON_GPU`);
    EasyOCR runs on the GPU when there is one;
  - **workers**: with a GPU, `config.WORKERS_PER_GPU` per GPU (never more than
    the CPUs); CPU only, one per `config.CPU_THREADS_PER_WORKER` cores (twice as
    many for a Hugging Face source). Always capped so that every worker's copy
    of the models fits in `config.MEMORY_BUDGET_FRACTION` of the free RAM / VRAM;
  - **threads**: PyTorch/OpenCV compute threads = CPUs // workers, so the workers
    share the cores instead of each grabbing all of them;
  - **buffer**: 2 × workers images in flight.

  On a single CPU this gives 1 worker × 1 thread; `--workers`, `--buffer` and
  `--torch-threads` (or `config.WORKERS`) force a value.

The pixel/scale pipeline itself now works on an **in-memory image** (a decoded
BGR array) rather than a file path, which is what makes disk-free HF streaming
possible.

## Module map

| File | Role |
|------|------|
| `config.py` | **All tunable parameters.** Edit this first. |
| `definitions.py` | Keypoint order/colours, measurements, L/R flip map — re-exported from `kp_infos.yaml`. |
| `measurements.py` | Keypoints → pixel measurements (sum of segments). |
| `pose_inference.py` | Load the pose ensemble, match the instance across models, return mean + std keypoints. |
| `confidence.py` | **All confidence formulas** (see below). |
| `tta.py` | Test-time augmentation for the TTA confidence signal. |
| `scale.py` | Scale-bar → ruler fallback, returns scale + confidence. |
| `pipeline.py` | The per-image pipeline, operating on an in-memory BGR array. |
| `image_source.py` | Local folder **or** Hugging Face dataset, unified. |
| `parallel.py` | Bounded, multi-thread, as-completed task runner. |
| `worker.py` | Per-thread model copies, on the device the plan hands out. |
| `hardware.py` | Sizes the run to the machine: devices, workers, threads, buffer. |
| `csv_writer.py` | Assemble and write the CSV. |

## Confidence methods (implemented in `confidence.py`)

**Per-measurement — two interchangeable signals.** Pick one with
`config.MEASUREMENT_CONFIDENCE_SIGNAL`:

- `"keypoint"` — aggregate the keypoint scores of the measurement. Aggregation
  set by `config.KEYPOINT_AGGREGATION` (`min` by default: a distance is only as
  good as its weakest endpoint). Per-keypoint visibility thresholds are
  hand-tunable in `config.PER_KEYPOINT_VISIBILITY_THRESHOLD`.
- `"tta"` — re-run inference on perturbed copies of the image (flip, small
  rotations, brightness), map every prediction back to the original frame,
  and turn the dispersion into a confidence
  `exp(-TTA_CV_BETA · cv)`. Enable with `config.ENABLE_TTA = True`.

**Overall pose** — `detection_conf × mean(keypoint_conf)` (or `min`, via
`config.OVERALL_POSE_METHOD`).

**Scale bar** — `bar_box_conf × text_box_conf × ocr_reliability` (a product, so
any weak step drops the confidence). Set the bar/text class ids in
`config.SCALE_BAR_BAR_CLASS_ID` / `SCALE_BAR_TEXT_CLASS_ID` to match your model
(they are printed as `model.names` at load time).

**Ruler** — one Fourier group → confidence `1`; several groups → mean relative
magnitude gap between the main group and the (up to 4) secondary groups.

**Millimetre values** — combined with `config.CONVERTED_CONF_METHOD`
(`min` of measurement and scale confidence by default). The measurement
confidence columns hold the raw measurement confidence; a commented block in
`process_folder.py` shows how to switch them to the combined mm confidence.

## Keypoint export and ground-truth analysis

With `config.EXPORT_KEYPOINTS = True` (default), the CSV also carries the
keypoints of the measured instance: `<kp> [kp_x]`, `<kp> [kp_y]`, `<kp> [kp_conf]`
(mean over the ensemble) and `<kp> [kp_x_std]`, `<kp> [kp_y_std]` (standard
deviation over the ensemble, 0 with a single model) — 5 × NUM_KEYPOINTS columns.
`n_pose_models` says how many models were averaged for the row: a model whose
instance does not overlap the reference one (`config.ENSEMBLE_MIN_IOU`) is left out.

`analyze_results.py` uses these, together with the annotated keypoints of
`annotation_data/annotation_data.csv` (`config.ANNOTATION_DATA_CSV`), to measure
real error on the images of the CSV that are annotated: it rebuilds the
ground-truth keypoints/measurements (rescaled when the `image_width/height`
columns show the image was processed at another size) and computes, among others, OKS vs overall confidence, per-keypoint error-vs-
confidence correlation and heatmap, per-measurement error, and needs-review vs
error, plus the per-keypoint disagreement of the ensemble. Run
`python analyze_results.py` (reads `config.OUTPUT_CSV`, writes to
`results/pipeline/<CSV name>/` at the repository root; `--input`/`--output-dir`
override them, `--no-gt` skips the ground-truth part). These errors are
optimistic: the pose models were trained on the same annotations. The
cross-validated score is `cv_estimate` in `retained_models/pose/ensemble.json`. OKS uses an uncalibrated falloff `--oks-kappa`
(default 0.05) since the bee keypoints have no standard COCO sigmas.

## Bugs fixed in the detector files (modules/ruler_detection, modules/scale_bar_detection)

- **Scale bar:** `_ensure_ocr_reader()` was called with no argument although it
  required one → replaced by a cached module-level reader.
- **Ruler:** `fft_dominant_frequency` returned 2 values on failure while the
  caller unpacked 3 → now returns `(None, None, None)`; and the "no group
  found" test now correctly ignores the `-1` (unclassified) label.

## Choosing a signal / calibrating (recommended next step)

The raw confidences are sensible, monotone signals but are **not yet
calibrated** (a `0.9` does not guarantee a given error). Once you can measure
the real error on the labelled **val** split, calibrate each signal by ranking
it against the true error (Spearman) and fitting a monotone map (e.g. isotonic
regression) so the reported number matches an actual error level. That is also
where the quadrature combination for millimetre errors
(`err_mm ≈ sqrt(err_px² + err_scale²)`) becomes meaningful.

## Things to verify on first run

- `model.names` printed for the pose model matches `KEYPOINT_NAMES` ordering.
- `model.names` printed for the scale-bar model matches the class ids in config.
- The `[group] <group>: N image(s) indexed` lines printed at startup look right for your
  image database.
