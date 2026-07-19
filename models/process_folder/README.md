# Insect morphometry pipeline

Runs a **trained YOLO-pose model** over a folder of images and writes one CSV
row per image: every measurement in pixels and millimetres, a confidence value
per measurement, an overall pose confidence, and the detected scale (px/mm) with
its own confidence. No training happens here — models are only loaded.

## Expected folder layout

```
project_root/
├── process_folder.py            # run this
├── processing/                  # the package (see below)
├── trained_models/
│   └── XXX.pt                   # your pose model (set the name in config.py)
├── scale_bar_detection/
│   └── best.pt                  # the YOLO scale-bar detector
├── datasets/                    # standard YOLO datasets (for train/val columns)
│   └── <dataset>/images/{train,val}/...
└── images_to_process/           # the images you want to measure
```

## Quick start

1. Edit **`processing/config.py`** — at least `POSE_MODEL_NAME` and, if needed,
   `INPUT_FOLDER`.
2. Run — pick a source with `--source`:

```bash
# LOCAL folder (default)
python process_folder.py --source folder --input images_to_process

# HUGGING FACE dataset, streamed into RAM, 16 parallel workers
python process_folder.py --source hf --dataset TomBellivier/all_images --workers 16

# only sub-folders 1 and 2 of the dataset
python process_folder.py --source hf --dataset TomBellivier/all_images --hf-folders 1 2
```

Common options: `--output results.csv`, `--model XXX.pt`,
`--workers 16`, `--buffer 32`, `--torch-threads N` (compute threads per worker),
`--hf-token` (or env `HF_TOKEN`) for private repos.

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
  model is not safe to call from several threads at once) and splits the CPU
  cores across workers (`cpus // workers`) so the 16 CPUs are used without
  oversubscription. Lower `--workers` if RAM is tight (fewer model copies);
  raise it to overlap more downloads/inferences.

The pixel/scale pipeline itself now works on an **in-memory image** (a decoded
BGR array) rather than a file path, which is what makes disk-free HF streaming
possible.

## Module map

| File | Role |
|------|------|
| `config.py` | **All tunable parameters.** Edit this first. |
| `definitions.py` | Keypoint order/colours, the 27 measurements, the L/R flip map. |
| `measurements.py` | Keypoints → pixel measurements (sum of segments). |
| `pose_inference.py` | Load the pose model, pick the instance, return keypoints. |
| `confidence.py` | **All confidence formulas** (see below). |
| `tta.py` | Test-time augmentation for the TTA confidence signal. |
| `scale_bar_detection_utils.py` | Scale-bar detector (adapted from your file). |
| `ruler_detection.py` | Ruler / Fourier detector (adapted from your file). |
| `scale.py` | Scale-bar → ruler fallback, returns scale + confidence. |
| `dataset_membership.py` | Train/val membership by exact file name. |
| `pipeline.py` | The per-image pipeline, operating on an in-memory BGR array. |
| `image_source.py` | Local folder **or** Hugging Face dataset, unified. |
| `parallel.py` | Bounded, multi-thread, as-completed task runner. |
| `worker.py` | Per-thread model copies + CPU-thread balancing. |
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

## Bugs fixed in the adapted detector files

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
- The train/val counts printed at startup look right for your datasets.
