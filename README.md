# CV-for-GRIT
Computer-vision models to measure insect traits from images, to help compile the Global Repository of Insect Traits.

From a photo of a pinned insect, the project predicts 42 anatomical keypoints, derives 27 morphometric measurements from them, flags the measurements that cannot be trusted, detects the scale (scale bar or ruler) and converts the measurements to millimetres.

## Contents
1. [Workflow at a glance](#workflow-at-a-glance)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Run everything with one command](#run-everything-with-one-command)
5. [Keypoints and measurements](#keypoints-and-measurements)
6. [Hardware](#hardware)
7. [Step 1 — Organise the images](#step-1--organise-the-images)
8. [Step 2 — Annotate the keypoints in Label Studio](#step-2--annotate-the-keypoints-in-label-studio)
9. [Step 3 — Classify the measurements](#step-3--classify-the-measurements)
10. [Step 4 — Annotate the scale](#step-4--annotate-the-scale)
11. [Step 5 — Gather every annotation into one table and check it](#step-5--gather-every-annotation-into-one-table-and-check-it)
12. [Step 6 — Train the pose models](#step-6--train-the-pose-models)
13. [Step 7 — Train the measurement-validity classifiers](#step-7--train-the-measurement-validity-classifiers)
14. [Step 8 — Scale detection](#step-8--scale-detection)
15. [Step 9 — Run the pipeline on new images](#step-9--run-the-pipeline-on-new-images)
16. [Results](#results)
17. [Running inference only](#running-inference-only)

Unless stated otherwise, every command is run **from the repository root**.

## Workflow at a glance

| Step | What | Tool | Output |
|---|---|---|---|
| 1 | Organise the images | `annotated_images/split_image_database.py` | `annotated_images/full databases/<group>/...` |
| 2 | Annotate the keypoints | Label Studio + `annotation_tools/` | `annotation_data/label_studio_annotations/<group>/*.json` |
| 3 | Classify each measurement as measurable or not | `annotation_tools/measurement_validation/` | `annotation_data/meas_classifier/*_measurements.csv` |
| 4 | Annotate the scale | by hand | `annotation_data/scale/scale_annotations.csv` |
| 5 | Merge every annotation, then check it | `annotation_tools/build_annotation_data.py`, `check_annotations.py` | `annotation_data/annotation_data.csv`, `results/annotation_check/` |
| 6 | Train the pose models | `modules/architectures` | `retained_models/pose/` |
| 7 | Train the measurement-validity classifiers | `modules/meas_classifier` | `retained_models/measurement_validity/` |
| 8 | Provide the scale-bar detector | trained outside this repository | `retained_models/scale_bar/best.pt` |
| 9 | Measure new images | `pipeline/` | one CSV row per image |

Steps 6 to 9 only need `annotation_data/annotation_data.csv`, which is versioned: to retrain or run the models on the current annotations, start at [Installation](#installation), then go to step 6. Steps 5 to 9 and the analyses run in one command: see [Run everything with one command](#run-everything-with-one-command).

## Repository layout

- `kp_infos.yaml`, `kp_infos.py` — the single definition of the keypoints, skeleton and measurements (see [below](#keypoints-and-measurements)).
- `run_config.yaml`, `run_all.py`, `run_windows.bat`, `run_linux.sh`, `run_macos.sh` — the settings and the launchers of the end-to-end run (see [below](#run-everything-with-one-command)).
- `annotated_images/` — the image databases (not versioned) and `split_image_database.py`.
- `annotation_data/` — every annotation:
  - `label_studio_annotations/<group>/` — Label Studio JSON exports, one folder per insect order;
  - `pose/pose_annotations.csv` — those exports converted to CSV;
  - `meas_classifier/` — the measurable / non-measurable status of every measurement, one CSV per annotation batch;
  - `scale/scale_annotations.csv` — the scale annotations, written by hand (see `annotation_data/scale/README.md`);
  - `whole_pipeline/pipeline_gt.csv` — an older scale ground truth, already folded into `scale/scale_annotations.csv`;
  - `annotation_data.csv` — the merged table every training module reads.
- `annotation_tools/` — tools used around the annotation campaigns: the Label Studio template and launcher, `import_volunteer_file.py`, `labelstudio_to_csv.py`, `measurement_validation/` (a desktop app), `build_annotation_data.py` and `check_annotations.py`.
- `modules/architectures/` — pose-model training and comparison (`insectpose` package; see `modules/architectures/README.md`, `CONVENTIONS.md` and `DECISIONS.md`).
- `modules/meas_classifier/` — the measurement-validity classifiers.
- `modules/ruler_detection/`, `modules/scale_bar_detection/` — scale detection (ruler by Fourier analysis; scale bar by YOLO and OCR). The pipeline imports them from here.
- `retained_models/` — the trained models the pipeline runs on (see `retained_models/README.md`).
- `pipeline/` — end-to-end inference over a folder of images (see `pipeline/processing/README.md`).
- `images_to_process/` — the default input folder of the pipeline.
- `results/` — every data analysis produced by the modules, one sub-folder per module (see `results/README.md`).

Images (`*.jpg`, `*.png`), model weights (`*.pt`, `*.joblib`) and the `.venv` are not versioned.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e "modules/architectures[dev]"
```

- `requirements.txt` covers every module, the pipeline and the annotation tools (including Label Studio and PySide6).
- The second command installs the `insectpose` package of the pose module; `[dev]` adds `pytest`, `ruff` and `mypy`.
- **NVIDIA GPU:** the default `pip install torch` may install a CPU-only build. Install the CUDA build of `torch` and `torchvision` first, with the command given on pytorch.org, then run the two commands above. Without a GPU everything still runs, on the CPU.

## Run everything with one command
Once the annotations exist (steps 1 to 4), one command runs the rest of the project, in this order:

| Step | What it does |
|---|---|
| `annotations` | converts the Label Studio exports and rebuilds `annotation_data.csv` (step 5) |
| `images` | fills the training image folders of the pose module from `annotated_images/full databases/` |
| `check` | checks every annotation file and reports the images missing from some of them (step 5) |
| `pose` | trains the pose models, or optimises them, on one or several folds (step 6) |
| `classifiers` | trains the measurement-validity classifiers (step 7) — optional |
| `pipeline` | measures the images to process; scale extraction optional (step 9) |
| `analysis` | pose report and comparison, Optuna plots, analysis of the pipeline output |

Every setting is in **`run_config.yaml`**, at the root, commented line by line. The main ones:
- `pose_training.mode`: `train` (fixed hyperparameters) or `optimise` (Optuna search, then one training per fold);
- `pose_training.folds`: the folds to train among the 5 of the protocol, e.g. `[0]`, `[0, 1, 2]` or `"all"`. Every fold trained is one model of the ensemble the pipeline averages;
- `scale_extraction.enabled` / `method`: scale bar then ruler (`auto`), one of them, or no scale at all;
- `measurement_classification.enabled` / `train`: use (and retrain) the measurement-validity classifiers or not;
- `pipeline.source` / `input` / `output`: the images to measure and the result CSV.

Then run the launcher of your system (it moves to the repository root first, so it can be called from anywhere):

```bash
run_windows.bat          # Windows (cmd, or .\run_windows.bat in PowerShell)
bash run_linux.sh        # Linux
bash run_macos.sh        # macOS
```

The launchers use `.venv` when it exists (else the Python of the PATH) and pass every option on to `run_all.py`:
- `--dry-run` prints every command without running anything: check it first;
- `--only pose analysis` runs only these steps, `--skip annotations check` all but these;
- `--config my_run.yaml` reads another settings file (paths relative to the repository root).

Each run writes all its commands and their output to `results/run_logs/run_<date>.log`, and ends with a summary of the steps. The run stops at the first failing step with the reason, e.g. images missing from the training folders, the annotation check finding an error, or the scale-bar detector missing.

## Keypoints and measurements
`kp_infos.yaml`, at the root, is the ONLY definition of the keypoints (order, difficulty, left/right symmetry, colour), the skeleton, the measurements, their left/right pairs, the anatomical parts and the insect groups (`coleoptera`, `diptera`, `hymenoptera`, `lepidoptera`). Every module reads it — directly, or through `kp_infos.py` (`from kp_infos import KEYPOINT_NAMES, MEASUREMENTS, ...`).

- The keypoint order is baked into the trained models: append new points at the end, never reorder or rename.
- The Label Studio labelling interface, `annotation_tools/label_studio_template.txt`, is not generated from it: when a keypoint is added, add its label to the template too, with the same name.

## Hardware
Nothing is tied to a machine: every module detects what it runs on (CUDA GPU(s), Apple GPU, or CPU only, with any number of cores) and sizes itself.
- Pose training (`modules/architectures`): `train.device: auto` and `train.num_workers: auto`; mixed precision on GPU only. The batch size stays at 16 on purpose, so that runs remain comparable.
- Pipeline: `pipeline/processing/hardware.py` picks the device(s), the number of worker threads, the compute threads per worker and the images in flight from the GPUs, the CPUs and the free RAM/VRAM. The plan is printed at start-up; `--workers`, `--buffer` and `--torch-threads` force a value.
- Measurement-validity classifiers: every CPU (`n_jobs=-1`).

## Step 1 — Organise the images
The images live in `annotated_images/full databases/`, one sub-folder per insect order:

```
annotated_images/full databases/
├── coleoptera/...
├── diptera/...
├── hymenoptera/...
└── lepidoptera/...
```

Nested sub-folders are allowed. Two tools find images there:
- the pipeline, which looks up the insect order of every image by its file name;
- the ruler-detection evaluation.

File names must therefore be unique across the databases.

To hand a large folder out to annotators in batches, split it into numbered sub-folders of at most N images each:

```bash
python annotated_images/split_image_database.py "annotated_images/<folder>" <N>             # 01/, 02/, ... with at most N images each
python annotated_images/split_image_database.py "annotated_images/<folder>" <N> --dry-run   # show what would be moved, move nothing
python annotated_images/split_image_database.py "annotated_images/<folder>" --reverse       # move the images back up, delete the empty sub-folders
```

## Step 2 — Annotate the keypoints in Label Studio

**Set up Label Studio (once):**
1. Start it with `annotation_tools\start_label_studio.bat` on Windows. On Linux or macOS, run `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start`. Local file serving must be enabled for Label Studio to show local images.
2. Create a project. Under *Settings > Labeling Interface > Code*, paste the content of `annotation_tools/label_studio_template.txt`.
3. Under *Settings > Cloud Storage > Add Source Storage > Local*, add the folder holding the images.

**Bring a volunteer's export back to this machine:** a volunteer's JSON export points at the images of *their* Label Studio instance. Re-point it at the local images:

```bash
python annotation_tools/import_volunteer_file.py --json_file "<export>.json" --image_folders "<image folder>" ["<other folder>" ...]
```

- Each folder is searched recursively. Images are matched on their file name, ignoring the 8-character hash Label Studio adds on upload.
- The result is written next to the input, as `<export>_local.json` (`--output` to change it).
- If Label Studio runs with `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` set, pass the same value with `--document_root`.

Import that file into the project: every task should show its image. Check and correct the keypoints.

**Save the result:** export the project as JSON and put the file in `annotation_data/label_studio_annotations/<group>/`, where `<group>` is the insect order (`coleoptera`, `diptera`, `hymenoptera` or `lepidoptera`). The name of that folder becomes the `group` of every image of the export.

## Step 3 — Classify the measurements
For each annotated image, a person marks every measurement as *measurable* or *non measurable* (e.g. a leg hidden under the body). The measurement-validity classifiers learn from these labels.

1. Convert the export of the batch to CSV (keypoints in pixels, one row per annotation):
   ```bash
   python annotation_tools/labelstudio_to_csv.py --json_files "annotation_data/label_studio_annotations/<group>/<batch>.json" --output "<batch>.csv"
   ```
2. Open the classification app on that CSV (a file dialog opens if `--annotations_csv` is omitted):
   ```bash
   python annotation_tools/measurement_validation/main.py --annotations_csv "<batch>.csv"
   ```
   - Each image is shown with its keypoints and the segments of every measurement. The image is read from the `image_path` column; if it is not found, the skeleton is drawn on a grey background and can still be classified.
   - Left click/drag on a segment marks it measurable, right click/drag marks it non measurable. Leg and antenna segments cascade to the whole measurement.
   - Clicking a measurement in the side panel toggles it as a whole. Presets store a recurring pattern.
   - Keys: `Enter` validates the image and moves to the next one; `←`/`→` navigate; `Ctrl+Z`/`Ctrl+Y` undo/redo; `Ctrl+S` saves; `Ctrl+←`/`Ctrl+→` rotate the view. Mouse wheel zooms, middle button pans.
3. The result is written to `annotation_data/meas_classifier/<batch>_measurements.csv`: one row per annotation, one `<measurement>_status` column per measurement. Closing and reopening the app on the same CSV resumes where you stopped (state kept in `annotation_tools/measurement_validation/state/`).

## Step 4 — Annotate the scale
`annotation_data/scale/scale_annotations.csv` is filled by hand, one row per image. It holds:
- the true scale in pixels per millimetre;
- the scale type (`ruler` or `scale_bar`);
- for a scale bar, its box and text;
- for a ruler, its direction and the band of rows or columns it covers.

The ruler band is what the ruler-detection evaluation checks against. Columns and current content: `annotation_data/scale/README.md`.

## Step 5 — Gather every annotation into one table and check it
Every training module reads a single table, `annotation_data/annotation_data.csv`: one row per image, one column per piece of information, and an empty cell where nothing was annotated.

1. Convert every pose export to CSV. The merge only reads CSVs; the folder is searched recursively and each export takes the name of its folder as `group`:
   ```bash
   python annotation_tools/labelstudio_to_csv.py --json_files annotation_data/label_studio_annotations --output annotation_data/pose/pose_annotations.csv
   ```
2. Merge the three sources:
   ```bash
   python annotation_tools/build_annotation_data.py
   ```
   With no argument it reads:
   - pose: `annotation_data/pose/pose_annotations.csv`;
   - scale: `annotation_data/scale/scale_annotations.csv`;
   - measurements: every CSV in `annotation_data/meas_classifier/`.

   `--pose`, `--scale`, `--measurements` and `--output` override these paths.
3. Re-run steps 1 and 2 after every new annotation batch: the modules never read the individual files.
4. Check everything:
   ```bash
   python annotation_tools/check_annotations.py
   ```
   It checks each annotation file (columns, insect groups, image sizes, keypoints inside the image, measurement statuses, scale values, duplicates), then compares them.
   - **Images missing from some files:** it prints how many images are present in one source but missing from another:
     - Label Studio exports vs pose CSV;
     - pose vs measurement statuses;
     - pose vs scale;
     - annotation sources vs `annotation_data.csv`;
     - pose vs image files.

     It then lists them.
   - **Out-of-date table:** it rebuilds `annotation_data.csv` in memory and reports any difference, so a table that is out of date is caught.
   - **Classifiers:** it reports the measurements that will get no validity classifier for lack of labels.

   Each problem is an `ERROR` (wrong data, a step will fail), a `WARNING` (data silently lost or inconsistent) or an `INFO` (expected while annotating). The full lists are written to `results/annotation_check/issues.csv` and `presence.csv` (one row per image, one column per source). The exit code is 1 when there is an error (`--strict`: a warning too). `--no-image-files` skips the image checks.

## Step 6 — Train the pose models
The pose module compares several YOLO-pose approaches under a frozen protocol (5 grouped outer folds, Optuna hyperparameter search, fixed metrics). Its own documentation: `modules/architectures/README.md`, `CONVENTIONS.md` and `DECISIONS.md`.

**Run every command below from `modules/architectures/`:** its paths (`data/`, `runs/`, `configs/`) are relative to that folder. Alternatively, set `INSECTPOSE_ROOT`, or pass `paths.root=<path>`.

**Images:** training needs the image files. Put the images of each order, flat, in `modules/architectures/data/raw/<group>/images/` (copy or link them). File names must match the `image_name` column of `annotation_data.csv`.

```bash
cd modules/architectures

# 1. annotation_data.csv -> canonical annotations of the 4 orders + keypoint coverage report
python -m insectpose.cli prepare

# 2. the folds shared by every approach (5 outer folds + inner folds for the search)
python -m insectpose.cli split

# 3a. quick run: train, predict and evaluate ONE fold
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0

# 3b. full protocol: hyperparameter search (20 trials on the inner folds of fold 0),
#     then one training per outer fold (5 models). Long: 20 x 3 + 5 trainings.
python -m insectpose.cli tune experiment=exp_a_yolo_pooled

# 4. aggregate every run into tables and figures (results/pose/)
python -m insectpose.cli report
```

**Several folds:** `train` also takes a list of folds, `folds=[0,1,2]` or `folds=all`, and `tune` retrains every fold unless `folds` restricts them. The number of folds itself stays 5 (protocol).

**Which models the pipeline uses:** `retained_models/pose/` holds an ensemble of `yolo_pooled` models, and every command replaces it:
- `train` replaces it with its model, one per fold with `folds=[...]`;
- `tune` replaces it with its fold models (5, or those of `folds`);
- `ensemble.json` lists the members and holds their cross-validated score;
- `python -m insectpose.cli evaluate run_id=<run_id>` re-evaluates an existing `yolo_pooled` run and makes it the ensemble, without retraining;
- `retain.enabled=false` trains without touching the ensemble in place.

**Other experiments:** `exp_b_yolo_per_dataset`, `exp_c_detect_then_pose`, `exp_d_lora`, `exp_e_group_bn`, `exp_f_yolo_reduced`, `exp_g_head_only`, `exp_h_lora_per_dataset` and `exp_ref_mean_pose` (in `configs/experiment/`). They are study objects: only `yolo_pooled` is exported to the pipeline.

**Analyses** (written to `results/pose/`):
```bash
python scripts/compare_models.py        # heatmaps and cost figures of every trained model
python plot_optuna.py                   # diagnostics of the hyperparameter searches
./compare_surface.sh                    # (bash) every approach on one fold, no search: a quick screening
```

**Tests:** `pytest -q` runs the full suite (about 10 minutes); `pytest -q -m smoke` runs only the end-to-end tests.

## Step 7 — Train the measurement-validity classifiers
One random forest per measurement predicts whether the measurement is trustworthy from the keypoint geometry and the insect order. They are trained from the `<measurement>_status` columns of `annotation_data.csv`:

```bash
python modules/meas_classifier/train_measure_validity.py [--min-per-class 20] [--n-folds 5]
```

- The models and their decision thresholds (`metrics.csv`) are written directly to `retained_models/measurement_validity/`, where the pipeline reads them.
- Metrics and figures go to `results/meas_classifier/training/`.
- A measurement gets a classifier only when each class (measurable / non measurable) has at least 20 annotated images. It gets one automatically at the next training once there are enough annotations.

To re-assess the choice of model, compare the 8 approaches originally evaluated (research only, saves no model; output in `results/meas_classifier/comparison/`):

```bash
python modules/meas_classifier/compare_measure_validity_approaches.py
```

## Step 8 — Scale detection
- **Scale bar:** a YOLO detector (a bar class and, optionally, a text class) followed by an OCR read of the value. The detector is trained outside this repository: put its weights at `retained_models/scale_bar/best.pt`. Check that `SCALE_BAR_BAR_CLASS_ID` / `SCALE_BAR_TEXT_CLASS_ID` in `pipeline/processing/config.py` match its classes (printed at load time). Without this file, run the pipeline with `--scale ruler` (`scale_extraction.method: ruler` in `run_config.yaml`, or `USE_SCALE_BAR = False` in that config): it then uses the ruler only.
- **Ruler:** Fourier analysis of the pixel rows; nothing to train. Its settings are `RULER_*` in `pipeline/processing/config.py`.
  - To evaluate it against the ruler bands of `annotation_data.csv`, run `python modules/ruler_detection/ruler_detection_evaluation.py`. It reads the images of `annotated_images/full databases/`, writes `results/ruler_detection/evaluation.json`, and `modules/ruler_detection/ruler_detection_evaluation.ipynb` plots it.
  - To recalibrate its confidence, run `python modules/ruler_detection/ruler_confidence.py --json <labels>.json --images_root <image folder>`. The JSON maps each image path, relative to `--images_root`, to `"0"` (nothing), `"1"` (scale bar) or `"2"` (ruler). The script prints thresholds to copy into `THRESHOLDS` in `ruler_confidence.py`.

## Step 9 — Run the pipeline on new images
The pipeline needs at least one pose model in `retained_models/pose/` (step 6) and, by default, the scale-bar detector (step 8). The measurement-validity classifiers (step 7) are optional: without them, the validity columns of the CSV stay empty.

```bash
# images of a local folder, searched recursively (default: images_to_process/)
python pipeline/process_folder.py --input "<image folder>" --output "<results>.csv"

# images of a Hugging Face dataset, streamed into memory (token: --hf-token or env HF_TOKEN)
python pipeline/process_folder.py --source hf --dataset TomBellivier/all_images [--hf-folders 1 2]
```

- Without `--input` / `--output`, it reads `images_to_process/` and writes `images_to_process/results.csv`.
- Two steps are optional. `--scale none` skips the scale extraction: no millimetre values, and the scale flags no image for review. `--scale scale_bar` or `--scale ruler` uses only one method; the default `auto` tries the scale bar, then the ruler. `--no-measurement-classifier` skips the validity classifiers.
- Every setting (thresholds, scale strategy, optional columns, TTA) is in `pipeline/processing/config.py`.
- `--models <folder or .pt>` runs another pose ensemble.

**Output:** one CSV row per image, with:
- every measurement in pixels (`[px]`) and in millimetres (`[mm]`), with a confidence (`[conf]`) and the validity predicted by the classifiers (probability and decision);
- the detected scale and its confidence;
- the mean and standard deviation of every keypoint coordinate over the pose ensemble (0 with a single model).

The insect order of each image is looked up in `annotated_images/full databases/<group>/`. If an image is not found there, its `<group>_one_hot` columns are all 0.

**Analyse a run:**
```bash
python pipeline/analyze_results.py --input "<results>.csv"
```

- Figures and `summary.txt` go to `results/pipeline/<CSV name>/`.
- For the images that are annotated in `annotation_data.csv`, it also measures the real error. That error is optimistic, since the models were trained on those annotations; the cross-validated score is `cv_estimate` in `retained_models/pose/ensemble.json`.
- `--no-gt` skips that part. `--print --images-dir "<image folder>"` also writes a copy of every image with its keypoints, measurements and scale drawn on it.

## Results
Every analysis is written to `results/`, one sub-folder per module:
- `annotation_check/` — step 5;
- `run_logs/` — the logs of the one-command runs;
- `pose/` — step 6;
- `meas_classifier/` — step 7;
- `ruler_detection/` — step 8;
- `pipeline/` — step 9.

`results/README.md` lists every file and the command that produces it.

## Running inference only
To measure images on another machine, without the training code, copy:
- `kp_infos.yaml` and `kp_infos.py`;
- `requirements.txt`;
- `pipeline/`;
- `modules/ruler_detection/` and `modules/scale_bar_detection/`;
- `retained_models/`.

Optionally, add `annotated_images/full databases/` for the insect-order columns. Then run `pip install -r requirements.txt` and step 9.
