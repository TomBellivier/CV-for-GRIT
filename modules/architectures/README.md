# insectpose

Experimental framework to compare several pose-estimation approaches on 4 insect datasets
(Coleoptera, Diptera, Hymenoptera, Lepidoptera).

**Read `CONVENTIONS.md` before any contribution.** This file only holds the quick start; the
whole doctrine (contracts, rules, protocol) is in `CONVENTIONS.md`, which is authoritative.

## Installation

```bash
pip install -e ".[dev]"
```

torch and ultralytics are first-rank dependencies (ADR-0019). Everything adapts to the
machine: `train.device: auto` uses GPU 0 if there is one, else the Apple GPU (`mps`), else
the CPU; `train.num_workers: auto` takes one data-loading worker per CPU minus one (capped at
8, 0 on a single-CPU machine); mixed precision is on on GPU, off on CPU and in `mode: debug`.
`train.batch_size` stays fixed at 16: two runs only compare at an equal batch size. The
resolved hardware (GPU, number of CPUs) is recorded in every manifest, and the aggregation
warns if compared runs come from different hardware.

## Data source

`prepare` reads the single annotation table of the repository,
`../../annotation_data/annotation_data.csv` (`annotation_csv` adapter), and derives the
canonical annotations of contract 1 from it: one row per image, filtered on the `group`
column. The scale and measurement-validity columns of that table do not concern the pose and
are ignored.

Rebuild the table after an annotation campaign:
`python annotation_tools/build_annotation_data.py` at the repository root. The `coco` and
`yolo` adapters remain available (`data.adapter=yolo`) to read a corpus in its raw format.

## Full chain

```bash
# 1. annotation_data.csv -> canonical format of the 4 orders (contract 1)
#    (data=coleoptera etc. prepares a single order)
python -m insectpose.cli prepare

# 2. folds shared by ALL the approaches (contract 2)
python -m insectpose.cli split

# 3. training + prediction + evaluation of one fold, or of several (ADR-0039)
#    experiments: configs/experiment/ (exp_a_yolo_pooled ... exp_h_lora_per_dataset,
#    exp_ref_mean_pose)
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0
python -m insectpose.cli train experiment=exp_a_yolo_pooled folds=[0,1,2]

# 4. Optuna optimisation (primary metric of configs/eval/default.yaml), then one
#    training per outer fold: every fold, or those of `folds`
python -m insectpose.cli tune experiment=exp_a_yolo_pooled
python -m insectpose.cli tune experiment=exp_a_yolo_pooled folds=[0,1]

# 5. aggregation of every run + tables
python -m insectpose.cli report
```

The repository root also runs this chain, among the other steps of the project:
`run_all.py` with `run_config.yaml` (see the root README).

## Retained models: an ensemble, used by `pipeline/`

`<repo>/retained_models/pose/` holds an **ensemble** of `yolo_pooled` models
(`retain.approaches`), one sub-folder per model with its `model_card.json` (approach, fold,
primary metric, keypoint schema, commit). The pipeline runs ALL of them on every image and
writes, for every keypoint coordinate, the mean and the standard deviation over the ensemble;
measurements and validity classifiers work on the mean. Every command REPLACES the previous
ensemble, so that two trainings never mix:

```
train                  -> 1 model + ensemble.json    (zero standard deviation in the pipeline)
train folds=[0,1,2]    -> 3 models + ensemble.json   (one per fold, ADR-0039)
tune     ├─ HPO trials (inner folds)           -> not exported
         └─ reruns on the outer folds          -> EXPORTED: 5 models (or those of `folds`) + ensemble.json
evaluate run_id=<run_id> -> 1 model (retain an already-trained run, without retraining)
```

```bash
# export nothing (exploration): the ensemble in place stays intact
python -m insectpose.cli train experiment=exp_a_yolo_pooled retain.enabled=false
```

After `train` and `tune`, `ensemble.json` lists the members and carries `cv_estimate`: the
mean and the standard deviation of the primary metric over the folds trained, each one
measured on a test it has not seen. Running `tune` again afterwards skips the folds already complete and rebuilds
the ensemble. `tuning.final_full_fit=true` ALSO trains a model on all the images (retained
hyperparameters; in `nested` mode, those of the best inner value) and ADDS it to the
ensemble. A fixed `retain.name` would make several runs write to the same place (only the
last one would survive, with a warning): leave it at `null`.

A `mode=smoke` run, an HPO trial or an approach outside `retain.approaches` is never exported
and does not touch the ensemble in place. `retain.*` enters neither the `run_id` nor the
`variant_hash`: changing these keys retrains nothing and does not split the report tables.
`paths.retained=<path>` moves the destination if the module root is not
`modules/architectures`.

## Keypoints, measurements and results

The `insect42_v1` schema (points, difficulties, symmetries, skeleton) and the 27 measurements
are read from `<repo>/kp_infos.yaml`, the single definition of the repository. The analyses
(`report`, `scripts/compare_models.py`, `plot_optuna.py`) are written to `<repo>/results/pose/`
(see `results/README.md`).

## Adding an approach

Copy `src/insectpose/approaches/TEMPLATE.py.txt` and follow `CONVENTIONS.md` §11
(6 artefacts). The smoke test is parametrised on the registry: a registered approach enters it
automatically, without changing `tests/`.

## Status

Complete generic framework (contracts, registry, splits, evaluation, tuning, reporting, CLI).

| Approach                | Status                                                                  |
| ----------------------- | ----------------------------------------------------------------------- |
| `mean_pose`           | implemented - reference and floor baseline (GT bbox, diagnostic)        |
| `yolo_pooled`         | **implemented** - CUDA GPU, AMP, FP16 at inference; the retained model |
| `yolo_per_dataset`    | **implemented** - N models routed by dataset (ADR-0023)                |
| `detect_then_pose`    | **implemented** - pooled detector + pose on a crop (ADR-0024)          |
| `lora`                | **implemented** - adapters on the neck, trainable heads (ADR-0025)     |
| `group_bn`            | **implemented** - BatchNorm per dataset, mixed batches (ADR-0026)      |
| `yolo_pooled_reduced` | **implemented** - A without legs nor hind wings (ADR-0027)             |

An approach whose heavy dependency is missing is **skipped** by the smoke test
(`availability()` mechanism), never failed.

Every run produces: `manifest.json`, resolved `config.yaml`, `predictions/`,
`metrics.parquet`, `logs/` and `figures/` (12 pred vs GT examples including the 6 worst cases).

### What `mean_pose` is for

It is not a candidate model: for each instance it predicts the **mean pose of the train set**
placed back into the ground-truth bbox. It plays three roles:

1. **floor baseline** - a trained model that does not clearly beat it has a problem
   (convergence, malformed labels, wrong keypoint order). It is a sanity test, not a
   competitor;
2. **template** - it is the reference implementation of the `Approach` protocol, to copy when
   writing a new approach;
3. **smoke test** - it runs in a few seconds, without a GPU nor a heavy dependency, which
   validates the whole `train -> predict -> evaluate -> figures` chain at every
   `pytest -m smoke`.

It uses the GT bboxes (`bbox_source: gt`): its numbers are therefore **not comparable** with
those of the end-to-end approaches and must never appear in the same table (§9.3).

## Frozen protocol

Every protocol decision is settled (ADR-0006 to 0039, see `DECISIONS.md`):

| Point          | Value                                                                  |
| -------------- | ---------------------------------------------------------------------- |
| Keypoints      | `insect42_v1`: 42 points, common to the 4 datasets, union = identity |
| OKS sigmas     | `difficulty x 0.0025` (10 -> 0.025 ... 40 -> 0.100)                  |
| PCK            | `alpha x thorax width`, reference alpha = 0.25                        |
| Primary metric | `oks_ap` (overridable without re-evaluation)                         |
| Measurements   | 27 morphometric measurements + 9 symmetry pairs                        |
| Folds          | 5 outer folds, group_id = image_id (one image = one specimen)          |
| HPO            | `tune_once`: search on the inner folds of outer fold 0, 20 trials (ADR-0031) |
| Resolution     | 640x640 for every approach (strict guard)                              |

**Cost of the fully nested HPO** (`tuning.mode=nested`): `n_folds x n_trials x inner_folds`
trainings per approach. To be calibrated before launching a heavy approach, then frozen
identically for all of them (fair budget).
