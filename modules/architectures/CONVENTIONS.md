# CONVENTIONS.md — Architecture and code-generation rules

**Project:** pose estimation on 4 insect datasets (Coleoptera, Diptera, Hymenoptera, Lepidoptera)
**Status:** normative contract. This file takes precedence over any other document of the repository.
**Contract version:** 2.2 — protocol decisions settled (ADR-0006 to 0039); keypoint schema and measurements in `kp_infos.yaml`, analyses in `<repo>/results/pose/`; everything written in English (§8.3)

---

## 0. How to use this file

This document is meant to be given **in full and in context** to any generative AI (or human contributor) asked to write code in this repository.

Rule zero: **every code generation must cite, in the header comment of the file it produces, the sections of this document it applies.** If a user instruction contradicts this file, the AI must stop and report the conflict instead of deciding alone.

Normative vocabulary: **MUST** / **MUST NOT** = hard, non-negotiable constraint. **SHOULD** = strong recommendation, a deviation is possible if documented in `DECISIONS.md`. **MAY** = free.

---

## 1. Guiding principles

1. **An approach is a plugin, not an `if` branch.** Adding a 6th approach MUST NOT modify the generic training, evaluation, optimisation or reporting code. If you have to modify `evaluation/` to add an approach, the abstraction is wrong: report it.
2. **The data contracts are the API of the project.** Approaches never talk to each other or to the evaluator through Python objects: they communicate through **files with a frozen format** (§3). This makes it possible to train with Ultralytics, plain PyTorch, HuggingFace/PEFT or an external model without the evaluator ever knowing.
3. **A single implementation of the metrics.** No metric MUST be read from the logs of a third-party framework. The internal metrics of Ultralytics, PyTorch Lightning or any other trainer are **for monitoring only**, never for comparing approaches (§7.1).
4. **Strict separation: `fit` ≠ `predict` ≠ `evaluate` ≠ `aggregate`.** Four steps, four artefacts, four resume points. It MUST be possible to re-evaluate a three-month-old experiment without retraining.
5. **Everything is configuration; nothing is hard-coded.** No path, hyperparameter, threshold, image size, class or keypoint name MUST appear literally in a `.py`. Everything comes from a YAML file or from the `RunContext`.
6. **Reproducibility by construction.** Deterministic `run_id`, explicit seeds, resolved config serialised in the run folder, pinned dependency versions (§6.4).
7. **Cost of ignorance.** Every approach MUST run in `smoke` mode (2 epochs, 8 images, 1 fold) to validate the end-to-end wiring in < 2 minutes, before any real training.

---

## 2. Repository layout

```
insectpose/
├── CONVENTIONS.md              # this file — authoritative
├── DECISIONS.md                # log of methodological choices (ADR, append-only)
├── README.md                   # quick start only, no doctrine
├── pyproject.toml              # pinned dependencies, ruff/mypy/pytest config
├── Makefile                    # shortcuts: make smoke / make tune / make eval / make report
│
├── configs/                    # Hydra composition — the ONLY source of parameters
│   ├── config.yaml             # root config + defaults list
│   ├── paths.yaml              # path roots (overridden per machine)
│   ├── data/                   # coleoptera.yaml diptera.yaml ... pooled.yaml
│   ├── keypoints/              # study schemas; the project's one is <repo>/kp_infos.yaml (§3.1)
│   ├── approach/               # yolo_pooled.yaml yolo_per_dataset.yaml
│   │                           # detect_then_pose.yaml lora.yaml group_bn.yaml
│   ├── cv/                     # kfold5.yaml kfold5_grouped.yaml holdout.yaml
│   ├── eval/                   # default.yaml (metrics, thresholds, OKS sigmas)
│   ├── tuning/                 # optuna_default.yaml + per-approach budgets
│   └── experiment/             # named, frozen compositions (§5.3)
│
├── data/
│   ├── raw/                    # IMMUTABLE, never written by the code, never committed
│   ├── interim/                # adapter outputs, rebuildable
│   ├── processed/              # canonical format (§3.2), rebuildable
│   └── splits/                 # versioned and hashed fold assignments (§3.3)
│
├── src/insectpose/
│   ├── contracts.py            # dataclasses/TypedDict of the 5 contracts — UNTOUCHABLE without a bump
│   ├── registry.py             # registry by name (approaches, metrics, adapters)
│   ├── paths.py                # the only place that builds paths
│   ├── context.py              # RunContext (run_id, seed, fold, folders, logger)
│   │
│   ├── data/
│   │   ├── schema.py           # validation of the canonical format
│   │   ├── adapters/           # raw -> canonical, one module per source
│   │   ├── keypoints.py        # per-dataset <-> union space mapping
│   │   ├── datamodule.py       # canonical -> batches (superset of fields, §4.3)
│   │   └── splits.py           # fold generation and reading
│   │
│   ├── approaches/
│   │   ├── base.py             # Approach protocol + BaseApproach
│   │   ├── yolo_pooled.py
│   │   ├── yolo_per_dataset.py
│   │   ├── detect_then_pose.py
│   │   ├── lora.py
│   │   └── group_bn.py
│   │
│   ├── models/                 # reusable building blocks (backbones, heads, LoRA adapters, GroupBN)
│   ├── training/               # generic loops, callbacks, early stopping
│   ├── evaluation/
│   │   ├── metrics/            # one metric = one registered module
│   │   ├── matching.py         # pred<->gt matching (OKS/IoU), shared
│   │   ├── evaluator.py        # predictions.parquet -> metrics.parquet
│   │   └── aggregate.py        # every run -> results/master.parquet
│   ├── tuning/
│   │   ├── search_spaces.py    # Optuna spaces, one per approach
│   │   └── objective.py        # generic objective (§6.3)
│   ├── reporting/              # tables, figures, statistical tests
│   ├── cli.py                  # entry points (§5.4)
│   └── utils/                  # seed, io, hashing, geometry, logging
│
├── runs/                       # run artefacts, not committed (§8)
└── tests/                      # unit + contract + smoke (§10)
```

The analyses (`paths.results`: aggregates, figures, reports) are not in the module: they go to `<repo>/results/pose/`, with those of the other modules (see `results/README.md`).

**Golden rule of the layout:** a `.py` file MUST NOT write outside `runs/<run_id>/`, `data/interim/`, `data/processed/`, `data/splits/`, `paths.results` (`<repo>/results/pose/`) and `paths.retained` (`<repo>/retained_models/pose/`, export of the retained models, ADR-0037). Any other write is a bug.

---

## 3. The five contracts

These are the five frozen formats that make the project modular. Each one carries a `schema_version` field. **Changing a contract MUST be done by a version increment + a backward-compatible reader**, never by an in-place change.

### 3.1 Contract 0 — Keypoint schema (`kp_infos.yaml`, repository root)

The four datasets share **a single 42-point schema** (ADR-0006). The union space is that schema itself: the mapping is the identity, and the union mechanism stays in place to absorb a future divergence without a redesign.

```yaml
schema_version: 1
name: insect42_v1
status: VALIDATED
union_space: insect42_v1
sigma_from_difficulty: {scale: 0.0025}   # sigma = difficulty * scale (ADR-0007)
keypoints:
  - {name: thorax-left,  difficulty: 30, flip: thorax-right}
  - {name: thorax-right, difficulty: 30, flip: thorax-left}
skeleton: [[0, 5], [0, 12], ...]         # 51 anatomical edges
```

Rules:

- **The order of the 42 points is frozen for life**: it is encoded in every artefact produced. Adding a point = appending it *at the end of the list* and bumping `schema_version`.
- OKS tolerances are **not** hard-coded: `sigma = difficulty × scale`, where `difficulty` (10 to 40) is the difficulty of placing the point precisely, given by the expert. A point that is hard to annotate is judged more leniently, which keeps the metric from being dominated by annotation noise. Changing `scale` changes the definition of the OKS: bump `eval.version` and replay the runs.
- `flip` defines the symmetry pairs; any mirror augmentation without this table is forbidden. Points of the midline are their own mirror.
- A schema marked `status: PLACEHOLDER` is refused when `strict.require_validated_keypoints` is true (the default).
- **Morphometric measurements** (`kp_infos.yaml`, repository root, ADR-0008): 27 measurements defined as keypoint polylines, plus 9 left/right pairs. They are the quantity actually consumed downstream, hence a first-class metric — not an appendix.

### 3.2 Contract 1 — Canonical annotations (`data/processed/<dataset>/annotations.parquet`)

One row = one **annotated instance**. A single format whatever the original source (COCO, CVAT, CSV…).

| column                            | type             | description                                                                    |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `schema_version`                | int              | 1                                                                              |
| `dataset`                       | str              | `coleoptera` \| `diptera` \| `hymenoptera` \| `lepidoptera`            |
| `image_id`                      | str              | **globally unique** identifier: `<dataset>/<file_name_without_ext>`    |
| `image_path`                    | str              | path **relative to `paths.data_root`**, never absolute                |
| `image_width`, `image_height` | int              | pixels, original image                                                         |
| `instance_id`                   | str              | `<image_id>#<n>`                                                             |
| `group_id`                      | str              | anti-leakage key: specimen, plate, capture session (§6.1)                    |
| `bbox_xywh`                     | list[float] (4)  | **original image** coordinates, absolute pixels                          |
| `kpts_xy`                       | list[float] (2K) | order of the local schema, absolute pixels, original image                    |
| `kpts_vis`                      | list[int] (K)    | 0 absent / 1 occluded / 2 visible                                             |
| `area`                          | float            | area of the segment or of the bbox                                             |
| `keypoint_schema`               | str              | name of the §3.1 schema                                                      |
| `split_source`                  | str              | `train` \| `official_test` \| `unknown` if an upstream split exists     |

Rules:

- **All coordinates, everywhere, in every file, are expressed in the frame of the original image, in absolute pixels.** No normalised format, no relative `xyxy`, no coordinate in a crop frame must ever leave a module.
- The adapters (`data/adapters/`) are the **only** modules allowed to know the source formats. An adapter only does: read → convert → validate (`schema.py`) → write. No filtering, no augmentation, no methodological decision.
- Invalid instances (keypoints outside the image, empty bbox) are **kept** with a `qc_flags` flag, not deleted; filtering is a config decision, not an adapter one.

### 3.3 Contract 2 — Splits (`data/splits/<split_id>.parquet` + `.json`)

| column       | type                              |
| ------------ | --------------------------------- |
| `split_id` | str, e.g.`kfold5_grouped_seed42` |
| `image_id` | str                               |
| `fold`     | int                               |
| `role`     | `train` \| `val` \| `test`  |

Rules:

- The folds are **generated once** and **shared by every approach**. An approach MUST NEVER create its own splits.
- The unit of the split is `group_id`, not `image_id` (§6.1).
- The companion `.json` holds: seed, strategy, stratification, counts per dataset/fold, and a `content_hash` of the annotations used. **If the hash of the annotations changes, the splits are invalidated** and the pipeline MUST refuse to run.

### 3.4 Contract 3 — Predictions (`runs/<run_id>/predictions/<split>_fold<k>.parquet`)

This is **the** contract that makes approaches interchangeable. One row = one predicted instance.

| column                                       | type             | description                                                                                                                                          |
| -------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                           | int              | 1                                                                                                                                                    |
| `run_id`, `fold`, `split`, `dataset` | str/int          |                                                                                                                                                      |
| `image_id`                                 | str              |                                                                                                                                                      |
| `pred_id`                                  | str              | unique                                                                                                                                               |
| `bbox_xywh`                                | list[float] (4)  | original image frame; mandatory even for a pose-only approach (then = GT bbox or bounding box of the kpts, and `bbox_source` says so)          |
| `bbox_score`                               | float            | 1.0 if not applicable                                                                                                                                |
| `kpts_xy`                                  | list[float] (2K) | **original image frame**, local schema of the dataset                                                                                         |
| `kpts_score`                               | list[float] (K)  |                                                                                                                                                      |
| `keypoint_schema`                          | str              | must match the schema of the image's dataset                                                                                                        |
| `bbox_source`                              | str              | `predicted` \| `gt` \| `derived`                                                                                                               |
| `inference_ms`                             | float            | time per instance, for the cost/performance comparison                                                                                              |

Rules:

- **No score threshold is applied when writing.** Every prediction above a very low threshold (e.g. 0.001) is written; thresholding is an evaluation operation, set in the config. Otherwise the P/R curves are truncated and the approaches become incomparable.
- Any approach working on a **crop** (detection→pose pipeline, §9.3) MUST keep the crop→image affine transform and **back-project** before writing. Writing coordinates in the crop frame is a blocking error.
- A model trained in the union space MUST project to the local schema before writing (§3.1).
- The `test` predictions of a fold MUST only contain the images of that fold.

### 3.5 Contract 4 — Metrics (`runs/<run_id>/metrics.parquet`) and 5 — Manifest (`runs/<run_id>/manifest.json`)

`metrics.parquet` — long format, never wide:

| column                                        | description                                            |
| --------------------------------------------- | ------------------------------------------------------ |
| `run_id`, `approach`, `fold`, `split` |                                                        |
| `scope`                                     | `overall` \| `dataset:<name>` \| `keypoint:<name>` |
| `metric`                                    | canonical name, e.g.`pck@0.05_bboxdiag`              |
| `value`                                     | float                                                  |
| `n`                                         | size of the underlying sample                          |

`manifest.json`: `run_id`, timestamp, `approach`, `split_id`, **resolved** Hydra config (not the CLI overrides), `content_hash` of the data, git commit + clean/dirty state of the repository, versions of the key dependencies, seeds, paths of the produced artefacts, durations, GPU resources, and `optuna_study`/`trial_number` when applicable. **A run without a complete manifest is excluded from the aggregation.**

---

## 4. Interfaces and registry

### 4.1 Registry

A single decorator, one namespace per family:

```python
@register_approach("lora")            # approaches
@register_metric("pck")               # metrics
@register_adapter("coleoptera_cvat")  # data adapters
```

The registered name MUST be identical to the name of the matching YAML config file. No conditional `import`, no `if approach == ...` anywhere but in the registry.

### 4.2 `Approach` protocol

Every approach MUST implement exactly this interface, no more and no less on the pipeline side:

```python
class Approach(Protocol):
    name: str

    def fit(self, data: FoldData, ctx: RunContext) -> None: ...
        # trains on data.train, validates on data.val; writes its weights to ctx.run_dir/weights/
        # MUST NOT touch data.test

    def predict(self, images: ImageSet, ctx: RunContext) -> Path: ...
        # returns the path of a predictions parquet that complies with Contract 3

    @classmethod
    def load(cls, run_dir: Path, cfg: DictConfig) -> "Approach": ...
        # rebuilds a predictive model from the artefacts, without retraining

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict: ...
        # config overrides proposed to Optuna; no training logic here
```

Rules:

- `fit` MUST NEVER access `data.test`. A unit test checks this property (§10).
- `predict` MUST NEVER compute a metric.
- An approach MAY rely on several sub-models (e.g. detection+pose): that is its internal business, invisible to the pipeline.
- A "per dataset" approach (§9.2) remains **a single** approach: it wraps N models and routes by `dataset`. The pipeline must not see the difference.

### 4.3 DataModule: superset of fields

The batch produced by the datamodule MUST always contain the **superset** of the fields useful to every approach, even if a given approach ignores them:

```
images, bboxes, keypoints, visibility, meta{image_id, instance_id, dataset,
dataset_index, group_id, orig_size, transform_matrix}
```

`dataset_index` is required by the per-group BatchNorm approach (§9.5); `transform_matrix` by the back-projection. Adding them one at a time breaks modularity: they are there from the start.

---

## 5. Configuration

### 5.1 Tool

Hydra + OmegaConf. Composition through `defaults`, CLI override through `key=value`. No hand-written `argparse`, no config dictionaries coded in Python.

### 5.2 Rules

- One YAML file per named entity; the file name is the identifier.
- Every key MUST have an explicit default value; `cfg.get("x", 3)` scattered through the code is forbidden.
- Approach configs contain **only** what is specific to the approach. Common parameters (image size, batch, epochs, evaluation thresholds) live in `config.yaml` and can be overridden.
- Hydra interpolations crossing more than one level (`${a.b.c.d}`, unreadable) are forbidden: prefer an explicit field.
- The **resolved** config is written to `runs/<run_id>/config.yaml` **before** any training.

### 5.3 Named experiments

Every run meant for the final report MUST go through a frozen, committed `configs/experiment/*.yaml` file (e.g. `exp_A_yolo_pooled_kfold5.yaml`). Ad hoc CLI overrides are reserved for exploration and MUST NOT produce results quoted in the report.

### 5.4 CLI

Five verbs, no more:

```
python -m insectpose.cli prepare   data=coleoptera
python -m insectpose.cli split     cv=kfold5_grouped
python -m insectpose.cli train     experiment=exp_A cv.fold=0
python -m insectpose.cli train     experiment=exp_A folds=[0,1,2]      # several folds, one ensemble (ADR-0039)
python -m insectpose.cli predict   run_id=<...> split=test
python -m insectpose.cli evaluate  run_id=<...>
python -m insectpose.cli tune      experiment=exp_A tuning=optuna_default
python -m insectpose.cli report
```

`train` MAY chain `predict` + `evaluate` for convenience, but each one MUST remain callable on its own.

---

## 6. Experimental protocol

### 6.1 Anti-leakage

- The split is done by `group_id`. If a specimen appears on several images, all its images are in the same fold. **If the `group_id` of a dataset is unknown, the default is `image_id`, and this limitation MUST be written in `DECISIONS.md`.**
- Stratification by `dataset` (and by number of instances per image if unbalanced) is mandatory for the pooled folds.
- No statistic (normalisation mean/std, anchor sizes, keypoint clustering) MUST be computed on anything but the `train` of the current fold.

### 6.2 Cross-validation

- Default scheme: **K=5 stratified grouped folds**, fixed seed, a single `split_id` shared by every approach. The same folds for everyone, otherwise no comparison is valid.
- The "per dataset" approaches (§9.2) use **the same folds**, simply restricted to their dataset. Never regenerate a local split.
- An approach is compared on the **mean ± standard deviation across folds**, and the per-fold results are kept for the paired tests (§8.4).

### 6.3 Optuna optimisation

- **Nested by default** (ADR-0012). For each outer fold, the search runs on **inner** folds built from the outer train only. These inner splits (`<split_id>__outer<k>`) are generated by `cli split` and versioned exactly like the outer folds. The best hyperparameters are then applied to the whole outer fold. **The outer test has never been used to choose a hyperparameter.** An automatic test checks this property.
- Degraded `tune_once` mode: the search only runs on the inner folds of one outer fold, and the result is reused for all the others. Acceptable if documented; the trial budget must then be identical across approaches.
- **Cost**: `n_folds × n_trials × inner_folds` trainings per approach. To be calibrated before launching a heavy approach; the actual budget is recorded in every manifest.
- The objective is **always the primary metric computed by the shared evaluator**, read from `metrics.parquet` — never a validation loss nor an internal framework metric.
- A trial = a complete run with its own `run_id` and manifest; trials can therefore be evaluated and audited like any run. They export no qualitative figures (useless noise).
- SQLite storage under `runs/optuna/`, one study per (approach, split, objective, outer fold), resume enabled.
- `MedianPruner` pruning by default; an approach that cannot report intermediate values declares `prunable: false`.
- **Fair budget**: comparing 100 trials against 10 invalidates the conclusion.

### 6.4 Determinism

- A single seed in the config, derived by `seed_for(run_id, fold, purpose)` for numpy / torch / python / dataloader workers.
- `torch.use_deterministic_algorithms(True)` in `debug` mode; in `full` mode cudnn benchmark is allowed but recorded in the manifest.
- The remaining non-determinism is absorbed by repetition: any final conclusion SHOULD rest on ≥ 2 seeds for the winning approach.

---

## 7. Evaluation

### 7.1 Absolute rule

The evaluator takes **only**: a `predictions.parquet` (Contract 3), the canonical annotations (Contract 1), and `configs/eval/*.yaml`. It loads no model, imports no approach module, and knows nothing about how the predictions were produced. **If the evaluator needs to know which approach fed it, the design is broken.**

### 7.2 Frozen set of metrics

Identical for every approach, computed `overall`, per `dataset:*`, per `keypoint:*` and per `measurement:*`:

- **Detection** (if `bbox_source == predicted`): `det_ap@0.5`, `det_ap@[.5:.95]`.
- **Pose**: `oks_ap`, `oks_ap@0.5`, `oks_ar` (sigmas derived from the difficulty, ADR-0007); `pck@{0.125, 0.25, 0.5}_thorax_width` — a point is correct if its error is below `alpha × thorax width` (ADR-0009), `alpha = 0.25` being the project's reference; `nme_matched_only`, `kpt_coverage`, PCK per keypoint.
- **Reference scale**: `pck_normalizer_fallback_rate`. When the thorax points are not annotated, the normalisation falls back to the bbox diagonal — and this fallback rate is **published**, never silent.
- **Morphometric measurements** (ADR-0008): `measurement_mape_median`, `measurement_mape_worst`, per-measurement detail, and `symmetry_gap_median` / `symmetry_gap_p90` — the left/right gap of the predicted measurements, computable **without ground truth**, hence usable as a quality check in production.
- **End to end**: the primary metric penalises detection failures. A pipeline that does not detect the insect does not have "0 keypoint evaluated", it has a counted failure.
- **Cost**: latency per instance and p95, number of parameters, VRAM, training time — first-order metrics, not appendices.

**Primary metric of the project**: `oks_ap` (ADR-0010). It is the **only freely overridable evaluation key** — it changes no computation, only the Optuna objective and the ranking of the approaches. Since every metric is computed at every run, changing the objective never requires a re-evaluation:

```
python -m insectpose.cli train ... eval.primary_metric=measurement_mape_median \
                                   eval.primary_direction=minimize
```

### 7.3 Matching

Prediction↔GT matching (by OKS or IoU, greedy by decreasing score) is implemented **once** in `evaluation/matching.py`. No metric reimplements its own matching.

### 7.4 Comparing approaches on a common scope

Approaches do not share the same natural scope (a per-dataset approach predicts nothing outside its dataset). Rule: **every comparison is made on the union of the test images of all folds**, a restricted approach being evaluated as the concatenation of its N models. A results table MUST show the underlying `n` of every cell (§3.5); two values with different `n` are not comparable and the report MUST say so.

---

## 8. Runs, artefacts and results

### 8.1 `run_id`

```
<approach>__<data_scope>__<split_id>__fold<k>__<tag>__<hash8>
e.g. lora__pooled__kfold5grouped_seed42__fold2__baseline__a3f91c07
```

`hash8` = first 8 characters of the hash of the resolved config + the `content_hash` of the data. Two identical runs have the same `run_id`: the pipeline MUST then skip the run (idempotence) unless `force=true`.

### 8.2 Content of a run folder

```
runs/<run_id>/
├── manifest.json          # Contract 5, written last -> its presence marks a complete run
├── config.yaml            # resolved config
├── weights/               # weights, checkpoints, LoRA adapters
├── predictions/           # Contract 3
├── metrics.parquet        # Contract 4
├── logs/                  # stdout, curves, tensorboard/mlflow
└── figures/               # qualitative visualisations (§8.5)
```

`manifest.json` is written **last**. A folder without a manifest = an interrupted run, ignored by the aggregation, deletable without discussion.

### 8.3 Language

**Everything is written in English**: the code, its comments and docstrings, log and error messages, the internal documentation (`CONVENTIONS.md`, `DECISIONS.md`, `README.md`), and everything written to a produced file — titles, axes, legends and annotations of figures, headers and text values of tables, fields of manifests and JSON reports, file names. The deliverables circulate outside the team and end up in publications.

Corollary: a metric, scope or column name is an identifier, never a sentence to translate. `oks_ap`, `dataset:coleoptera`, `measurement_mape_median` are frozen (§3.5).

### 8.4 Aggregation and reporting

- `aggregate.py` scans `runs/*/metrics.parquet` + manifests → `results/master.parquet`. **It is the only path to a results table.** No figure, no table of the report MUST be produced from a console copy-paste.
- Comparisons between approaches SHOULD use paired per-fold tests (signed Wilcoxon or paired t) with a correction for multiple comparisons, and report confidence intervals rather than raw ranks.
- `reporting/` produces: main table (approach × dataset × metric), PCK curves, cost vs performance scatter, per-keypoint error matrix, qualitative failures.

### 8.5 Mandatory qualitative output

Every run MUST export at least 12 annotated test images pred vs GT, in three categories answering three distinct questions: the **worst cases** (where the model fails), the **best case of each dataset** (what it can do at best — without it one cannot tell whether the failures are a ceiling or an accident), and a **random draw** (the only unbiased sample). A model is never validated on numbers alone.

---

## 9. Approach-specific constraints

These notes pin down the known pitfalls of each family. They add no interface: everything goes through §4.2.

### 9.1 Pooled YOLO (one "insect" class, all datasets) — IMPLEMENTED

The keypoint schema being common to the 4 orders (ADR-0006), the model predicts directly in the expected schema: no union → local reprojection is needed. Points absent from a dataset (ADR-0016) are written with `vis = 0` in the labels and masked in the loss, never learnt as zeros.

All the risky logic is isolated in `data/yolo_export.py`, tested by a round trip without a GPU:

- the YOLO bbox is **centred**, contract 1 uses the top-left corner;
- `flip_idx` is mandatory in `data.yaml` as soon as `fliplr > 0`, otherwise the mirror swaps left and right without permuting the labels;
- file names are flattened (`coleoptera__img000`), otherwise two datasets that both have an `img000.png` silently overwrite each other.

The YOLO files are a **derived** artefact, regenerated per fold under `runs/<run_id>/yolo_dataset/`, never written to `data/processed/`. `conf = 0.001` at inference: thresholding is an evaluation operation.

Hardware (ADR-0019): `train.device: auto` takes GPU 0 if CUDA is available, else the Apple GPU, else the CPU; AMP enabled by default but disabled on CPU and in `mode: debug`; FP16 at inference on GPU. Peak VRAM, training time and number of parameters go into the manifest — they are first-order cost metrics, and the aggregation warns if compared runs come from different hardware.

### 9.2 Per-dataset YOLO — IMPLEMENTED

A **single** `Approach` class wrapping N models, routed by `meta.dataset`. The pipeline does not see the difference: this is what guarantees that A and B are evaluated identically.

Three protocol choices (ADR-0023):

- each model starts again from the **base weights**, never from the pooled model — A and B remain independent, and the question asked is indeed "is a specialist worth a generalist?";
- the hyperparameters are **shared** by the N models, one Optuna trial training them all. The HPO budget thus stays strictly equal to A's. An independent search per dataset would quadruple it, and B would win through the optimisation rather than through the method;
- **same number of epochs** for every dataset. Consequence to keep in mind when reading the results: with 192 images for Hymenoptera against 935 for Coleoptera, the former sees five times fewer optimisation steps. A performance gap between orders is therefore not necessarily a difference of difficulty.

The folds are those of the shared split, simply restricted (§6.2). Each sub-model stores its artefacts under `weights/<dataset>/`, `yolo_dataset/<dataset>/`, `logs/<dataset>/`, and its costs are prefixed in the manifest.

### 9.3 Detection then pose on a crop — IMPLEMENTED

Two models in one run: a **pooled** detector (one class, whole images, labels without keypoints) then a YOLO-pose model trained on crops normalised to the protocol resolution (ADR-0024).

- The pose model is trained on crops taken from **noisy** GT bboxes (`jitter_scale`, `jitter_shift`), never on perfect framings: otherwise a train/test shift is guaranteed, since at inference the framings come from a detector. Validation, for its part, uses clean framings — a noisy validation metric would be useless.
- A margin (`crop.padding`) surrounds the bbox. Without it, tarsi and antennae fall outside the crop and become unrecoverable whatever the quality of the model. Points outside the frame are marked `vis = 0`: neither learnt as zeros nor counted as errors.
- The crop → image transform is kept and **every prediction is back-projected** to the frame of the original image before writing (contract 3).
- The end-to-end evaluation uses the **predicted** bboxes. The `pose_on_gt_boxes: true` mode writes `bbox_source: gt`: diagnostic only, never in the same table as the end-to-end approaches.

### 9.4 LoRA — IMPLEMENTED

Adapters injected on the convolutions of the last segment of the neck, backbone and neck frozen, trainable heads (ADR-0025). The block indices are **computed from the structure of the model**, never hard-coded: changing the network size (n/s/m/l) shifts everything.

The manifest records the **number of trainable parameters** and the list of adapted modules. Without it, "LoRA rank 8" means nothing: the same label covers very different configurations depending on what stays unfrozen next to the adapters.

### 9.5 Per-group BatchNorm — IMPLEMENTED

Every `BatchNorm2d` duplicated into N copies, statistics **and** affine parameters per dataset (ADR-0026). The convolution weights stay shared: that is the hypothesis being tested. Each branch is initialised from the statistics of the pre-trained model, never randomly.

Batches are **mixed**; the forward pass splits by group then recomposes in order. The group comes from the exported file name (`<dataset>__<stem>`) at training time, and from an explicit declaration at inference. An unknown dataset is an **explicit error** (ADR-0014), never a guessed fallback.

### 9.6 Reduced-keypoint variant — IMPLEMENTED

Approach A deprived of the supervision on the legs and hind wings (ADR-0027): these 16 points get `vis = 0` in the training and validation labels, **never in the test**.

Mandatory reading precaution: the ground truth keeps these points and the evaluation counts them. The `overall` metrics of this variant are therefore **mechanically worse** and do not compare with A's. The valid comparison is on the `keypoint:*` scopes of the kept points:

```
python scripts/compare_models.py --exclude-keypoints leg hindwing
```

which adds a `MEAN (retained)` row — the number to compare.

### 9.7 Patching the Ultralytics model

Approaches 9.4 and 9.5 modify the `nn.Module` built by Ultralytics, which foresees neither. All this dependency on the internals is isolated in `training/patching.py`, with two constraints checked on the installed version:

- **callbacks do not fit**: `on_pretrain_routine_start` comes before the model is built, `on_pretrain_routine_end` after the optimiser and the EMA. A derived trainer is therefore passed (`train(trainer=...)`) and the patch is applied in `get_model`;
- **Ultralytics unfreezes what we freeze**: its `freeze` loop sets `requires_grad=True` again on every frozen parameter outside `args.freeze`. The freeze is therefore re-applied in `_build_train_pipeline`, just before the optimiser is built.

A count of trainable parameters is logged and recorded in the manifest at every run. If a future Ultralytics version changes this order, this number reveals it immediately instead of letting a silently wrong training through.

### 9.8 Future approaches

Any new approach (multi-task, distillation, self-supervised pre-training, ensembles…) is added through §11 without exception. If an approach does not fit the `Approach` protocol, **the protocol is changed for everyone, with a version bump** — no special case is created.

---

## 10. Tests

Three levels, all mandatory before any long run:

1. **Contract tests** (`tests/contracts/`): check that a produced parquet follows the schema, the coordinate bounds, the uniqueness of identifiers, the keypoint_schema ↔ dimension consistency. Run automatically whenever an artefact is written in `debug` mode.
2. **Unit tests**: crop→image back-projection (round trip = identity to 1e-6), local↔union mapping, matching, every metric on a hand-computed case, no leakage (`fit` does not read `test`), reproducibility (two runs with the same seed = the same predictions).
3. **Smoke test** (`make smoke`): every registered approach is run on an 8-image, 1-fold fixture, from `train` to `report`. **An approach that does not pass the smoke test is not considered implemented.** The fixture is committed in `tests/fixtures/`.

CI: ruff + mypy (strict on `contracts.py`, `registry.py`, `evaluation/`) + pytest + smoke.

---

## 11. Procedure: adding an approach

Exactly 6 artefacts, no more, no less. If you have to touch a 7th existing file, it is a design signal to report.

1. `src/insectpose/approaches/<name>.py` — class decorated with `@register_approach("<name>")`, implementing §4.2.
2. `configs/approach/<name>.yaml` — default hyperparameters, `_target_` pointing to the class.
3. `search_space` in the class (or in `tuning/search_spaces.py` if large).
4. `tests/approaches/test_<name>.py` — smoke + specific tests (e.g. §9.5).
5. `configs/experiment/exp_<letter>_<name>.yaml` — frozen experiment for the report.
6. An entry in `DECISIONS.md`: what the approach tests, its hypotheses, its known limits.

---

## 12. Generation rules for AIs

To be followed by any AI producing code in this repository.

**Obligations**

- Declare heavy dependencies through `availability()`: the smoke test cleanly skips an unavailable approach instead of failing.
- Write everything in **English** (§8.3): code, comments, messages, documentation and produced files.
- Read this file and state, before writing, which contracts are touched.
- Write typed signatures; `contracts.py` is authoritative for data types.
- Every public function has a docstring giving: inputs, outputs, **file side effects** (exact path written).
- Validate inputs at module boundaries (parquet schema, presence of config keys) and fail early, loudly, with an actionable message.
- Produce, with any new module, its matching test. Code without a test = not delivered.
- Any non-trivial methodological decision taken along the way → a line in `DECISIONS.md`, not a comment buried in the code.

**Prohibitions**

- No hard-coded path, no magic constant, no literal threshold in a `.py`.
- No silent `try/except`, no `except Exception: pass`, no fallback value hiding missing data.
- No approach logic in `training/`, `evaluation/`, `tuning/`, `reporting/` — no `if approach == ...` anywhere.
- No metric computation outside `evaluation/metrics/`.
- No mutation of `data/raw/`. Ever.
- No new dependency without a justification and an addition to `pyproject.toml`.
- No notebook as a source of truth: a notebook calls the package, it contains no logic.
- No catch-all "utils.py" file: one module = one responsibility that can be named in one sentence.
- No opportunistic refactoring outside the requested scope.

**When to stop and ask**
An AI MUST interrupt the generation and ask the question if: a contract would have to change; two approaches would require an incompatible field; a metric is ambiguous; the keypoint schema of a dataset is unknown; a decision would affect the comparability between approaches. Inventing a convention to carry on is the most expensive mistake of the project.

**Recommended task prompt template**

```
Context: CONVENTIONS.md v2.2 (given in full).
Task: implement <X>.
Scope: files allowed to be created/modified = [...]. Everything else is read-only.
Contracts touched: [none | no. ...].
Deliverables: code + tests + DECISIONS.md entry if a decision is taken.
Acceptance criterion: `make smoke` passes for approach <X>.
If a rule of CONVENTIONS.md blocks you: stop and explain.
```

---

## 13. Protocol decisions (all settled)

Recorded in `DECISIONS.md`. They are **closed**: changing them invalidates the results already produced.

| #                  | Decision                 | Value retained                                                                                    |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------------------------- |
| ADR-0006           | Keypoint schema          | `insect42_v1`, 42 points, common to the 4 datasets                                              |
| ADR-0007           | OKS sigmas               | `sigma = difficulty × 0.0025`                                                                  |
| ADR-0008           | Morphometric measurements | 27 measurements + 9 symmetric pairs                                                              |
| ADR-0009           | PCK normalisation        | `alpha × thorax width`, reference 0.25                                                        |
| ADR-0010           | Primary metric           | `oks_ap`, overridable without re-evaluation                                                    |
| ADR-0011           | Anti-leakage grouping    | one image = one specimen                                                                          |
| ADR-0013           | Input resolution         | 640×640 for every approach                                                                       |
| ADR-0014           | Dataset at inference     | always declared; unknown = error                                                                 |
| ADR-0016           | Absent keypoints         | masked, never imputed                                                                             |
| ADR-0017           | Instances per image      | a single one; violation is blocking                                                              |
| **ADR-0031** | **HPO budget**      | `tune_once`, 20 trials, 5 startup, 3 inner folds, 100 epochs, **4 hyperparameters**   |
| **ADR-0032** | **Augmentation**    | fixed, never searched                                                                            |
| **ADR-0033** | **Base model**      | `yolo26n` for the six approaches                                                                 |
| **ADR-0037** | **Retained model**  | an ensemble of `yolo_pooled` models: 1 after `train`, 1 per outer fold after `tune`             |

**HPO budget — the overriding rule.** The six approaches share exactly the same budget and the same number of dimensions. Comparing approaches optimised with different budgets would measure the budget, not the method. The Optuna study name includes a hash of the search space and of the settings: any change automatically creates a new study, which makes it impossible to mix two protocols in the same database.

The four hyperparameters per approach:

| Approaches | Hyperparameters                                                                        |
| ---------- | --------------------------------------------------------------------------------------- |
| A, B, E, F | `lr0`, `pose`, `kobj`, `weight_decay`                                           |
| C          | `pose.lr0`, `pose.pose`, `crop.padding`, `crop.jitter_scale`                    |
| D          | `lr0`, `lora.r`, `lora.neck_blocks`, `pose` (with `alpha = 2r`, not searched) |

No open decision remains.

---

*End of the contract. Any change goes through a version increment of this file and an entry in `DECISIONS.md`*
