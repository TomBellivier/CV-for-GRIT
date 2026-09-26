
# DECISIONS.md - log of methodological choices

Append-only. One entry = one decision. Lightweight ADR format.
Any non-trivial decision taken while writing code lands here (CONVENTIONS.md §12).

---

## ADR-0001 - File contracts rather than a Python API between modules

**Date**: initialisation - **Status**: accepted
**Context**: 5+ approaches on heterogeneous frameworks (Ultralytics, PyTorch, PEFT).
**Decision**: the approaches communicate with the pipeline through files with a frozen schema
(annotations / splits / predictions / metrics / manifest). The evaluator loads no model.
**Consequences**: re-evaluation without retraining; I/O overhead accepted; any external
approach (third-party model, manual prediction) can be integrated without code.

## ADR-0002 - All coordinates in the frame of the original image

**Date**: initialisation - **Status**: accepted
**Context**: the detection->pose pipeline works on crops; YOLO works in normalised coordinates.
**Decision**: no normalised format nor crop frame leaves a module. Conversion and
back-projection happen at the boundary, checked by a round-trip test (1e-6).
**Consequences**: a single, comparable evaluation for every approach.

## ADR-0003 - A single evaluator, metrics never read from a framework

**Date**: initialisation - **Status**: accepted
**Decision**: the metrics of Ultralytics and the like are used for monitoring only.
Only `insectpose.evaluation` produces quotable numbers.

## ADR-0004 - Mandatory qualitative export at every run

**Date**: initialisation - **Status**: accepted
**Context**: a metrics table does not show *how* a model fails.
**Decision**: every run exports 12 pred vs GT figures including the 6 worst cases by
per-instance OKS, plus a traceable JSON index. A missing image is a BLOCKING error
(`eval.qualitative.allow_missing_images=false`): an empty export would hide a broken path.
**Consequences**: `image_path` must stay valid and relative to `paths.data`.

## ADR-0005 - Point metrics aggregated through counters, not by stacking

**Date**: initialisation - **Status**: accepted
**Context**: the 4 orders do not have the same number of keypoints; in the pooled scope,
stacking (N, K) arrays of different schemas is impossible.
**Decision**: PCK / NME aggregate numerators and denominators per homogeneous schema block.
**Consequences**: the `overall` and `dataset:*` scopes remain valid with several schemas;
a direct consequence of OPEN-01 and one more reason to settle it early.

## ADR-0006 - A single keypoint schema for the 4 datasets: insect42_v1

**Date**: expert specification - **Status**: accepted (closes OPEN-01)
**Context**: the main risk of the project was an anatomical divergence between orders.
**Decision**: the 4 datasets share a single 42-point schema (`kp_infos.yaml`, repository root), with a skeleton of 51 edges and a complete left/right symmetry table.
The union space is that schema itself: the mapping is the identity.
**Consequences**: the pooled model has no point to mask, the pooled vs per-dataset
comparisons are only about the method. The union-space mechanism stays in place and tested,
to absorb a future divergence without a redesign. **The order of the 42 points is frozen
for life**: it is encoded in every artefact produced.

## ADR-0007 - OKS sigmas derived from the placement difficulty

**Date**: expert specification - **Status**: accepted (closes OPEN-02, OKS part)
**Decision**: `sigma = difficulty * 0.0025`, where `difficulty` (10 to 40) is given by the
expert per point. Correspondence: 10 -> 0.025, 20 -> 0.050, 30 -> 0.075, 40 -> 0.100,
i.e. the range of the COCO sigmas. The rule is declared in the schema, not in the code.
**Consequences**: a point that is hard to annotate is judged more leniently, which keeps the
metric from being dominated by annotation noise. Changing `scale` changes the definition of
the OKS: bump `eval.version` and replay the runs.

## ADR-0008 - Error on the morphometric measurements as a first-class metric

**Date**: expert specification - **Status**: accepted
**Context**: the project downstream consumes 27 measurements (lengths, widths), not keypoints.
**Decision**: `kp_infos.yaml` (repository root) defines the 27 measurements and the 9
symmetric pairs. Two metrics: `measurement_mape_median` (relative error pred vs GT, per
measurement and overall) and `symmetry_gap_median/p90` (left/right gap of the PREDICTED
measurements).
**Consequences**: `symmetry_gap` needs no ground truth: it is a quality check usable in
production on unannotated images. A measurement is only evaluated if all its points are
annotated as visible.

## ADR-0009 - PCK normalised by the thorax width

**Date**: expert specification - **Status**: accepted (closes OPEN-02, PCK part)
**Decision**: a point is correct if its error is below `alpha x thorax width`, the width
being the `thorax-left` <-> `thorax-right` distance. Reference value of the project:
**alpha = 0.25**; alphas 0.125 and 0.5 give the curve.
**Consequences**: anatomical normalisation, insensitive to the framing of the bbox (unlike the
diagonal). If the two thorax points are not annotated, fallback to the bbox diagonal, and the
**fallback rate is published** (`pck_normalizer_fallback_rate`): a silently replaced
reference scale would bias the comparison.

## ADR-0010 - Primary metric: oks_ap, changeable without invalidating the runs

**Date**: expert specification - **Status**: accepted (closes OPEN-03)
**Decision**: `eval.primary_metric = oks_ap`. It is **the only freely overridable evaluation
key**: it changes no computation, only the Optuna objective and the ranking.

```
python -m insectpose.cli train ... eval.primary_metric=measurement_mape_median \
                                   eval.primary_direction=minimize
```

**Consequences**: every metric is computed at every run, so changing the objective never
requires a re-evaluation. The manifest records the objective used; the aggregation warns if
runs tuned on different objectives are compared.

## ADR-0011 - One image = one specimen: group_id = image_id

**Date**: expert specification - **Status**: accepted (closes OPEN-04)
**Decision**: no dataset contains several images of the same specimen or of the same plate.
The group split stays active with `group_id = image_id`.
**Consequences**: no leakage. If a future dataset brings several views per specimen, filling
`data.adapter_options.group_id_field` is enough: no code to change.

## ADR-0012 - Nested HPO

**Date**: expert specification - **Status**: accepted (closes OPEN-05)
**Decision**: for each outer fold, the hyperparameter search runs on **inner** folds built from
the outer train only (`<split_id>__outer<k>`, generated by `cli split` and versioned like the
outer folds). The best hyperparameters are then applied to the whole outer fold; the outer
test has never been used to choose a hyperparameter.
**Consequences**: cost = `n_folds x n_trials x inner_folds` trainings, i.e. 5 x 40 x 3 = 600
runs per approach with the default values. **To be calibrated before launching a heavy
approach**: reduce `n_trials`, `inner_folds`, or switch to `tune_once` (documented as such). The
actual budget is recorded in every manifest, and it must stay identical across approaches.

## ADR-0013 - Common input resolution 640x640

**Date**: expert specification - **Status**: accepted (closes OPEN-06)
**Decision**: `protocol.image_size = [640, 640]` for every approach. The
`strict.enforce_common_image_size` guard refuses any divergence.
**Consequences**: the comparison is about the method, not the resolution. An exploration at
another resolution remains possible by disabling the guard, but its results cannot be quoted
in the report.

## ADR-0014 - The insect group is always known at inference

**Date**: expert specification - **Status**: accepted (closes OPEN-07)
**Decision**: the end user declares the insect order of the processed images. The
dataset-conditioned approaches (per-group BatchNorm, per-dataset models) can therefore rely on
`meta.dataset` without a fallback strategy.
**Consequences**: a missing or unknown dataset at inference is an **explicit error**, not a
case to guess. To be implemented as such in the `group_bn` approach.

## ADR-0015 - No additional experiment tracking

**Date**: expert specification - **Status**: accepted (closes OPEN-08)
**Decision**: no MLflow nor W&B. The JSON manifests and `results/master.parquet` are the
only source of truth.

## ADR-0016 - Keypoints absent depending on the insect order: masked, never imputed

**Date**: expert specification - **Status**: accepted (closes OPEN-10)
**Context**: the `insect42_v1` schema is common to the 4 orders, but some points do not exist
in all of them (wings, antennae depending on the groups). They carry `vis = 0`.
**Decision**: these points are **masked** everywhere, never replaced by a value: excluded from
the OKS and the PCK, masked in the loss (YOLO label `0 0 0`), and the measurements depending on
them are declared not computable for that dataset. `cli prepare` produces a coverage report
(`data/processed/coverage_*.parquet` + `coverage_summary.json`) that distinguishes three cases:
absent (<= 1 % annotated), rare (< 50 %, per-point PCK not very informative) and present.
**Consequences**: a point absent from ALL the datasets is reported as a warning — the model
would predict it without any supervision, and removing it from the schema should be
considered. `keypoint:<dataset>:<point>` scopes missing from the results are not a bug: they
signal an absence of annotation, and the `n` of each row lets one check it.

## ADR-0017 - One image = one insect

**Date**: expert specification - **Status**: accepted
**Decision**: `data.single_instance_per_image = true`. A violation is a **blocking error at
data preparation**, not a warning.
**Consequences**: the detection approaches use `max_det = 1`. Without this guard, a
multi-instance image would silently make the top-1 detection lie. Detection is still
evaluated (`det_ap@0.5`): framing a single insect is not trivial for all that.

## ADR-0018 - yolo_pooled: first approach implemented

**Date**: implementation - **Status**: accepted
**Decision**: Approach A = a single YOLO-pose model, one `insect` class, the 4 datasets
pooled, 42 keypoints. Since the schema is common (ADR-0006), no union -> local reprojection is
needed. The risky logic (coordinate conversion, label format, symmetry table) is isolated in
`data/yolo_export.py` and tested by a round trip, without a GPU.
**Consequences**:

- `data.yaml` carries `flip_idx`: without it, `fliplr` would learn a mirrored anatomy.
  The approach refuses `fliplr > 0` if the schema has no symmetry pair.
- The YOLO files are a **derived** artefact, regenerated per fold under `runs/<run_id>/`,
  never written to `data/processed/`.
- File names are flattened (`coleoptera__img000`): otherwise two datasets both having an
  `img000.png` would silently overwrite each other.
- `conf = 0.001` at inference: thresholding is an evaluation operation, not a writing one.
- Since Ultralytics reports no usable intermediate value, `prunable = false`: Optuna pruning
  happens at the fold level, not the epoch level.
- Heavy dependency declared through `availability()`: the smoke test **cleanly skips** the
  approach if `ultralytics` is missing, instead of failing.

## ADR-0019 - Hardware: CUDA available, torch and ultralytics as first-rank dependencies

**Date**: environment specification - **Status**: accepted
**Context**: the target environment has ultralytics and a CUDA GPU.
**Decision**:

- `torch`, `torchvision` and `ultralytics` move to `dependencies` (the `[yolo]` extra is kept
  empty, for compatibility);
- `train.device: auto` resolves to GPU 0 if CUDA is available, else the Apple GPU (`mps`), else
  `cpu`; an explicit value (`cpu`, `"0,1"`, `mps`) is always respected. `train.num_workers:
  auto` takes one data-loading worker per CPU minus one, capped at 8;
- mixed precision (`train.amp: true`) enabled by default, **automatically disabled in
  `mode: debug`** where reproducibility prevails over speed, and on the CPU;
- inference precision declared explicitly (`approach.inference_precision: fp16 | fp32`),
  translated into `quantize` (Ultralytics >= 8.4) or `half` (earlier versions); in fp32 no
  argument is passed, which avoids a deprecation warning;
- **streamed inference is mandatory** (`stream=True`): without it Ultralytics keeps one
  `Results` object per image, original image included. On full-resolution specimen photos, a
  few hundred images are enough to saturate the RAM and the process gets killed by the OOM
  killer after training. The test double now refuses a non-streamed call;
- peak VRAM (`peak_vram_mb`), training time and number of parameters are recorded in the
  manifest, as first-order cost metrics.
  **Consequences**: the resolved hardware (GPU name, capability, total VRAM, CUDA/cuDNN
  versions, number of CPUs) goes into `manifest.environment.device`, and the aggregation
  **warns if compared runs come from different hardware**: the costs (latency, VRAM) would then
  not be comparable, even if the OKS is.
  The `availability()` mechanism stays in place: a CI without a GPU cleanly skips the approach
  instead of failing. The Ultralytics integration is checked by a double (`tests/approaches/test_yolo_pooled_integration.py`) that tests what we control — centred bbox to
  top-left corner conversion, protocol arguments, weight copy — without requiring a GPU.

## ADR-0020 - Resolution cap of the source images at export

**Date**: field diagnosis - **Status**: CANCELLED (replaced by ADR-0021)
**Cancellation**: this cap had been introduced on a wrong diagnosis (see below). The real cause
of the memory saturation was the NUMBER of images passed in one call, not their resolution
(ADR-0021). The resizing was therefore removed: it added a coordinate transform to maintain and
test for a benefit that was not the one sought. Images are exported as they are, through a
symbolic link, and the predictions natively stay in the frame of the original image.
If decoding one day becomes a measured bottleneck, this lead remains valid and the
implementation is in the git history.

**Original content, kept for the record** - **Initial status**: accepted
**Context**: the real datasets contain photos of 10 to 50 MP (median 36 MP for the
Lepidoptera). A decoded 36 MP image takes ~108 MB in uint8, x4 in float for the augmentation,
x4 in mosaic: training saturated 31 GB of RAM and the process was killed by the OOM killer,
before even the first useful epoch.
**Decision**: `protocol.export_max_side: 1280` caps the long side of the images at the YOLO
export (0 = no cap). Consequences used:

- the YOLO labels are NORMALISED, hence **invariant to a uniform resizing**: no label to
  recompute, no extra conversion to test;
- the scale factor is kept per image in `scales.json`, and the predictions are
  **back-projected to the original resolution** before writing (contract 3);
- inference also runs on the reduced copies: decoding 36 MP at prediction time would cost the
  same RAM as at training time;
- JPEG decoding uses PIL's `draft` mode, which decodes directly at the reduced size.
  **Consequences**: `export_max_side` is a PROTOCOL parameter, recorded in the manifest and
  **identical for every approach** — two approaches trained on different source resolutions
  would not be comparable. 1280 px for a model working at 640 leaves a comfortable margin; to be
  revisited if the fine keypoints (tarsi, antennae) degrade, documenting it here.
  In addition, the Ultralytics trainer (dataloaders, workers, augmentation buffers) is
  explicitly released between `fit` and `predict`: without it, inference started with several
  GB already taken.

## ADR-0021 - Inference split into chunks

**Date**: field diagnosis - **Status**: accepted
**Context**: `predict(source=<whole fold list>)` made the RAM grow by ~1 GB every 5 seconds
until the OOM, whereas the same model trained group by group went through without a problem.
`tracemalloc` pointed at the Ultralytics loader (`data/loaders.py`, `self.im0 = [...]`,
`bs = len(im0)`): **every image of the `source` is materialised when the loader is built,
before any inference**. `stream=True` changes nothing, the accumulation happening upstream.
**Decision**: inference goes through the images in chunks of `approach.predict_chunk_size`
(16 by default), with an explicit release between two chunks. The conversion of the `Results`
is isolated in `_rows_from_results` so that none of them outlives its chunk.
**Consequences**: the memory footprint of inference becomes independent of the fold size. This
parameter affects **no result**, only memory: it can be tuned freely, unlike the protocol
parameters.
**Honesty note**: the initial diagnosis attributed the saturation to the resolution of the
images (ADR-0020). It was wrong — the user was already training on these same images without a
problem, group by group. The discriminating factor was the NUMBER of images per call.
ADR-0020 remains useful (faster decoding and I/O) but was not the cause.

## ADR-0023 - Approach B: one YOLO-pose model per dataset

**Date**: expert specification - **Status**: accepted
**Decisions**:

- **Initialisation**: each model starts again from the base weights (COCO), not from the
  pooled model. A and B remain independent; the question asked is "is a specialist worth a
  generalist?", not "does specialisation bring anything after pooling?".
- **HPO budget**: total equal to A's. Consequence retained: the hyperparameters are **shared**
  by the N models, one trial training them all. An independent search per dataset would have
  quadrupled the budget, and B would have won through the optimisation rather than through the
  method (§6.3).
- **Epochs**: identical for every dataset, whatever its size.
  **Consequences**: with 192 images (Hymenoptera) against 935 (Coleoptera), the former sees five
  times fewer optimisation steps. **A performance gap between orders is therefore not
  necessarily a difference of difficulty**: this limit must be recalled in the report.
  If it becomes a problem, the "equalised optimisation steps" variant will be a new decision,
  not a setting. Technically, B is ONE approach wrapping N models: the pipeline does not see
  the difference, so A and B are evaluated exactly the same way.

## ADR-0024 - Approach C: pooled detection then pose on a crop

**Date**: expert specification - **Status**: accepted
**Decisions**:

- **Detector**: single and pooled (one "insect" class), trained INSIDE the run. Reusing the
  weights of an A run would have made C dependent on A and complicated the fold handling.
- **Pose model**: YOLO-pose on crops. A top-down heatmap model (HRNet, ViTPose) remains
  possible as a 6th approach if the gain of C over A is clear.
- **Crop resolution**: 640x640, identical to the protocol (ADR-0013). A lower resolution would
  have divided the cost, but the comparison with A would then partly have been about the
  resolution. A 256 variant remains possible as a cost study, outside the main table.
- **Framing noise**: `jitter_scale=0.15`, `jitter_shift=0.10` at training time, none in
  validation nor at inference. **Crop margin**: `padding=0.15`.
  **Consequences**: C trains two models per fold, so its cost exceeds A's and B's at an equal
  trial budget - to be mentioned in the cost/performance comparison. Points falling outside the
  crop are masked (`vis = 0`), never learnt as zeros. The `pose_on_gt_boxes` mode isolates the
  quality of the pose from that of the detection, but its results carry `bbox_source=gt` and
  never appear in the end-to-end table.

## ADR-0025 - Approach D: LoRA adapters

**Date**: expert specification - **Status**: accepted
**Decisions**: start from the COCO weights (not from the pooled model, to keep D independent);
adapters on the convolutions of the last segment of the NECK; trainable heads, everything else
frozen; implementation through **peft** (`inject_adapter_in_model`, which injects in place
without wrapping the model).
**Consequences**:

- the block indices are **computed from the structure of the model**, never hard-coded:
  changing the network size (n/s/m/l) would shift everything;
- the manifest records the **number of trainable parameters** and the list of adapted modules.
  It is essential: "LoRA rank 8" means nothing as long as one does not know what stays
  unfrozen next to the adapters, and two very different configurations get published under the
  same label;
- **grouped** (depthwise) convolutions are left out of the targets: peft then requires a rank
  divisible by `groups`, which would impose a rank of several dozens for no gain, a depthwise
  convolution carrying only a handful of parameters. The neck of the YOLO architectures has
  some; the number of exclusions is recorded in the manifest (`lora_skipped_grouped`);
- if peft proves unsuited to the `Conv2d` of this architecture, the alternative is a home-made
  wrapper (~60 lines): that would be a revision of this ADR, not a setting.

## ADR-0026 - Approach E: BatchNorm per insect group

**Date**: expert specification - **Status**: accepted
**Decisions**: every `BatchNorm2d`, statistics AND affine parameters per group; full training
from COCO (E stays independent of A); **mixed** batches, the forward pass splitting by group
then recomposing in order.
**Consequences**:

- each branch is initialised from the statistics of the pre-trained model, never randomly: the
  specialisation starts from a common point instead of destroying the COCO weights;
- the group comes from the exported file name (`<dataset>__<stem>`) at training time and from
  an explicit declaration at inference. An unknown dataset raises an error (ADR-0014);
- inference groups the images by dataset. It is not an optimisation: it is the only way to
  declare the active group, the information not existing at the layer level;
- the equivalence mixed batch / N pure batches is NOT tested (deliberate choice: compute
  cost). If a doubt arises on the training dynamics, it is the first test to write.

## ADR-0027 - Approach F: variant without legs nor hind wings

**Date**: expert request - **Status**: accepted
**Context**: the leg and hind-wing points are the hardest and the most mobile. Question asked:
does removing them free capacity for the others?
**Decision**: the 16 points concerned get `vis = 0` in the TRAINING and validation labels. The
schema stays `insect42_v1`, the test stays intact, and the predictions keep the 42 points
(contract 3 imposes the schema of the dataset).
**Consequences - to be recalled in every report**: the ground truth still contains these points
and the evaluation counts them. The `overall` metrics of F are therefore **mechanically worse**
than A's and **are not comparable**. The only valid comparison is on the `keypoint:*` scopes of
the kept points:
`python scripts/compare_models.py --exclude-keypoints leg hindwing`, which adds a
`MEAN (retained)` row. Mixing up the two readings would lead to the conclusion that F is bad
when it is simply evaluated on points it never learnt.

## ADR-0028 - Patching the Ultralytics model

**Date**: implementation - **Status**: accepted
**Context**: D and E modify the `nn.Module` built by Ultralytics, which foresees neither
adapters nor conditional normalisation. Two internals were checked on the installed version,
and both invalidate the naive approach:

- `on_pretrain_routine_start` fires BEFORE the model is built, `on_pretrain_routine_end` AFTER
  the optimiser and the EMA. **No callback fits**: a patch applied there would either be lost
  or missing from the optimiser;
- the `freeze` loop sets `requires_grad=True` again on every frozen parameter whose name does
  not match `args.freeze`. **A plain `requires_grad=False` would be silently undone.**
  **Decision**: pass a derived trainer (`train(trainer=...)`, supported); apply the patch in
  `get_model` and the freeze in `_build_train_pipeline`, just before the optimiser. All this
  dependency on the internals is isolated in `training/patching.py`.
  **Additions checked at run time (revision)**: three more pitfalls, all observed on a real
  training:
- the **validator** has its own `preprocess` and does not go through the trainer's. Without a
  relay, the per-group normalisation received the indices of the last TRAINING batch while
  facing a validation batch of a different size. The context is therefore filled on both sides;
- the **final evaluation** of Ultralytics reloads the best checkpoint and FUSES it. On a patched
  model, the fusion fails (LoRA wrappers) or would be wrong (N sets of statistics crushed into
  one). It is disabled: its metrics are only used for monitoring (§7.1);
- a class built INSIDE a function cannot be pickled, and Ultralytics serialises the model at
  every checkpoint save. `GroupBatchNorm2d` is therefore published at module level (fixed
  identity + module `__getattr__`) while keeping the torch import deferred;
- the peft LoRA wrappers do not expose the attributes of a convolution (`out_channels`). The
  adapters are therefore **merged into the base weights** before saving: the checkpoint becomes
  a standard YOLO again, reloadable and fusable, with no dependency on peft. For the per-group
  normalisation, the fusion is simply neutralised at inference.

**Consequences**: a count of trainable parameters is logged and recorded in the manifest at
every run, and a zero count raises an error. If a future Ultralytics version changes this order,
this number reveals it instead of letting a silently wrong training through. The decision logic
(which modules, which parameters, which group) is written as pure functions, testable without
torch: it is the part that breaks silently.

---

## ADR-0029 - Model identity independent of the fold

**Date**: field incident - **Status**: accepted
**Context**: two runs of the same approach with different starting weights but the same tag
were **averaged together** in the tables, as if they were two folds of one model. A wrong
result, and a silent one.
**Decision**: every manifest carries a `variant_hash`, a fingerprint of the resolved
configuration **without the fold**. Two runs share this fingerprint if and only if they are the
same model trained on different folds. The aggregation now groups by variant (`model` column),
no longer by approach. The label stays short (`approach · tag`) and is only completed by the
hash if two variants share it.
**Consequence for the nested HPO**: each outer fold LEGITIMATELY retains different
hyperparameters (ADR-0012). The keys coming from the search are therefore excluded from the
fingerprint (`hpo_overridden_keys`), otherwise each fold would form an isolated variant and the
dispersion across folds would disappear from the tables.
**Addition (revision)**: `study.optimize(n_trials=N)` adds N trials AT EVERY CALL. On a study
resumed after an interruption, an outer fold could thus get 40 trials and another one 16 — the
dispersion across folds then mixed two effects. `tune` now targets a TOTAL budget per study: it
tops up to `n_trials` and does nothing if the count is reached. The budget consumed stays
recorded in the manifest and in `<study>_best.json`.

**Addition**: the `outer_fold` and `inner_fold` columns are published. For a final run they
are the outer fold; for an HPO trial, the outer one comes from the split name
(`<split_id>__outer<k>`) and the fold is the inner index.

## ADR-0030 - Readability of the produced figures

**Date**: user feedback - **Status**: accepted
**Decisions**:

- the text colour of the heatmaps follows the **real luminance of the cell** (sRGB coefficients
  after gamma linearisation), not a threshold on the numeric value. With a palette such as
  viridis going from dark purple to bright yellow, a threshold on the value is wrong at both
  ends. The switch point retained, 0.179, is the one where the WCAG contrast of black equals
  that of white;
- `report` also writes a folder of figures **per run** under `results/runs/<run_id>/`
  (`report.per_run_figures`). The global report compares the models with each other; these
  folders let one examine a run in isolation without the next one overwriting it.

## ADR-0031 - Frozen HPO budget (closes OPEN-09)

**Date**: field calibration - **Status**: accepted
**Context**: the nested HPO at 40 trials x 5 folds x 3 inner folds required ~7 days per
approach, i.e. more than 6 weeks for the six. Unsustainable.
**Decision, identical for the SIX approaches**:
`mode=tune_once`, `n_trials=20`, `n_startup_trials=5`, `inner_folds=3`,
`pruner_warmup_steps=1`, `epochs=100`, and **four hyperparameters** per approach.
**Justifications**:

- `tune_once` rather than nested: the search runs on the inner folds of outer fold 0, then the
  retained hyperparameters are applied to the 5 folds. The test sets of folds 1 to 4 stay
  **untouched by any search**; only the estimate of fold 0 becomes slightly optimistic. A fifth
  of the rigour is lost, not all of it — and the 5 folds are kept, hence the variance, without
  which no comparison holds;
- `pruner_warmup_steps=1`: at 5, pruning NEVER fired, a trial only reporting `inner_folds`
  intermediate values. The budget was paid in full;
- `n_startup_trials=5`: Optuna's default (10) would have spent half the budget on random draws;
- **four dimensions**: 20 trials on 9 dimensions do not let the TPE learn anything. A budget of
  16 to 20 trials per task is the practice retained in published comparison protocols.

**Parameters retained**:

| Approaches | Hyperparameters                                              |
| ---------- | ------------------------------------------------------------ |
| A, B, E, F | `lr0`, `pose`, `kobj`, `weight_decay`                        |
| C          | `pose.lr0`, `pose.pose`, `crop.padding`, `crop.jitter_scale` |
| D          | `lr0`, `lora.r`, `lora.neck_blocks`, `pose`                  |

The learning rate is kept everywhere: the importance analyses (fANOVA) put it first or just
behind the depth of the network, which is fixed here by the choice of the model. `pose` and
`kobj` matter particularly outside the COCO domain — with the default values, the pose loss may
not decrease at all on non-human keypoints, and we have 42 of them. For C, the detection is
almost trivial (one insect per image, ADR-0017): the budget goes to the pose model and to the
crop geometry. For D, `alpha` is **tied to the rank** (`alpha = 2r`) instead of being searched:
the alpha/r prefactor makes the optimal learning rate almost independent of the rank, so
searching both would explore a redundancy.
**Cost**: ~65 trainings at worst, ~45 with pruning, plus 5 final ones — about 15 h per approach.
**Guard**: the Optuna study name now includes a hash of the search space and of the tuning
settings (`__sp<hash>`). Changing either one AUTOMATICALLY creates a new study, the old one
staying intact. Without it, a resume after a change would mix trials evaluated under two
protocols, the TPE would build its densities on noise, and `best_trial` could retain a trial
whose parameters are not even searched any more.

## ADR-0032 - Augmentation parameters fixed, not searched

**Date**: field calibration - **Status**: accepted
**Decision**: `degrees`, `scale`, `translate`, `fliplr`, `mosaic`, `hsv_*` and `lrf` are fixed
and taken out of the search spaces.
**Justifications**: the official YOLO26 training guide **prescribes** values for small datasets
(< 1000 images) rather than suggesting to search them — our four datasets (192 to 935 images)
are in that regime. Besides, the search systematically converged to `mosaic=0`, so a dimension
was spent confirming a known result. `lrf` interacts with the training duration and weighs much
less than `lr0`.
**Consequence**: `fliplr=0.5` stays legitimate because `flip_idx` is correct (§3.1). `degrees`
is kept at 10: the specimens are mounted, hence globally aligned.

## ADR-0033 - yolo26n as the frozen base model (closes OPEN-11)

**Date**: expert specification - **Status**: accepted
**Decision**: `yolo26n` (and `yolo26n-pose`) for the six approaches. A different size would
change the capacity of the network and the comparison would be about it rather than about the
method.
**Comparison option kept**: other starting weights remain testable as a **side study**,
outside the main table:

```
python -m insectpose.cli train experiment=exp_a_yolo_pooled \
    approach.weights=yolo11n-pose.pt tag=yolo11n
```

The `tag` is mandatory: without it, the two variants would indeed be told apart by their
`variant_hash` (ADR-0029) but would carry the same label in the figures.

## ADR-0034 - Zero prediction is a result, not an error

**Date**: field incident - **Status**: accepted
**Context**: `predict` raised a blocking error when an approach produced no detection. An
under-trained, badly tuned model, or one with too high a threshold, does exactly that — and the
pipeline stopped instead of measuring it.
**Decision**: an **empty but compliant** predictions file (contract 3) is written, with an
explicit warning. The evaluator then publishes **zero** metrics (primary metric and coverage at
0), over the denominator of the GT instances of the fold.
**Consequences**:

- the run stays COMPLETE: manifest written, hence aggregatable and auditable. A model that fails
  appears in the tables with a score of 0, which is the information sought;
- the split and the fold are now derived from the **file name** (`<split>_fold<k>.parquet`)
  and not from its content, which an empty file does not carry;
- the evaluation scope of an empty file comes from the **split**, not from the predictions:
  otherwise the denominator would be zero and the failure would become invisible;
- the qualitative export is skipped for such a run, for lack of a prediction to draw.
  An error is still raised if predictions exist but no metric is produced: that case does
  signal a broken configuration.

## ADR-0035 - Approach G: training the heads only

**Date**: implementation - **Status**: accepted (entry written after the fact, from the code)
**Context**: on YOLO26 the heads weigh about two thirds of the parameters at training time. A
LoRA variant that keeps them trainable (approach D, ADR-0025) therefore trains ~67 % of the
network, its adapters weighing only ~1.3 %.
**Decision**: approach G freezes the backbone and the neck entirely and trains the
detection/pose heads only, without any adapter (`head.blocks: 1`). It is the **control of D**,
with the same base model, the same search space keys as A/B/E/F and the same budget.
**Consequences**: comparing D and G gives three readings — G close to D: the adapters bring
nothing, only the head retraining counts; D clearly above G: the adapters do bring something;
G close to A: the COCO backbone transfers well and freezing most of the network is enough.

## ADR-0036 - Approach H: LoRA adapters per insect group

**Date**: implementation - **Status**: accepted (entry written after the fact, from the code)
**Decision**: a trunk common to every order plus one set of LoRA adapters per order, trained
in TWO PHASES within one run: (1) the whole model is trained on the whole train of the fold,
without adapters, for `epoch_split` of the epochs; (2) the trunk is reloaded, NEW adapters are
injected, everything else is frozen, and they are trained per order for the remaining epochs.
**Consequences**:

- the adapters are injected in phase 2 only: saving merges them into the base weights
  (ADR-0025), so adapters injected in phase 1 would no longer exist to specialise;
- no data is set aside: the adapters see images the trunk has already seen — a split specific
  to this approach would break §6.2 and measure the data volume rather than the method;
- the heads are trained in phase 1 and frozen in phase 2: unfreezing them per group would give
  four almost complete models, the category of B;
- the total epoch budget equals the other approaches' (`epoch_split=0.6` = 60 % trunk, 40 %
  adapters), and `epoch_split` replaces `kobj` among the four searched hyperparameters.

## ADR-0037 - The delivered model is an ensemble of `yolo_pooled`

**Date**: project decision - **Status**: accepted (replaces the export of a single model
trained on all the images)
**Decision**: `retained_models/pose/` holds an ENSEMBLE. `train` (and `evaluate
run_id=...`) replaces it with a single model; `tune` replaces it with the models of its outer
folds, one per fold. The pipeline runs every model of the folder, writes the mean and the
standard deviation of every keypoint coordinate, and computes the measurements and their
validity on the mean. Only the approaches of `retain.approaches` (`yolo_pooled`) are exported.
**Consequences**:

- every command empties the folder before writing: two trainings never mix;
- the standard deviation of the ensemble measures the disagreement between models, zero with
  a single one;
- `tuning.final_full_fit` becomes `false` by default. At `true`, the model trained on all the
  images is ADDED to the ensemble; it has seen the tests of the folds;
- `ensemble.json` carries `cv_estimate`, the measured performance of the outer folds.

## ADR-0038 - `kp_infos.yaml`, single definition of the keypoints and measurements

**Date**: project decision - **Status**: accepted (replaces `configs/keypoints/insect42_v1.yaml`
and `configs/measurements/insect42_v1.yaml`)
**Decision**: the `insect42_v1` schema (points, difficulties, symmetries, skeleton) and the
measurements are read from `kp_infos.yaml`, at the repository root, shared with the annotation,
the measurement classifiers and the pipeline. `configs/keypoints/<name>.yaml` is still read
first, for a study schema. `intertegular distance` is defined on `left/right-forewing-base`, as
in the pipeline and the measurement classifiers (the file of this module said `hindwing-base`).

## ADR-0039 - `folds`: the outer folds a command runs

**Date**: project decision - **Status**: accepted (amends ADR-0037)
**Decision**: the `folds` key (a list of outer folds, a single fold, or `"all"`) says which
outer folds `train` and `tune` run. `train` without `folds` still runs `fold` alone; with
`folds`, it trains each fold in turn and the folds form ONE ensemble in
`retained_models/pose/` (the first replaces the previous ensemble, the next ones are added,
as the final folds of `tune`). `tune` without `folds` still retrains every outer fold; with
`folds`, only these are retrained (and, in `nested` mode, searched).
**Consequences**:

- `folds` is operational, like `retain`: it enters neither the `run_id` nor the
  `variant_hash`. Training fold 0 alone, then `folds=[0,1]`, skips fold 0 the second time;
- `ensemble.json` is now written after `train` too, with `source: train`, the folds and
  `cv_estimate` over these folds: each one is measured on its own untouched test, so the
  estimate stays honest, but on fewer folds it is less precise;
- the number of outer folds itself (`cv.n_folds`) stays frozen at 5 (CONVENTIONS.md §6.2): `folds`
  chooses among them, it never changes the split.

---

**No open decision.** Every protocol question is settled (ADR-0006 to ADR-0039). Any later
change goes through a new ADR and, if it touches a metric, through an increment of
`eval.version`.
