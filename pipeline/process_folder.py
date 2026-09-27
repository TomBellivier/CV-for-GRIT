#!/usr/bin/env python3
"""
process_folder.py
=================

Main entry point. Runs the trained YOLO-pose model over every image of a source
and writes one CSV row per image (measurements in px and mm, per-measurement
confidences, overall pose confidence, scale and scale confidence).

Two interchangeable sources (choose with --source):
    --source folder   a local folder of images (default)
    --source hf       a Hugging Face dataset repo, streamed into RAM

Parallelism (inspired by test_process_hf.py, extended to the whole task):
    a pool of worker threads runs "download/read + decode + full pipeline",
    a bounded number of images in flight, results written as they complete.
    Each thread holds its own model copies (see worker.py). How many threads,
    on which device(s) and with how many compute threads each is decided from
    the machine (GPU(s), CPUs, free memory) by processing/hardware.py; the
    --workers / --buffer / --torch-threads options force a value.

Examples
--------
    # local folder (as before)
    python process_folder.py --source folder --input "../annotated_images/full databases"

    # the whole Hugging Face dataset (workers sized to the machine)
    python process_folder.py --source hf --dataset TomBellivier/all_images

    # only folders 1 and 2 of the dataset
    python process_folder.py --source hf --dataset TomBellivier/all_images --hf-folders 1 2

    # pose and pixel measurements only: no scale, no measurement-validity classifiers
    python process_folder.py --input images_to_process --scale none --no-measurement-classifier

Everything else is configured in processing/config.py. No model is trained.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from processing import config
from processing.csv_writer import CsvWriter
from processing.image_source import build_source
from processing.insect_group import build_group_index
from processing.measurement_classifier import load_measurement_classifiers
from processing.parallel import bounded_unordered_map
from processing.pose_inference import pose_model_paths
from processing.hardware import apply_threads, plan_resources, set_plan
from processing.worker import make_task


def parse_args():
    p = argparse.ArgumentParser(description="Measure insects over a folder or a HF dataset.")

    # --- source selection ---
    p.add_argument("--source", choices=["folder", "hf"], default="folder",
                   help="Where images come from: local 'folder' or 'hf' dataset.")
    # local folder
    p.add_argument("--input", type=str, default=None,
                   help="[folder] images folder (default: config.INPUT_FOLDER).")
    # hugging face
    p.add_argument("--dataset", type=str,
                   default=os.environ.get("HF_DATASET", "TomBellivier/all_images"),
                   help="[hf] dataset repo id (default: env HF_DATASET or "
                        "TomBellivier/all_images).")
    p.add_argument("--hf-folders", nargs="*", default=None,
                   help="[hf] restrict to these sub-folders, e.g. --hf-folders 1 2 3 "
                        "(default: the whole repo).")
    p.add_argument("--hf-token", type=str, default=None,
                   help="[hf] token for private repos (default: env HF_TOKEN).")

    # --- output / model ---
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: config.OUTPUT_CSV).")
    p.add_argument("--models", type=str, default=None,
                   help="Pose ensemble: a folder (every *.pt under it) or a single .pt "
                        "(default: config.POSE_MODELS_DIR).")

    # --- parallelism ---
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker threads (default: config.WORKERS, 'auto' = sized "
                        "from the GPUs, CPUs and free memory).")
    p.add_argument("--buffer", type=int, default=None,
                   help="Max tasks in flight = images pre-loaded ahead (default: 2 x workers).")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Compute threads PER worker (default: cpus // workers).")
    p.add_argument("--flush-every", type=int, default=None,
                   help="Force the CSV to disk every N rows for crash safety "
                        "(default: config.CSV_FLUSH_EVERY_N_ROWS; 0 disables).")

    # --- optional steps ---
    p.add_argument("--scale", choices=["auto", "scale_bar", "ruler", "none"], default=None,
                   help="Scale extraction: 'auto' = scale bar, then ruler as a fallback; "
                        "'scale_bar' or 'ruler' alone; 'none' skips it (no millimetre "
                        "values). Default: config.USE_SCALE_BAR / USE_RULER_FALLBACK.")
    p.add_argument("--no-measurement-classifier", action="store_true",
                   help="Skip the measurement-validity classifiers (and the insect-group "
                        "lookup they need). Default: config.RUN_MEASUREMENT_CLASSIFIER.")

    p.add_argument("-only_scale_annotated", action="store_true") # /!\ needs "annotations.json" file in project root

    return p.parse_args()


# --scale value -> (config.USE_SCALE_BAR, config.USE_RULER_FALLBACK)
SCALE_METHODS = {
    "auto": (True, True),
    "scale_bar": (True, False),
    "ruler": (False, True),
    "none": (False, False),
}


def main():
    args = parse_args()

    output_csv = Path(args.output) if args.output else config.OUTPUT_CSV

    # Optional steps: set before the workers start, since they read the config.
    if args.scale is not None:
        config.USE_SCALE_BAR, config.USE_RULER_FALLBACK = SCALE_METHODS[args.scale]
    if args.no_measurement_classifier:
        config.RUN_MEASUREMENT_CLASSIFIER = False

    # Point the config at the requested ensemble so every worker loads it.
    if args.models:
        config.POSE_MODELS_DIR = Path(args.models)
    pose_models = pose_model_paths()
    if not pose_models:
        raise SystemExit(
            f"No pose model (*.pt) under {config.POSE_MODELS_DIR}.\n"
            f"Models are retained by the modules under {config.RETAINED_MODELS_DIR} "
            f"(see its README): train or tune a pose model (modules/architectures) "
            f"and its ensemble lands there."
        )
    # Full scale-bar + ruler evaluation data is only needed when evaluating
    # against the manual scale annotations; see scale.py / detect_scale().
    config.ONLY_SCALE_ANNOTATED = args.only_scale_annotated

    # ---- build the image source (list of items + a per-item loader) ---------
    items, load_fn = build_source(
        args.source,
        folder=args.input,
        repo=args.dataset,
        hf_folders=args.hf_folders,
        hf_token=args.hf_token,
        only_scale_annotated=args.only_scale_annotated
    )
    total = len(items)

    # ---- size the run to the machine (GPU(s), CPUs, free memory) ------------
    model_files = list(pose_models)
    if config.USE_SCALE_BAR:
        model_files.append(config.SCALE_BAR_MODEL_PATH)
    plan = plan_resources(model_files, workers=args.workers, threads=args.torch_threads,
                          buffer=args.buffer, source=args.source)
    set_plan(plan)
    apply_threads(plan)

    print("=" * 70)
    print(f"Source        : {args.source}"
          + (f"  ({args.input or config.INPUT_FOLDER})" if args.source == "folder"
             else f"  ({args.dataset}, folders={args.hf_folders or 'ALL'})"))
    print(f"Output CSV    : {output_csv}")
    print(f"Pose ensemble : {len(pose_models)} model(s) under {config.POSE_MODELS_DIR}")
    print(f"Confidence    : {config.MEASUREMENT_CONFIDENCE_SIGNAL}")
    scale_steps = [name for name, on in (("scale bar", config.USE_SCALE_BAR),
                                         ("ruler", config.USE_RULER_FALLBACK)) if on]
    print(f"Scale         : {' then '.join(scale_steps) if scale_steps else 'disabled'}")
    print(f"Validity clf  : {'on' if config.RUN_MEASUREMENT_CLASSIFIER else 'disabled'}")
    print(f"Hardware      : {plan.describe()}")
    print(f"Images        : {total}")
    print("=" * 70)

    # ---- shared, read-only context ------------------------------------------
    group_index = build_group_index() if config.RUN_MEASUREMENT_CLASSIFIER else None
    measurement_classifiers = (
        load_measurement_classifiers() if config.RUN_MEASUREMENT_CLASSIFIER else None
    )
    task = make_task(load_fn, measurement_classifiers, group_index)

    # ---- process in parallel, write results as they complete ----------------
    ok, err = 0, 0
    start_time = time.time()
    with CsvWriter(output_csv, flush_every=args.flush_every) as writer:
        for i, (item, record, error) in enumerate(
                bounded_unordered_map(task, items, plan.workers, plan.buffer), start=1):
            key, image_name = item
            if error is not None:
                err += 1
                print(f"[error] {image_name}: {error}")
                # Still emit a row so the failed image is visible in the CSV.
                writer.write_record({"image_name": image_name})
            else:
                ok += 1
                writer.write_record(record)

            if i % 50 == 0 or i == total:
                T = time.time() - start_time
                print(f"    {i}/{total} done in {T:.1f}s (speed: {i/T:.1f} img/s) (ok={ok}, err={err})")

    print(f"\nDone. {ok} processed, {err} error(s). Results written to: {output_csv}")


if __name__ == "__main__":
    main()