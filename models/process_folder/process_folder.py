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
    a pool of --workers threads runs "download/read + decode + full pipeline",
    at most --buffer in flight, results written as they complete. Each thread
    holds its own model copies (see worker.py) and the CPU cores are shared
    across workers so the 16 CPUs are used without oversubscription.

Examples
--------
    # local folder (as before)
    python process_folder.py --source folder --input images_to_process

    # the whole Hugging Face dataset, 16 workers
    python process_folder.py --source hf --dataset TomBellivier/all_images --workers 16

    # only folders 1 and 2 of the dataset
    python process_folder.py --source hf --dataset TomBellivier/all_images --hf-folders 1 2

Everything else is configured in processing/config.py. No model is trained.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from processing import config
from processing.csv_writer import CsvWriter
from processing.dataset_membership import build_membership
from processing.image_source import build_source
from processing.parallel import bounded_unordered_map
from processing.worker import configure_cpu_threads, make_task


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
    p.add_argument("--model", type=str, default=None,
                   help="Pose model file name or path (default: config.POSE_MODEL_PATH).")

    # --- parallelism ---
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel worker threads (default: 16).")
    p.add_argument("--buffer", type=int, default=32,
                   help="Max tasks in flight = images pre-loaded ahead (default: 32).")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Compute threads PER worker (default: cpus // workers).")

    return p.parse_args()


def resolve_pose_model_path(model_arg: str | None) -> Path:
    """A bare name is looked up in TRAINED_MODELS_DIR; a path is used as-is."""
    if not model_arg:
        return config.POSE_MODEL_PATH
    candidate = Path(model_arg)
    return candidate if candidate.exists() else config.TRAINED_MODELS_DIR / model_arg


def main():
    args = parse_args()

    output_csv = Path(args.output) if args.output else config.OUTPUT_CSV

    # Point the config at the requested pose model so every worker loads it.
    config.POSE_MODEL_PATH = resolve_pose_model_path(args.model)

    # ---- build the image source (list of items + a per-item loader) ---------
    items, load_fn = build_source(
        args.source,
        folder=args.input,
        repo=args.dataset,
        hf_folders=args.hf_folders,
        hf_token=args.hf_token,
    )
    total = len(items)

    print("=" * 70)
    print(f"Source        : {args.source}"
          + (f"  ({args.input or config.INPUT_FOLDER})" if args.source == "folder"
             else f"  ({args.dataset}, folders={args.hf_folders or 'ALL'})"))
    print(f"Output CSV    : {output_csv}")
    print(f"Pose model    : {config.POSE_MODEL_PATH}")
    print(f"Confidence    : {config.MEASUREMENT_CONFIDENCE_SIGNAL}")
    print(f"Workers       : {args.workers} | buffer: {args.buffer}")
    print(f"Images        : {total}")
    print("=" * 70)

    # ---- shared, read-only context ------------------------------------------
    configure_cpu_threads(args.workers, torch_threads=args.torch_threads)
    membership = build_membership()
    task = make_task(load_fn, membership)

    # ---- process in parallel, write results as they complete ----------------
    ok, err = 0, 0
    with CsvWriter(output_csv) as writer:
        for i, (item, record, error) in enumerate(
                bounded_unordered_map(task, items, args.workers, args.buffer), start=1):
            key, image_name = item
            if error is not None:
                err += 1
                print(f"[error] {image_name}: {error}")
                # Still emit a row so the failed image is visible in the CSV.
                writer.write_record({
                    "image_name": image_name,
                    "in_train": membership.in_train(image_name),
                    "in_val": membership.in_val(image_name),
                })
            else:
                ok += 1
                writer.write_record(record)

            if i % 50 == 0 or i == total:
                print(f"    {i}/{total} done  (ok={ok}, err={err})")

    print(f"\nDone. {ok} processed, {err} error(s). Results written to: {output_csv}")


if __name__ == "__main__":
    main()
