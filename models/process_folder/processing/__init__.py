"""
processing
==========

Image-processing pipeline that runs a trained YOLO-pose model over a folder of
insect images and produces one CSV row per image containing:

    - the image name and its membership in the training / validation splits,
    - every measurement of interest, in pixels and in millimetres,
    - a confidence value for each measurement,
    - an overall pose-confidence value,
    - the detected scale (px/mm) and a confidence value for that scale.

The package is intentionally split into small, single-responsibility modules so
that each step can be read, tested and swapped independently:

    config.py                     -> all tunable parameters (edit this first)
    definitions.py                -> keypoint order, colours and measurement graph
    measurements.py               -> turn keypoints into pixel measurements
    pose_inference.py             -> load the pose model and extract keypoints
    tta.py                        -> test-time augmentation (for the TTA signal)
    confidence.py                 -> ALL confidence computations live here
    scale_bar_detection_utils.py  -> scale-bar detector (adapted from your file)
    ruler_detection.py            -> ruler detector       (adapted from your file)
    scale.py                      -> orchestrates scale-bar -> ruler fallback
    dataset_membership.py         -> train / val membership by exact file name
    pipeline.py                   -> the per-image pipeline (in-memory image)
    image_source.py               -> local folder OR Hugging Face dataset source
    parallel.py                   -> bounded, multi-thread, as-completed map
    worker.py                     -> per-thread models + CPU thread balancing
    csv_writer.py                 -> assemble and write the output CSV
"""
