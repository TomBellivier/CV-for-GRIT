"""
processing
==========

Image-processing pipeline that runs the ensemble of YOLO-pose models of
retained_models/pose/ over a folder of
insect images and produces one CSV row per image containing:

    - the image name,
    - every measurement of interest, in pixels and in millimetres,
    - a confidence value for each measurement,
    - an overall pose-confidence value,
    - the detected scale (px/mm) and a confidence value for that scale.

The package is intentionally split into small, single-responsibility modules so
that each step can be read, tested and swapped independently:

    config.py                     -> all tunable parameters (edit this first)
    definitions.py                -> keypoints and measurements, from kp_infos.yaml
    measurements.py               -> turn keypoints into pixel measurements
    pose_inference.py             -> run the pose ensemble, mean + std of the keypoints
    tta.py                        -> test-time augmentation (for the TTA signal)
    confidence.py                 -> ALL confidence computations live here
    scale.py                      -> scale-bar -> ruler fallback; the two detectors
                                     are imported from modules/scale_bar_detection/
                                     and modules/ruler_detection/
    pipeline.py                   -> the per-image pipeline (in-memory image)
    image_source.py               -> local folder OR Hugging Face dataset source
    parallel.py                   -> bounded, multi-thread, as-completed map
    worker.py                     -> per-thread models, on the device handed out
    hardware.py                   -> sizes the run to the machine (GPU/CPU, memory)
    csv_writer.py                 -> assemble and write the output CSV
"""
