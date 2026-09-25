# CV-for-GRIT
Implementation of Computer Vision model to measure insect traits from images. This would be used to help compiling the Global Repository of Insect Traits. 

# Keypoints and measurements
`kp_infos.yaml`, at the root, is the ONLY definition of the keypoints (order, difficulty, left/right symmetry, colour), the skeleton, the measurements, their left/right pairs and the anatomical parts. Every module reads it — directly, or through `kp_infos.py` (`from kp_infos import KEYPOINT_NAMES, MEASUREMENTS, ...`). The keypoint order is baked into the trained models: append new points at the end.

# How to use 

## Split databases
1. In a terminal, run the command "python ./all_images/split_image_database.py ./all_images/[image folder name] [number of subfolder to create]".
2. You can add "--reverse" to undo the previous action. In this case, "number of subfolder to create" is not required.

## Check annotations 
1. In a terminal, run "python annotation_tools/import_volunteer_file.py --json_file [volunteer export].json --image_folders [image folder] [other image folder] ...". Any number of image folders can be given, each one is searched recursively. The rewritten export is written next to the input file, as "[volunteer export]_local.json".
Checking : 
1. On a terminal, type the command "start_label_studio". 
2. On Label Studio project, add a storage source (local) at the image directory specified just before. 
3. Import the modified file in Label studio, it should find every images.

## Convert annotations JSON into CSV
1. In a terminal, run "python annotation_tools/labelstudio_to_csv.py --json_files [export].json --output annotations.csv". Several JSON files and/or folders of JSON files can be given at once, they are all merged into the same CSV.
2. The CSV holds one row per annotation, with the keypoint coordinates in pixels ("[keypoint]_x", "[keypoint]_y", "[keypoint]_v"). Building a YOLO dataset from that CSV is a separate step.

## Classify the measurements
1. In a terminal, run "python annotation_tools/measurement_validation/main.py --annotations_csv annotations.csv", with the CSV produced at the previous step.
2. Each annotated image is displayed with its keypoints and the segments of every measurement. Left click/drag on a segment marks it measurable, right click/drag marks it non measurable; clicking a measurement in the side panel toggles it as a whole. Enter validates the image and moves to the next one.
3. The result is written to "annotation_data/meas_classifier/[annotations]_measurements.csv": one row per annotation, one "[measurement]_status" column per measurement, and no keypoint position. This is the file `modules/meas_classifier/` trains on.

## Gather every annotation into one file
Every training module reads a single table, `annotation_data/annotation_data.csv`: one row per image, one column per piece of information, an empty cell where nothing was annotated.

1. Convert the pose annotations to CSV first — the merge only ever reads CSVs: "python annotation_tools/labelstudio_to_csv.py --json_files annotation_data/label_studio_annotations --output annotation_data/pose/pose_annotations.csv".
2. Annotate the scale by hand in "annotation_data/scale/scale_annotations.csv" (columns and current content: see `annotation_data/scale/README.md`).
3. In a terminal, run "python annotation_tools/build_annotation_data.py". With no argument it merges the three CSV sources at their standard locations:
   - pose: "annotation_data/pose/pose_annotations.csv" (step 1);
   - scale: "annotation_data/scale/scale_annotations.csv";
   - measurements: every CSV in "annotation_data/meas_classifier/".
4. Re-run steps 1 and 3 after any new annotation batch: the modules never read the individual files again.

## Modules
- `kp_infos.yaml` / `kp_infos.py` — keypoints, skeleton, measurements (see above).
- `results/` — every data analysis produced by the modules, one sub-folder per module (see `results/README.md`).
- `annotation_data/` — every annotation. `annotation_data.csv` is the merged table the modules train on; the sub-folders hold the sources it is built from (`label_studio_annotations/`, `scale/`, `meas_classifier/`, `ruler_detection/`).
- `annotation_tools/` — tools used around the annotation campaigns: `import_volunteer_file.py` (re-point a volunteer export at local images), `labelstudio_to_csv.py` (Label Studio export → CSV), `measurement_validation/` (app classifying each measurement as measurable or not), `build_annotation_data.py` (merge everything into `annotation_data/annotation_data.csv`).
- `datasets/` — YOLO-format datasets and the scripts that build/maintain them (`fuze_datasets.py`, `list_datasets.py`, `restore_dataset.py`, `create_background_class.py`, `generate_ground_truth_csv.py`).
- `modules/architectures/` — pose model training, from `annotation_data.csv` via its `annotation_csv` adapter (see its own README). `train` retains one `yolo_pooled` model, `tune` one per outer fold: this ensemble is what the pipeline runs.
- `modules/meas_classifier/` — measurement-validity classifiers, trained from `annotation_data.csv` (keypoint geometry + measurement statuses).
- `modules/ruler_detection/`, `modules/scale_bar_detection/` — scale detection. `ruler_detection` is evaluated against the ruler columns of `annotation_data.csv`.
- `retained_models/` — the trained artifacts the pipeline runs on (pose ensemble, scale-bar detector, measurement-validity classifiers). Each module writes its models here; nothing is copied by hand (see `retained_models/README.md`).
- `pipeline/` — end-to-end inference over a folder of images, using the models in `retained_models/`: every pose model is run, and the mean and standard deviation of each keypoint coordinate are written (see `pipeline/processing/README.md`).
