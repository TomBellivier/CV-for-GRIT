from ruler_detection import detect_ruler_from_rgb # returns value and line

from pathlib import Path
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt
import cv2
import time
import os
import numpy as np
import json

REPO_ROOT = Path(__file__).resolve().parents[2]

# Single annotation table: pose + scale + measurement validity, one row per image
# (see annotation_tools/build_annotation_data.py). The rulers are the rows that carry
# a ruler_line_min / ruler_line_max range.
ANNOTATION_DATA = REPO_ROOT / "annotation_data" / "annotation_data.csv"
ALL_IMAGES_DIR = REPO_ROOT / "annotated_images" / "full databases"
# Outputs in the shared results/ folder of the repository (see results/README.md); the
# ruler_detection_evaluation.ipynb notebook reads them back from there.
RESULTS_DIR = REPO_ROOT / "results" / "ruler_detection"

path_images = list(ALL_IMAGES_DIR.rglob("*.jpg")) + list(ALL_IMAGES_DIR.rglob("*.png")) + list(ALL_IMAGES_DIR.rglob("*.jpeg"))+ list(ALL_IMAGES_DIR.rglob("*.JPG"))
images = [Path(str(i).replace("\\", "/")).name for i in path_images]

if not ANNOTATION_DATA.is_file():
    raise SystemExit(
        f"Annotation table missing: {ANNOTATION_DATA}\n"
        "Build it with: python annotation_tools/build_annotation_data.py"
    )
df = pd.read_csv(ANNOTATION_DATA)
df = df.rename(columns={"image_name": "Name", "ruler_line_min": "Min", "ruler_line_max": "Max"})
annotated = df["Min"].notna() & df["Max"].notna()
print(f"{int(annotated.sum())}/{len(df)} image(s) with an annotated ruler range")
df = df.loc[annotated].reset_index(drop=True)

ONLY_HORIZONTAL_RULER = True

def process_images(ratio, phase=0):
    predicted_lines = []
    ruler_confidences = []
    prediction_times = []
    image_sizes = []
    scale_values = []
    label_mins = []
    label_maxs = []
    directions = []

    for img in tqdm.tqdm(df["Name"]):
        if img in images:
            path = Path(path_images[images.index(img)])
            label_min, label_max = df[df["Name"] == img]["Min"].iloc[0], df[df["Name"] == img]["Max"].iloc[0]
            img_bgr = cv2.imread(str(path))
            shape = img_bgr.shape
            if img_bgr is None:
                raise ValueError(f"cv2 could not read image: {path}")
            rotated_img = np.rot90(img_bgr)

            px_per_mm, line, ruler_conf = None, None, 0.0

            T = time.time()
            px_per_mm_h, line_h, ruler_conf_h = detect_ruler_from_rgb(img_bgr, ratio=ratio, phase=phase)

            if not ONLY_HORIZONTAL_RULER:
                px_per_mm_v, line_v, ruler_conf_v = detect_ruler_from_rgb(rotated_img, ratio=ratio, phase=phase)
            else:
                px_per_mm_v, line_v, ruler_conf_v  = None, None, 0.0
                
            if ruler_conf_v is None or (ruler_conf_h is not None and ruler_conf_h > ruler_conf_v):
                px_per_mm, line, ruler_conf = px_per_mm_h, line_h, ruler_conf_h
                directions.append(1)
            elif ruler_conf_h is None or (ruler_conf_v is not None and ruler_conf_v > ruler_conf_h):
                px_per_mm, line, ruler_conf = px_per_mm_v, line_v, ruler_conf_v
                directions.append(0)
            else:
                if path.name in problems:
                    problems[path.name] += 1
                else:
                    problems[path.name] = 1
                directions.append(-1)

            predicted_lines.append(line if line is not None else -100)
            ruler_confidences.append(ruler_conf if ruler_conf is not None else 0)
            prediction_times.append(round(time.time() - T, 5))
            image_sizes.append(shape[0] * shape[1])
            scale_values.append(px_per_mm * (shape[0] if directions[-1]==1 else shape[1]) if px_per_mm is not None else 0)
            label_maxs.append(label_max)
            label_mins.append(label_min)
        else:
            print(img, "does not exist")
    return predicted_lines, ruler_confidences, prediction_times, image_sizes, scale_values, label_mins, label_maxs, directions

all_ratios = [i for i in range(1, 11)]

data = {
    "all_lines" : [],
    "all_confs" : [],
    "all_times" : [],
    "all_sizes" : [],
    "all_scales" : [],
    "all_label_mins" : [],
    "all_label_maxs" : [],
    "all_directions" : [],
}
data_keys = list(data.keys())

problems = {}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Your codes .... 
json.dumps(data, cls=NpEncoder)

if __name__ == "__main__":
    for ri in tqdm.tqdm(range(len(all_ratios)), colour="red"):
        r = all_ratios[ri]
        for d in data.keys():
            data[d].append([])
        for p in tqdm.tqdm(range(r), colour="blue"):
            results = process_images(r, p)
            for di in range(len(data_keys)):
                data[data_keys[di]][ri].append(results[di])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "evaluation.json", "w") as fp:
        json.dump(data, fp, cls=NpEncoder)
    with open(RESULTS_DIR / "problems.json", "w") as fp:
        json.dump(problems, fp, cls=NpEncoder)