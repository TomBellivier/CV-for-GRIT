"""
Convert Label Studio keypoint annotations (butterfly/insect pose) to COCO keypoints format.

COCO output format:
  {
    "images": [{"id": ..., "file_name": ..., "width": ..., "height": ...}],
    "annotations": [{
        "id": ..., "image_id": ..., "category_id": 1,
        "keypoints": [x1,y1,v1, x2,y2,v2, ...],
        "num_keypoints": <count of v>0>
    }],
    "categories": [{"id": 1, "name": "insect", "keypoints": [...], "skeleton": [...]}]
  }

Visibility flags:
  0 = absent
  1 = occluded (labeled but not visible, or inferred missing)
  2 = visible (labeled and visible)

Rules:
  - Central keypoints are ALWAYS expected: if missing → (0, 0, 1)  [occluded]
  - Limb keypoints belong to named groups. A group is "active" if ANY point of
    that group appears at least once across the ENTIRE file. If active and a
    specific point is missing on an image → (0, 0, 1). If the whole group is
    absent from the entire file → (0, 0, 0) for every instance.
"""

import json
import os
import sys
from collections import defaultdict
import pathlib

# ── Ordered keypoint list ──────────────────────────────────────────────────────

CENTRAL_KPS = [
    "head-top",
    "head-left",
    "head-right",
    "left-eye",
    "right-eye",
    "neck",
    "thorax-left",
    "thorax-right",
    "thorax-bottom",
    "body-left",
    "body-right",
    "body-tip",
]

LIMB_KPS = [
    "left-antenna-0",
    "left-antenna-1",
    "left-antenna-2",
    "right-antenna-0",
    "right-antenna-1",
    "right-antenna-2",
    "left-forewing-base",
    "left-forewing-tip",
    "left-forewing-front",
    "left-forewing-rear",
    "right-forewing-base",
    "right-forewing-tip",
    "right-forewing-front",
    "right-forewing-rear",
    "left-hindwing-base",
    "left-hindwing-tip",
    "left-hindwing-front",
    "left-hindwing-rear",
    "right-hindwing-base",
    "right-hindwing-tip",
    "right-hindwing-front",
    "right-hindwing-rear",
    "left-leg-0",
    "left-leg-1",
    "left-leg-2",
    "left-leg-3",
    "right-leg-0",
    "right-leg-1",
    "right-leg-2",
    "right-leg-3",
]

ALL_KPS = CENTRAL_KPS + LIMB_KPS
KP_INDEX = {name: i for i, name in enumerate(ALL_KPS)}

# Limb groups: group_prefix → list of member keypoint names
LIMB_GROUPS = {
    "left-antenna":   ["left-antenna-0",   "left-antenna-1",   "left-antenna-2"],
    "right-antenna":  ["right-antenna-0",  "right-antenna-1",  "right-antenna-2"],
    "left-forewing":  ["left-forewing-base","left-forewing-tip","left-forewing-front","left-forewing-rear"],
    "right-forewing": ["right-forewing-base","right-forewing-tip","right-forewing-front","right-forewing-rear"],
    "left-hindwing":  ["left-hindwing-base","left-hindwing-tip","left-hindwing-front","left-hindwing-rear"],
    "right-hindwing": ["right-hindwing-base","right-hindwing-tip","right-hindwing-front","right-hindwing-rear"],
    "left-leg":       ["left-leg-0","left-leg-1","left-leg-2","left-leg-3"],
    "right-leg":      ["right-leg-0","right-leg-1","right-leg-2","right-leg-3"],
}

# margin for the bounding box, in pixel
MARGIN = 10

# Map each limb keypoint → its group name
KP_TO_GROUP = {}
for group_name, members in LIMB_GROUPS.items():
    for m in members:
        KP_TO_GROUP[m] = group_name


# ── Helpers ────────────────────────────────────────────────────────────────────

def pct_to_abs(x_pct: float, y_pct: float, width: int, height: int):
    """Convert Label Studio percentage coordinates to absolute pixel coordinates."""
    return round(x_pct / 100.0 * width, 2), round(y_pct / 100.0 * height, 2)


def extract_dims_and_kps(annotation_results):
    """
    Parse a list of Label Studio result dicts.
    Returns:
        (orig_width, orig_height, {kp_name: (abs_x, abs_y)})
    """
    orig_w = orig_h = None
    kp_map = {}

    for res in annotation_results:
        if res.get("type") != "keypointlabels":
            continue
        orig_w = res["original_width"]
        orig_h = res["original_height"]
        val = res["value"]
        label = val["keypointlabels"][0]
        ax, ay = pct_to_abs(val["x"], val["y"], orig_w, orig_h)
        kp_map[label] = (ax, ay)

    return orig_w, orig_h, kp_map


# ── Main conversion ────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # ── Pass 1: Discover which limb groups are "active" across the whole file ──
    active_groups = defaultdict(bool)  # group_name → True if any member seen

    for task in tasks:
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for res in annotation.get("result", []):
                if res.get("type") != "keypointlabels":
                    continue
                label = res["value"]["keypointlabels"][0]
                if label in KP_TO_GROUP:
                    active_groups[KP_TO_GROUP[label]] = True

    print("Active limb groups:", [g for g, v in active_groups.items() if v])
    print("Absent limb groups:", [g for g in LIMB_GROUPS if not active_groups[g]])

    # ── Pass 2: Build COCO structures ──────────────────────────────────────────
    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for task in tasks:
        task_id = task["id"]
        file_name = task["data"]['img']  # original upload filename
        if "\\data\\local-files\\?d=" in str(pathlib.Path(file_name)):
            file_name = str(pathlib.Path(file_name)).split("%5C")[-1]
        else:
            file_name = str(pathlib.Path(file_name)).split("\\")[-1]
        print(file_name)

        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue

            orig_w, orig_h, kp_map = extract_dims_and_kps(annotation.get("result", []))

            if orig_w is None:
                print(f"  ⚠ Task {task_id}: no keypoint results found, skipping annotation.")
                continue

            ## Build flat keypoints list [x1,y1,v1, x2,y2,v2, ...]
            # limits of the bounding box  : max and min for both x and y
            xs = []
            ys = []
            keypoints = []
            for kp_name in ALL_KPS:
                if kp_name in kp_map:
                    x, y = kp_map[kp_name]
                    keypoints.extend([x, y, 2])          # labeled → visible
                    xs.append(x)
                    ys.append(y)

                elif kp_name in CENTRAL_KPS:
                    keypoints.extend([0, 0, 0])           # central, missing → occluded
                else:
                    group = KP_TO_GROUP[kp_name]
                    if active_groups[group]:
                        keypoints.extend([0, 0, 0])       # group active, point missing → occluded
                    else:
                        keypoints.extend([0, 0, 0])       # group entirely absent → absent

            limits = (max(0, min(xs) - MARGIN), min(orig_w, max(xs)+MARGIN),
                      max(0, min(ys) - MARGIN), min(orig_h, max(ys)+MARGIN))
            
            num_kps = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)

            coco_annotations.append({
                "id": annotation_id,
                "image_id": task_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": num_kps,
                "bbox" : [limits[0], limits[2], limits[1] - limits[0], limits[3] - limits[2]]
            })
            annotation_id += 1

        # Add image entry (deduplicated by task_id)
        coco_images.append({
            "id": task_id,
            "file_name": file_name,
            "width": orig_w,
            "height": orig_h,
            "date_captured" : "2026-03-26 09:42:10"
        })

    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {
                "id": 1,
                "name": "insect",
                "keypoints": ALL_KPS,
                "skeleton": [],   # add connectivity pairs here if needed
            }
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Converted {len(coco_images)} image(s), "
          f"{len(coco_annotations)} annotation(s) → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_json_to_coco.py <input.json> <output_coco.json>")
        input_dir  = "./annotations/to convert/"
        processed_dir = "./annotations/coco-converted/"
        converted_dir = "./annotations/convert-done/"

        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(processed_dir, filename)
                convert(input_path, output_path)
                os.rename(input_path, os.path.join(converted_dir, filename))

        sys.exit(1)

    input_json  = sys.argv[1]
    output_json = sys.argv[2]

    if not os.path.isfile(input_json):
        print(f"Error: input file '{input_json}' not found.")
        sys.exit(1)
    
    print("Converting ")

    convert(input_json, output_json)