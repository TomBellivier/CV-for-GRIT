# convert processed files in coco format to yolo format with keypoints
import os

from ultralytics.utils.files import increment_path
from ultralytics.utils import TQDM
from ultralytics.data.converter import coco91_to_coco80_class
from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import shutil
import yaml

TOTAL = [
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

# get skeleton idx
SKELETON_NAMES = [
    ["head-top", "neck"],
    ["head-top", "left-antenna-0"],
    ["head-top", "right-antenna-0"],
    ["left-antenna-0", "left-antenna-1"],
    ["left-antenna-1", "left-antenna-2"],
    ["right-antenna-0", "right-antenna-1"],
    ["right-antenna-1", "right-antenna-2"],
    ["neck", "head-left"],
    ["neck", "head-right"],
    ["neck", "left-eye"],
    ["neck", "right-eye"],
    ["neck", "thorax-left"],
    ["neck", "thorax-right"],
    ["thorax-left", "thorax-bottom"],
    ["thorax-right", "thorax-bottom"],
    ["thorax-bottom", "body-left"],
    ["thorax-bottom", "body-right"],
    ["body-left", "body-tip"],
    ["body-right", "body-tip"],
    ["thorax-bottom", "left-forewing-base"],
    ["left-forewing-base", "left-forewing-front"],
    ["left-forewing-base", "left-forewing-rear"],
    ["left-forewing-tip", "left-forewing-front"],
    ["left-forewing-tip", "left-forewing-rear"],
    ["thorax-bottom", "right-forewing-base"],
    ["right-forewing-base", "right-forewing-front"],
    ["right-forewing-base", "right-forewing-rear"],
    ["right-forewing-tip", "right-forewing-front"],
    ["right-forewing-tip", "right-forewing-rear"],
    ["thorax-bottom", "left-hindwing-base"],
    ["left-hindwing-base", "left-hindwing-front"],
    ["left-hindwing-base", "left-hindwing-rear"],
    ["left-hindwing-tip", "left-hindwing-front"],
    ["left-hindwing-tip", "left-hindwing-rear"],
    ["thorax-bottom", "right-hindwing-base"],
    ["right-hindwing-base", "right-hindwing-front"],
    ["right-hindwing-base", "right-hindwing-rear"],
    ["right-hindwing-tip", "right-hindwing-front"],
    ["right-hindwing-tip", "right-hindwing-rear"],
    ["thorax-bottom", "left-leg-0"],
    ["left-leg-0", "left-leg-1"],
    ["left-leg-1", "left-leg-2"],
    ["left-leg-2", "left-leg-3"],
    ["thorax-bottom", "right-leg-0"],
    ["right-leg-0", "right-leg-1"],
    ["right-leg-1", "right-leg-2"],
    ["right-leg-2", "right-leg-3"]
]


def make_config_file(dataset_name, printing=False):
    base_idx = {TOTAL[i]:i for i in range(len(TOTAL))}
    flip_idx = {}
    for x in TOTAL:
        if "left" in x:
            flip_idx[x.replace("left", "right")] = base_idx[x]
            
        elif "right" in x:
            flip_idx[x.replace("right", "left")] = base_idx[x]
        
        else:
            flip_idx[x] = base_idx[x]

    fliped_total = sorted([k for k, v in flip_idx.items()], key=lambda x : flip_idx[x])
    if printing :
        print("Flipped indices:", [base_idx[x] for x in fliped_total])

    squeleton_idx = [[base_idx[x] for x in pair] for pair in SKELETON_NAMES]
    if printing:
        print("Skeleton indices:", squeleton_idx)
    
    config = {
        "path" : dataset_name,
        "train" : "images/train",
        "val" : "images/val",
        "test" : "images/test",
        "kp_shape" : [len(TOTAL), 3],
        "skeleton" : squeleton_idx,
        "flip_idx" : [base_idx[x] for x in fliped_total],
        "names": {0 : "insects"},
        "kp_names": {0 : TOTAL}
    }

    with open(f"{dataset_name}/yolo-config.yaml", "w") as f:
        yaml.dump(config, f)

def convert_coco(
    labels_dir: str = "../coco/annotations/",
    image_dir: str = "../coco/images/",
    save_dir: str = "coco_converted/",
    yolo_conversion_done_dir = "./yolo-conversion-done/",
    TVT_split: list[int] = [0.8, 0.1, 0.1],
    use_keypoints: bool = False,
    cls91to80: bool = True
):
    """Convert COCO dataset annotations to a YOLO annotation format suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        TVT_split (list[int], optional): Proportions for train/val/test splits. Should sum to 1.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
    """
  
    # Convert classes
    coco80 = coco91_to_coco80_class()
    all_files = sorted([p for p in Path(labels_dir).resolve().glob("*.json")])
    print(all_files)

    # Import json
    for json_file in all_files:
        lname = json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / lname  # folder name
        fn = increment_path(fn)
        if TVT_split:
            (fn / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (fn / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (fn / "labels" / "test").mkdir(parents=True, exist_ok=True)
            (fn / "images" / "train").mkdir(parents=True, exist_ok=True)
            (fn / "images" / "val").mkdir(parents=True, exist_ok=True)
            (fn / "images" / "test").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Create image dict
        images = {f"{x['id']:d}": x for x in data["images"]}
        # Create image-annotations dict
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        # makes random splits for TVT
        if TVT_split:
            img_ids = list(annotations.keys())
            np.random.shuffle(img_ids)
            n = len(img_ids)
            train_ids = set(img_ids[: int(TVT_split[0] * n)])
            val_ids = set(img_ids[int(TVT_split[0] * n) : int((TVT_split[0] + TVT_split[1]) * n)])
            split_dict = {img_id: "train" if img_id in train_ids else "val" if img_id in val_ids else "test" for img_id in img_ids}

        # Write labels file
        img_idx = 0
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = img["file_name"]

            bboxes = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    if use_keypoints:
                        if ann.get("keypoints") is None:
                            continue
                        kp = box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        keypoints.append(
                            kp
                        )
                    bboxes.append(box)

            # Write
            if TVT_split:
                fd = fn / Path("labels") / split_dict[img_id] 
            else:
                fd = fn / Path("labels")
            with open((fd / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    line = (*(keypoints[i]),)  # cls, box, keypoints
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
            
            # copy corresponding image to new location
            if image_dir:
                if TVT_split:
                    shutil.copy(Path(image_dir) / img["file_name"], fn / Path("images") / split_dict[img_id] / img["file_name"])
                else:
                    shutil.copy(Path(image_dir) / img["file_name"], fn / Path("images") / img["file_name"])
            
            img_idx += 1

        os.rename(json_file, Path(yolo_conversion_done_dir) / json_file.name)
        
        make_config_file(fn, printing=False)


if __name__ == "__main__":
    # IT NEEDS TO BE ONLY ONE FILE IN THE FOLDER
    # except if every files uses images from the same directory 
    IMAGE_DIR = "C:/Users/tombe/Documents/_MLE/CV-for-GRIT/databases/hawaii_beetles_images/individual_specimens/08/"
    convert_coco(
        labels_dir="./coco-converted/", 
        image_dir = IMAGE_DIR, 
        save_dir = "../models/datasets/", use_keypoints=True
    )

