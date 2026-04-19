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
    ["neck", "thorax-bottom"],
    ["thorax-left", "thorax-bottom"],
    ["thorax-right", "thorax-bottom"],
    ["thorax-bottom", "body-left"],
    ["thorax-bottom", "body-right"],
    ["thorax-bottom", "body-tip"],
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
    ["body-left", "left-leg-1"],
    ["left-leg-0", "left-leg-1"],
    ["left-leg-1", "left-leg-2"],
    ["left-leg-2", "left-leg-3"],
    ["thorax-bottom", "right-leg-0"],
    ["body-right", "right-leg-1"],
    ["right-leg-0", "right-leg-1"],
    ["right-leg-1", "right-leg-2"],
    ["right-leg-2", "right-leg-3"]
]


def check_filter(kp_name, keywords):
    for kw in keywords:
        if kw in kp_name:
            return False
    return True

def filter(all_kps, keywords):
    new_kps = []
    for element in all_kps:
        if isinstance(element, list):
            new_element = filter(element, keywords)
            if len(new_element) == len(element):
                new_kps.append(new_element)
        elif isinstance(element, str):
            if check_filter(element, keywords):
                new_kps.append(element)
    return new_kps


def make_config_file(dataset_name, filter_keywords=[], printing=False):
    filtered_total = filter(TOTAL, filter_keywords)
    filtered_skeleton = filter(SKELETON_NAMES,filter_keywords)

    base_idx = {filtered_total[i]:i for i in range(len(filtered_total))}
    flip_idx = {}
    for x in filtered_total:
        if "left" in x:
            flip_idx[x.replace("left", "right")] = base_idx[x]
            
        elif "right" in x:
            flip_idx[x.replace("right", "left")] = base_idx[x]
        
        else:
            flip_idx[x] = base_idx[x]

    fliped_total = sorted([k for k, v in flip_idx.items()], key=lambda x : flip_idx[x])
    if printing :
        print("Flipped indices:", [base_idx[x] for x in fliped_total])

    squeleton_idx = [[base_idx[x] for x in pair] for pair in filtered_skeleton]
    if printing:
        print("Skeleton indices:", squeleton_idx)
    print(str(dataset_name))
    config = {
        "path" : "models/datasets/" + Path(dataset_name).name,
        "train" : "images/train",
        "val" : "images/val",
        "test" : "images/test",
        "kpt_shape" : [len(filtered_total), 3],
        "skeleton" : squeleton_idx,
        "flip_idx" : [base_idx[x] for x in fliped_total],
        "names": {0 : "insects"},
        "kpt_names": {0 : filtered_total}
    }

    with open(f"{dataset_name}/yolo-config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)

def convert_coco(
    labels_dir: str = "../coco/annotations/",
    image_dir: str = "../coco/images/",
    save_dir: str = "coco_converted/",
    yolo_conversion_done_dir = "./annotations/yolo-conversion-done/",
    filter_keywords = [],
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

    filtered_kpts = filter(TOTAL, filter_keywords)

    # Import json
    for json_file in all_files:
        new_image_dir = image_dir
        if image_dir == "ask":
            new_image_dir = input(f"Put the path to the images corresonding to {json_file} (or skip to skip this file) : ")
            if new_image_dir == "skip":
                continue
        elif image_dir is not None:
            new_image_dir = image_dir
        else:
            continue
        
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
            file_name = Path(img["file_name"]).name
            file_name = file_name.split("%5C")[-1]

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

                        filtered_kp = []
                        for kp_idx in range(len(filtered_kpts)):
                            if check_filter(TOTAL[kp_idx], filter_keywords):
                                for i in range(3):
                                    filtered_kp.append(ann["keypoints"][kp_idx*3 + i])

                        kp = box + (np.array(filtered_kp).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        keypoints.append(
                            kp
                        )
                    bboxes.append(box)

            # Write
            if TVT_split:
                fd = fn / "labels" / split_dict[img_id] 
            else:
                fd = fn / "labels"

            with open((fd / file_name).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    line = (*(keypoints[i]),)  # cls, box, keypoints
                    file.write(("%g " * len(line)).rstrip() % line)
            
            if TVT_split:
                fd_images = fn /"images" / split_dict[img_id] 
            else:
                fd_images = fn / "images"
            
            
            # copy corresponding image to new location
            if new_image_dir:
                
                if TVT_split:
                    shutil.copy(Path(new_image_dir) / file_name, fd_images / file_name, follow_symlinks=False)
                else:
                    shutil.copy(Path(new_image_dir) / file_name, fd_images / file_name, follow_symlinks=False)
            
            img_idx += 1

        os.rename(json_file, Path(yolo_conversion_done_dir) / json_file.name)
        
        make_config_file(fn, filter_keywords=filter_keywords, printing=False)


if __name__ == "__main__":
    # IT NEEDS TO BE ONLY ONE FILE IN THE FOLDER
    # except if every files uses images from the same directory 
    IMAGE_DIR = "C:\\Users\\tombe\\Documents\\_MLE\\CV-for-GRIT\\databases\\hawaii_beetles_images\\individual_specimens\\01"
    
    convert_coco(
        labels_dir="./annotations/coco-converted/", 
        image_dir = "ask", 
        save_dir = "./models/datasets/", 
        filter_keywords = [],
        use_keypoints=True
    )

