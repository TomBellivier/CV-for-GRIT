from pathlib import Path
import shutil
from ultralytics.utils.files import increment_path
import tqdm
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

def make_pose_config_file(dataset_name, filter_keywords=[], printing=False, cls_groups=[]):
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
        "flip_idx" : [base_idx[x] for x in fliped_total]
    }
    
    if cls_groups:
        config["names"] = {i: group for i, group in enumerate(cls_groups)}
        config["kpt_names"] = {}
        for i, group in enumerate(cls_groups):
            config["kpt_names"][i] = [kp for kp in filtered_total]
            print(config["kpt_names"][i])
    else:
        config["names"] = {0 : "insects"}
        config["kpt_names"] = {0 : filtered_total}


    with open(f"{dataset_name}/yolo-config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)

def make_detect_config_file(dataset_name, groups=[]):
    config = {
        "path" : "models/datasets/" + Path(dataset_name).name,
        "train" : "images/train",
        "val" : "images/val",
        "test" : "images/test",
        "names" : {i: group for i, group in enumerate(groups)}
    }

    with open(f"{dataset_name}/yolo-config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)

def create_pose_dataset(dataset_list, final_folder, dataset_folder = "./models/datasets/", cls=False):
    all_images_files = []
    all_label_files = []

    for dataset in tqdm.tqdm(dataset_list, colour="red"):
        dataset_path = Path(dataset_folder) / dataset
        for split in ["train", "val", "test"]:
            for label_file in tqdm.tqdm((dataset_path / "labels" / split).glob("*.txt"), colour = "yellow"):
                if label_file in all_label_files:
                    print(f"Warning: label file {label_file} already exists")
                else:
                    all_label_files.append(label_file)
                    new_label_file_path = final_folder / "labels" / split / label_file.name
                    shutil.copy(label_file, new_label_file_path)
                    # modify label file to attribute correct class id
                    if cls:
                        with open(new_label_file_path, "r") as f:
                            lines = f.readlines()
                        with open(new_label_file_path, "w") as f:
                            for line in lines:
                                line = line.strip()
                                if line:
                                    parts = line.split()
                                    class_id = int(parts[0])
                                    new_class_id = dataset_list.index(dataset)
                                    parts[0] = str(new_class_id)
                                    f.write(" ".join(parts) + "\n")
            for image_file in tqdm.tqdm((dataset_path / "images" / split).glob("*.*"), colour="blue"):
                if image_file in all_images_files:
                    print(f"Warning: image file {image_file} already exists")
                else:
                    all_images_files.append(image_file)
                    shutil.copy(image_file, final_folder / "images" / split / image_file.name)
    
    make_pose_config_file(final_folder, cls_groups=dataset_list if cls else [])

def create_detect_dataset(dataset_list, final_folder, dataset_folder = "./models/datasets/"):
    all_images_files = []
    all_label_files = []

    for dataset in tqdm.tqdm(dataset_list, colour="red"):
        dataset_path = Path(dataset_folder) / dataset
        for split in ["train", "val", "test"]:
            for label_file in tqdm.tqdm((dataset_path / "labels" / split).glob("*.txt"), colour = "yellow"):
                if label_file in all_label_files:
                    print(f"Warning: label file {label_file} already exists")
                else:
                    all_label_files.append(label_file)
                    new_label_file_path = final_folder / "labels" / split / label_file.name
                    shutil.copy(label_file, new_label_file_path)
                    # modify label file to attribute correct class id
                    with open(new_label_file_path, "r") as f:
                        lines = f.readlines()
                    with open(new_label_file_path, "w") as f:
                        for line in lines:
                            line = line.strip()
                            if line:
                                parts = line.split()[:5]
                                new_class_id = dataset_list.index(dataset)
                                parts[0] = str(new_class_id)
                                f.write(" ".join(parts) + "\n")

            for image_file in tqdm.tqdm((dataset_path / "images" / split).glob("*.*"), colour="blue"):
                if image_file in all_images_files:
                    print(f"Warning: image file {image_file} already exists")
                else:
                    all_images_files.append(image_file)
                    shutil.copy(image_file, final_folder / "images" / split / image_file.name)
    
    make_detect_config_file(final_folder, groups=dataset_list)

def create_cls_dataset(dataset_list, final_folder, dataset_folder = "./models/datasets/"):
    all_images_files = []

    for dataset in tqdm.tqdm(dataset_list, colour="red"):
        dataset_path = Path(dataset_folder) / dataset
        for split in ["train", "val", "test"]:
            for image_file in tqdm.tqdm((dataset_path / "images" / split).rglob("*.*"), colour="blue"):
                if image_file in all_images_files:
                    print(f"Warning: image file {image_file} already exists")
                else:
                    all_images_files.append(image_file)
                    print(image_file)
                    shutil.copy(image_file, final_folder / split / dataset / image_file.name, follow_symlinks=False)

def fuze(dataset_name, dataset_list, dataset_folder = "./models/datasets/", erase=False, task="pose"):
    f = Path(dataset_folder) / dataset_name
    f = increment_path(f)

    if "pose" in task or task == "detect":  
        (f / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (f / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (f / "labels" / "test").mkdir(parents=True, exist_ok=True)
        (f / "images" / "train").mkdir(parents=True, exist_ok=True)
        (f / "images" / "val").mkdir(parents=True, exist_ok=True)
        (f / "images" / "test").mkdir(parents=True, exist_ok=True)
    elif task == "cls":  
        for dataset in dataset_list:
            (f / "train" / dataset).mkdir(parents=True, exist_ok=True)
            (f / "val" / dataset).mkdir(parents=True, exist_ok=True)
            (f / "test" / dataset).mkdir(parents=True, exist_ok=True)

    if task == "pose":
        create_pose_dataset(dataset_list, f, dataset_folder)
    elif task == "pose+cls":
        create_pose_dataset(dataset_list, f, dataset_folder, cls=True)
    elif task == "detect":
        create_detect_dataset(dataset_list, f, dataset_folder)
    elif task == "cls":
        create_cls_dataset(dataset_list, f, dataset_folder)
    else:
        print(f"Error: task {task} not supported")

    if erase and input("Are you sure you want to delete the previous files? (y/n) ") == "y":
        print("Deleting previous files")
        for dataset in dataset_list:
            shutil.rmtree(Path(dataset_folder) / dataset)

if __name__ == "__main__":
    fuze("AllSpecies-detect", 
         [
            # "HawaiiBeetles01",
            # "HawaiiBeetles03-Henrique",
            # "HawaiiBeetles07-Marija",
            # "HawaiiBeetles08",
            # "HawaiiBeetles16",
            # "HawaiiBeetles17",
            # "ORBITBees-01-12-13-14",
            # "ORBITBees02-Marija",
            "Coleoptera",
            "Hymenoptera",
            "Lepidoptera"
          ], 
          task = "detect",
          erase=False)