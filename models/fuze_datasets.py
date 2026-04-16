from pathlib import Path
import shutil
from ultralytics.utils.files import increment_path
import tqdm

def fuze(dataset_name, dataset_list, dataset_folder = "./models/datasets/", erase=False):
    f = Path(dataset_folder) / dataset_name
    f = increment_path(f)

    (f / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (f / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (f / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (f / "images" / "train").mkdir(parents=True, exist_ok=True)
    (f / "images" / "val").mkdir(parents=True, exist_ok=True)
    (f / "images" / "test").mkdir(parents=True, exist_ok=True)

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
                    shutil.copy(label_file, f / "labels" / split / label_file.name)
            for image_file in tqdm.tqdm((dataset_path / "images" / split).glob("*.*"), colour="blue"):
                if image_file in all_images_files:
                    print(f"Warning: image file {image_file} already exists")
                else:
                    all_images_files.append(image_file)
                    shutil.copy(image_file, f / "images" / split / image_file.name)
                    
    if erase and input("Are you sure you want to delete the previous files? (y/n) ") == "y":
        print("Deleting previous files")
        for dataset in dataset_list:
            shutil.rmtree(Path(dataset_folder) / dataset)

if __name__ == "__main__":
    fuze("fuzed-01-07-08-16-17-noleg-noHwing", 
         [
            "HawaiiBeetles17",
            "HawaiiBeetles16",
            "HawaiiBeetles07Marija-noleg-noHwing-Corrected",
            "fuzed-01-08-noleg-noHwing"
          ], 
          erase=True)