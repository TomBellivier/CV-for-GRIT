# list and show informations on every dataset found

import os
from pathlib import Path


dataset_dir = "./models/datasets/"

for dataset_name in os.listdir(dataset_dir):
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isdir(dataset_path):
        print(f"Dataset: {dataset_name}")
        num_images = len(list(Path(dataset_path).rglob("./**/*.jpg"))) + len(list(Path(dataset_path).rglob("./**/*.png")))
        print(f"  Number of images: {num_images}")
        num_labels = len(list(Path(dataset_path).rglob("./**/*.txt")))
        print(f"  Number of label files: {num_labels}")
        print("")