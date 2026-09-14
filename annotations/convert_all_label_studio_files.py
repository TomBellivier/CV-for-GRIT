from convert_coco_to_yolo import convert_coco
from convert_json_to_coco import convert
import os
import time
from pathlib import Path

GROUPS = ["Coleoptera", "Diptera", "Hymenoptera", "Lepidoptera"]
IMAGE_DIR = Path("/home/tombellivier/Documents/CV/CV-for-GRIT/databases/full databases")

if __name__ == "__main__":
    annotation_dir  = "./label_studio_annotations/"
    processed_dir = "./annotations/coco-converted/"
    converted_dir = "./annotations/convert-done/"

    for group_name in os.listdir(annotation_dir):
        group_dir = os.path.join(annotation_dir, group_name)
        if os.path.isdir(group_dir):
            print(f"Processing group: {group_name} ({len(os.listdir(group_dir))} files)")
            for filename in os.listdir(group_dir):
                if filename.endswith(".json"):
                    input_path = os.path.join(group_dir, filename)
                    output_path = os.path.join(processed_dir, filename)
                    convert(input_path, output_path)
            print(f"Converted all Label Studio files for group: {group_name}")
            time.sleep(0.5)

            for filename in os.listdir(processed_dir):
                if filename.endswith(".json"):
                    input_path = os.path.join(processed_dir, filename)
                    output_path = os.path.join(processed_dir, filename.replace(".json", ".txt"))
                    print(IMAGE_DIR / group_name)
                    convert_coco(
                        labels_dir="./annotations/coco-converted/", 
                        image_dir = IMAGE_DIR / group_name, # or "ask"
                        save_dir = "./models/datasets/", 
                        dataset_name = group_name,
                        filter_keywords = [], 
                        use_keypoints=True
                    )
            time.sleep(2)