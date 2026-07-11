from convert_coco_to_yolo import convert_coco
from convert_json_to_coco import convert
import os
import time

GROUPS = ["Coleoptera", "Diptera", "Hymenoptera", "Lepidoptera"]
IMAGE_DIR = "C:/Users/tombe/Documents/_MLE/CV-for-GRIT/databases/full databases/diptera/"

if __name__ == "__main__":
    annotation_dir  = "./label-studio-annotations/"
    processed_dir = "./annotations/coco-converted/"
    converted_dir = "./annotations/convert-done/"

    for group_name in os.listdir(annotation_dir):
        group_dir = os.path.join(annotation_dir, group_name)
        if os.path.isdir(group_dir):
            for filename in os.listdir(group_dir):
                if filename.endswith(".json"):
                    input_path = os.path.join(group_dir, filename)
                    output_path = os.path.join(processed_dir, filename)
                    convert(input_path, output_path)
                    os.rename(input_path, os.path.join(converted_dir, filename))
            time.sleep(0.5)

            for filename in os.listdir(processed_dir):
                if filename.endswith(".json"):
                    input_path = os.path.join(processed_dir, filename)
                    output_path = os.path.join(processed_dir, filename.replace(".json", ".txt"))
                    convert_coco(
                        labels_dir="./annotations/coco-converted/", 
                        image_dir = IMAGE_DIR, # or "ask"
                        save_dir = "./models/datasets/", 
                        dataset_name = group_name,
                        filter_keywords = [], 
                        use_keypoints=True
                    )
    
        