## create classification dataset with background class by removing the detection from each images

from pathlib import Path
from PIL import Image, ImageFilter
import shutil
import tqdm
import numpy as np

dataset_dir = "./models/datasets/"
dataset_path = Path(dataset_dir)
pose_dataset_path = dataset_path / "AllSpecies-pose"
cls_dataset_path = dataset_path / "AllSpecies-cls"

Path("C:/Users/tombe/Documents/_MLE/CV-for-GRIT/models/datasets/AllSpecies-cls/train/background").mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    all_images = list((pose_dataset_path/"images"/split).glob("*.png")) + \
        list((pose_dataset_path/"images"/split).glob("*.jpg"))

    print((pose_dataset_path/"images"/split).resolve())
    print(len(all_images))

    background_cls_path = cls_dataset_path / split / "background"
    print(background_cls_path.resolve())
    background_cls_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm.tqdm(all_images):
        # get label path in the label folder
        img_name = img_path.stem
        label_path = pose_dataset_path / "labels" / split / f"{img_name}.txt"

        img = Image.open(img_path)

        # get detection cordinates from the label file
        with open(label_path, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                # if there is no detection, copy the image to the new location
                shutil.copy(img_path, background_cls_path / img_path.name)
                continue
        
        x, y, w, h = map(float, lines[0].split()[1:5])
        img_width, img_height = img.size

        # convert the cordinates to pixel values
        x1 = int((x - w / 2) * img_width)
        y1 = int((y - h / 2) * img_height)
        x2 = int((x + w / 2) * img_width)
        y2 = int((y + h / 2) * img_height)

        # draw a square with mean color around the detection area
        borders = (max(x1-1, 0), max(y1-1, 0), min(x2+1, img_width-1), min(y2+1, img_height-1))
        img_array = np.asarray(img)
        mean_colors  = [np.mean(img_array[borders[1], borders[0]:borders[2]], axis=(0)), 
                        np.mean(img_array[borders[1]:borders[3], borders[0]], axis=(0)), 
                        np.mean(img_array[borders[3], borders[0]:borders[2]], axis=(0)), 
                        np.mean(img_array[borders[1]:borders[3], borders[2]], axis=(0))]
        mean_color = np.mean(mean_colors, axis=0).astype(np.uint8)
        img.paste(tuple(mean_color), (x1, y1, x2, y2))

        # add noise in the detection area
        noise = (np.random.rand(y2 - y1, x2 - x1, 3) * (mean_color * 0.1) + mean_color + 0.9).astype(np.uint8)
        img.paste(Image.fromarray(noise), (x1, y1, x2, y2))

        # add a gaussian blur to the edges of the detection area
        region = img.crop((max(x1 - 20, 0), max(y1 - 20, 0), min(x2 + 20, img_width-1), min(y2 + 20, img_height-1)))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=20))
        img.paste(blurred_region, (max(x1 - 20, 0), max(y1 - 20, 0), min(x2 + 20, img_width-1), min(y2 + 20, img_height-1)))

        img.save(background_cls_path / img_path.name)
