# restore corrupted dataset with loading and saving using PIL
from PIL import Image
from pathlib import Path
import tqdm

## vitesse : 1400 images en 1 min 50

img_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

def restore_dataset(dataset_folder):
    for img_file in tqdm.tqdm(list(Path(dataset_folder).rglob("*.*"))):
        if img_file.suffix.lower() in img_ext:
            try:
                img = Image.open(img_file)
                img.save(Path(img_file))  # Save the image to the final folder
            except (IOError, SyntaxError) as e:
                print(f"Warning: {img_file} is corrupted and will be skipped. Error: {e}")

if __name__ == "__main__":
    restore_dataset("./models/datasets/AllSpecies-posecls")