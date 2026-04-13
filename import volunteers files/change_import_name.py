from pathlib import Path
import os
import json

DOWNLOADED = Path("./import volunteers files/downloaded/")
EMPTY = Path("./import volunteers files/empty/")
MODIFIED = Path("./import volunteers files/modified/")

DOWNLOADED_DONE = Path("./import volunteers files/downloaded-done/")
EMPTY_DONE = Path("./import volunteers files/empty-done/")

# only one at a time !
def run_with_empty_file():
    downloaded_files = DOWNLOADED.glob("*.json")
    downloaded_file = list(downloaded_files)[0]
    with open(downloaded_file, "r") as f:
        downloaded_data = json.load(f)

    empty_files  = EMPTY.glob("*.json")
    empty_file = list(empty_files)[0]
    with open(empty_file, "r") as f:
        empty_data = json.load(f)

    all_empty_images_names = {"".join(x["file_upload"].split("-")[1:]) : x["file_upload"].split("-")[0] for x in empty_data}
    all_empty_images_path = {"".join(x["file_upload"].split("-")[1:]) : str(Path(x["data"]["img"]).parent) for x in empty_data}

    for img_idx in range(len(downloaded_data)):
        img = downloaded_data[img_idx]
        base = "".join(img["file_upload"].split("-")[1:])

        if base in all_empty_images_names:
            downloaded_data[img_idx]["file_upload"] = f"{all_empty_images_names[base]}-{base}"
            downloaded_data[img_idx]["data"]["img"] = f"{all_empty_images_path[base]}/{all_empty_images_names[base]}-{base}"
        else:
            print(f"Image {base} not found in empty export")
    
    with open(MODIFIED / empty_file.name, "w") as f:
        json.dump(downloaded_data, f, indent=2)
    
    os.rename(downloaded_file, DOWNLOADED_DONE / downloaded_file.name)
    os.rename(empty_file, EMPTY_DONE / empty_file.name)

def run_with_local_files(folder_path):

    if not os.path.isdir(folder_path):
        print(f"Error: input directory '{folder_path}' not found.")
        return

    formatted_folder_path = "/data/local-files/?d=" + "%5C".join(str(Path(folder_path).resolve()).split("\\")[1:]) + "%5C"

    downloaded_files = DOWNLOADED.glob("*.json")
    downloaded_file = list(downloaded_files)[0]
    with open(downloaded_file, "r") as f:
        downloaded_data = json.load(f)

    for img_idx in range(len(downloaded_data)):
        img = downloaded_data[img_idx]
        base = "".join(img["data"]["img"].split("/")[-1].split("-")[-1])

        downloaded_data[img_idx]["data"]["img"] = f"{formatted_folder_path}{base}"
    
    with open(MODIFIED / downloaded_file.name, "w") as f:
        json.dump(downloaded_data, f, indent=2)
    
    os.rename(downloaded_file, DOWNLOADED_DONE / downloaded_file.name)
    
    #"/data/local-files/?d=Users%5Ctombe%5CDocuments%5C_MLE%5CCV-for-GRIT%5Cdatabases%5Chawaii_beetles_images%5Cindividual_specimens%5C01%5CIMG_0093_specimen_1_MECKON_NEON.BET.D20.000001.png"

if __name__ == "__main__":
    IMAGE_REPO = "C:\\Users\\tombe\\Documents\\_MLE\\CV-for-GRIT\\databases\\hawaii_beetles_images\\individual_specimens\\01"
    run_with_local_files(IMAGE_REPO)
    # run_with_local_files("./databases/hawaii_beetles_images/individual_specimens/01/")