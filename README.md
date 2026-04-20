# CV-for-GRIT
Implementation of Computer Vision model to measure insect traits from images. This would be used to help compiling the Global Repository of Insect Traits. 

# How to use 

## Split databases
1. In a terminal, run the command "python ./databases/split_image_database.py ./databases/[image folder name] [number of subfolder to create]".
2. You can add "--reverse" to undo the previous action. In this case, "number of subfolder to create" is not required.

## Check annotations 
1. Put the volunteers file in "import volunteers files/downloaded", and make sure there is only one file 
2. Open "import volunteers files/change_import_name.py" and change the value of "IMAGE_REPO". Run the code.
Checking : 
1. On a terminal, type the command "start_label_studio". 
2. On Label Studio project, add a storage source (local) at the image directory specified just before. 
3. Import the modified file in Label studio, it should find every images.

## Convert annotations JSON into YOLO format
1. Put the JSON file in "annotations/to convert". Do only one at a time.
2. Run "annotations/convert_json_to_coco.py" file. 
3. Open "annotations/convert_coco_to_yolo.py" and change the IMAGE_DIR value. Run the file.
4. A new dataset should have been created in "models/datasets/".
