import os
import tqdm
from shp_to_binary import create_binary_mask
from binary_to_COCO import create_annotation_COCO
from macro import INPUT_PATH, OUTPUT_PATH

def create_dataset():
    '''
    Create a dataset from SHP and TIF files in COCO format

    '''

    # Delete the dataset folder if it already exists
    if os.path.exists(OUTPUT_PATH):
        os.system(f"rm -r {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok = True)
    os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok = True)
    os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok = True)

    bar = tqdm.tqdm(total = len(os.listdir(INPUT_PATH)), desc = "Creating binary masks", leave=False)

    # Iterate over the folders in the input path
    for folder_name in os.listdir(INPUT_PATH):
        create_binary_mask(os.path.join(INPUT_PATH, folder_name))
        bar.update(1)

    # Delete the temp_masks folder
    os.system(f"rm -r {os.path.join(OUTPUT_PATH, "temp_masks")}")

    # Create the COCO annotation file
    create_annotation_COCO(os.path.join(OUTPUT_PATH, "masks"), os.path.join(OUTPUT_PATH, "annotations.json"))

if __name__ == "__main__":
    create_dataset()

