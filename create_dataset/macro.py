import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) # Path to the root folder (create_dataset)
INPUT_PATH = os.path.join(ROOT_PATH, "shp") # Path to the SHP files

OUTPUT_PATH = os.path.join(ROOT_PATH, "../dataset") # Path to the dataset folder
OUTPUT_IMAGES_PATH = os.path.join(OUTPUT_PATH, "images") # Path to the images folder
OUTPUT_MASKS_PATH = os.path.join(OUTPUT_PATH, "masks") # Path to the masks folder
OUTPUT_ANNOTATIONS_PATH = os.path.join(OUTPUT_PATH, "annotations.json") # Path to the annotations file