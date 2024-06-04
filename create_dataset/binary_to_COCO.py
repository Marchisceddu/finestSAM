"""
This code automates the conversion of binary masks representing different 
object categories into the COCO (Common Objects in Context) JSON format. 

The code is based on the following folder structure for training and validation
images and masks. You need to change the code based on your folder structure 
or organize your data to the format below.

dataset/            #Primary data folder for the project
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── masks/        #All binary masks organized in respective sub-directories.
│   ├── shape/          # Sub-directory for each class
│   │   ├── 0/          # Sub-directory for each image
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── 1/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   └── 2/
│   │       ├── 0.png
│   │       ├── 1.png
│   │       └── ...
│   ├── next_class/     # NON IMPLEMENTATA ANCORA, MA POSSIBILE AGGIUNGERE ALTRE CLASSI
│   └── ...
└── ...


For each binary mask, the code extracts contours using OpenCV. 
These contours represent the boundaries of objects within the images.This is a key
step in converting binary masks to polygon-like annotations. 

Convert the contours into annotations, including 
bounding boxes, area, and segmentation information. Each annotation is 
associated with an image ID, category ID, and other properties required by the COCO format.

The code also creates an images section containing 
metadata about the images, such as their filenames, widths, and heights.
In my example, I have used exactly the same file names for all images and masks
so that a given mask can be easily mapped to the image. 

All the annotations, images, and categories are 
assembled into a dictionary that follows the COCO JSON format. 
This includes sections for "info," "licenses," "images," "categories," and "annotations."

Finally, the assembled COCO JSON data is saved to a file, 
making it ready to be used with tools and frameworks that support the COCO data format.
"""

import glob
import json
import os
import cv2
import tqdm


# Label IDs of the dataset representing different categories
category_ids = {
    "shape": 1,
}

MASK_EXT = 'png'
ORIGINAL_EXT = 'png'
image_id = 0
annotation_id = 0


def images_annotations_info(maskpath):
    """
    Process the binary masks and generate images and annotations information.

    :param maskpath: Path to the directory containing binary masks
    :return: Tuple containing images info, annotations info, and annotation count
    """

    global image_id, annotation_id
    annotations = []
    images = []

    # Iterate through categories and corresponding masks
    for category in category_ids.keys():
        # Costruisci il percorso completo per la categoria corrente all'interno di maskpath
        category_path = os.path.join(maskpath, category)
        
        # Ottieni una lista dei nomi delle cartelle all'interno del percorso della categoria corrente
        category_folders = [folder for folder in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, folder))]

        bar = tqdm.tqdm(total = len(category_folders), desc = f"Processing {category} masks", leave=False)

        for image_folder in category_folders:
            original_file_name = f'{os.path.basename(image_folder)}.{ORIGINAL_EXT}'

            # Definire una lista per contenere tutti i contorni
            all_contours = []
            height, width = 0, 0

            for mask_image in glob.glob(os.path.join(category_path, image_folder, f'*.{MASK_EXT}')):

                mask_image_open = cv2.imread(mask_image)
                
                # Get image dimensions
                height, width, _ = mask_image_open.shape

                # Find contours in the mask image
                gray = cv2.cvtColor(mask_image_open, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

                # Aggiungere i contorni trovati alla lista complessiva dei contorni
                all_contours.extend(contours)

            # Create or find existing image annotation
            if original_file_name not in map(lambda img: img['file_name'], images):
                image = {
                    "id": image_id + 1,
                    "width": width,
                    "height": height,
                    "file_name": original_file_name,
                }
                images.append(image)
                image_id += 1
            else:
                image = [element for element in images if element['file_name'] == original_file_name][0]

            # Create annotation for each contour
            for contour in all_contours:
                bbox = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                segmentation = contour.flatten().tolist()

                annotation = {
                    "iscrowd": 0,
                    "id": annotation_id,
                    "image_id": image['id'],
                    "category_id": category_ids[category],
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [segmentation],
                }

                # Add annotation if area is greater than zero
                if area > 0:
                    annotations.append(annotation)
                    annotation_id += 1

            bar.update(1)
            
    return images, annotations, annotation_id


def create_annotation_COCO(mask_path, dest_json):
    global image_id, annotation_id
    image_id = 0
    annotation_id = 0

    # Initialize the COCO JSON format with categories
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()],
        "annotations": [],
    }

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    # Create the destination directory if it does not exist
    os.makedirs(os.path.dirname(dest_json), exist_ok=True)

    # Save the COCO JSON to a file
    with open(dest_json, "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

    print(f"Created {annotation_cnt} annotations for images in folder: {mask_path}")

if __name__ == "__main__":
    # esempio di utilizzo
    # il percorso deve partire dalla cartella dove si trova il file .py
    train_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset/masks/")
    train_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset/annotations.json")
    create_annotation_COCO(train_mask_path, train_json_path)