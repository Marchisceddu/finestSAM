import os
import argparse
from tqdm import tqdm
from dataset_functions import (
    get_folder_path,
    crete_binary_mask, 
    create_annotation_COCO,
    display_COCO
)


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(ROOT_PATH, "shp/")
DATASET_PATH = os.path.join(ROOT_PATH, "../dataset/")
IMAGES_PATH = os.path.join(ROOT_PATH, "../dataset/images/")
MASKS_PATH = os.path.join(ROOT_PATH, "../dataset/masks/")
MASKS_TIF_PATH = os.path.join(ROOT_PATH, "binary_mask/")
JSON_PATH = os.path.join(ROOT_PATH, "../dataset/annotations.json")


def create_dataset(scegli_input = False, mostra_output = False):
    '''
    Create a dataset from SHP and TIF files in COCO format

    Args:
        scegli_input (bool): If True, allows you to choose the input folder 
                            (must be formatted like the INPUT_PATH folder) 
                         
        mostra_output (bool): If True, shows the output of the dataset
    '''

    if (scegli_input):
        print("Seleziona la cartella di input:")
        input_path = get_folder_path()
    else:
        input_path = INPUT_PATH

    # Creating a progress bar
    num_folders = sum(1 for item in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, item)))
    bar_folder = tqdm(total = num_folders, desc = "Processo immagini", position = 0, leave = False)

    # Delete the dataset folder if it already exists
    if os.path.exists(DATASET_PATH):
        os.system(f"rm -r {DATASET_PATH}")
    os.makedirs(DATASET_PATH, exist_ok = True)
    os.makedirs(IMAGES_PATH, exist_ok = True)
    os.makedirs(MASKS_PATH, exist_ok = True)

    for folder in os.listdir(input_path):
        shp_file_path = ""
        tif_file_path = ""

        # Check if the folder is a directory
        if (os.path.isdir(os.path.join(input_path, folder))):
            # Iterate over the files to get the shp
            for file in os.listdir(os.path.join(input_path, folder)):
                if (file.endswith(".shp")):
                    shp_file_path = os.path.join(input_path, folder, file)
                    break
            
            if (shp_file_path != ""):
                # iterate over the files to get the tif
                for file in os.listdir(os.path.join(input_path, folder)):
                    if (file.endswith(".tif") or file.endswith(".tiff")):
                        tif_file_path = os.path.join(input_path, folder, file)
                        crete_binary_mask(tif_file_path, shp_file_path)

        bar_folder.update(1)

    # Delete the binary mask folder
    os.system(f"rm -r {MASKS_TIF_PATH}")

    # Create the COCO annotation file
    create_annotation_COCO(MASKS_PATH, JSON_PATH)

    if (mostra_output): 
        display_COCO("dataset/images/", "dataset/annotations.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crea un dataset a partire da file SHP e file TIF in formato COCO")
    parser.add_argument("--scegli_input", action="store_true", help="Se impostato a True, permette di scegliere la cartella di input", default=False)
    parser.add_argument("--mostra_output", action="store_true", help="Se impostato a True, mostra l'output del dataset", default=True)
    args = parser.parse_args()

    create_dataset(scegli_input=args.scegli_input, mostra_output=args.mostra_output)

# POSSIBILE MODIFICA:
# passare il nome della cartella da dare al dataset in input (potendo aggiungere così un potenziale dataset_val)