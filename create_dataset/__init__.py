import os
from tqdm import tqdm
from dataset_functions import crete_binary_mask, create_annotation_COCO, get_shp_file_path, get_tif_file_path, display_image_with_annotations_COCO

MODALITA = 1 
INPUT_PATH = "./create_dataset/shp"
MASKS_PATH = "./dataset/coco/masks/"
JSON_PATH = "./dataset/coco/annotations.json"

if __name__ == "__main__":
    
    # Creazione maschere binarie
    # MODALITA = 0 -> singolo file
    # MODALITA = 1 -> cartelle
    if (MODALITA == 0):
        print("Seleziona un file SHP")
        shp_file_path = get_shp_file_path()

        print("Seleziona un file TIF")
        tif_file_path = get_tif_file_path()

        crete_binary_mask(tif_file_path, shp_file_path)
    else:
        bar_folder = tqdm(total = len(os.listdir(INPUT_PATH)), desc = "Cartelle Paesi", position = 0, leave = False)

        for folder in os.listdir(INPUT_PATH):
            shp_file_path = ""
            tif_file_path = ""

            # Controllo se è una directory
            if (os.path.isdir(os.path.join(INPUT_PATH, folder))):
                # Ciclo per i file
                for file in os.listdir(os.path.join(INPUT_PATH, folder)):
                    if (file.endswith(".shp")):
                        shp_file_path = os.path.join(INPUT_PATH, folder, file)
                        break
                
                if (shp_file_path != ""):
                    bar_file = tqdm(total = len(os.listdir(os.path.join(INPUT_PATH, folder))), desc = "Iterazione tif all' interno di {folder}", position = 1, leave = False)

                    for file in os.listdir(os.path.join(INPUT_PATH, folder)):
                        if (file.endswith(".tif") or file.endswith(".tiff")):
                            tif_file_path = os.path.join(INPUT_PATH, folder, file)
                            crete_binary_mask(tif_file_path, shp_file_path)

                        bar_file.update(1)

                bar_folder.update(1)

    # Creazione delle delle annotazioni COCO
    create_annotation_COCO(MASKS_PATH, JSON_PATH)
    display_image_with_annotations_COCO("./dataset/coco/images", "./dataset/coco/annotations.json")
    
                