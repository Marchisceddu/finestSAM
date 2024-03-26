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
    Crea un dataset a partire da file SHP e file TIF in formato COCO

    Args:
        scegli_input (bool): Se True, permette di scegliere la cartella di input 
                             (deve essere formattata come la cartella INPUT_PATH)
        mostra_output (bool): Se True, mostra l'output del dataset
    '''

    if (scegli_input):
        print("Seleziona la cartella di input")
        input_path = get_folder_path()
    else:
        input_path = INPUT_PATH

    # Creazione barra di caricamento che conta il numero di immagini da processare
    num_folders = sum(1 for item in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, item)))
    bar_folder = tqdm(total = num_folders, desc = "Processo immagini", position = 0, leave = False)

    # Elimina il vecchio dataset e crea delle nuove cartelle
    if os.path.exists(DATASET_PATH):
        os.system(f"rm -r {DATASET_PATH}")
    os.makedirs(DATASET_PATH, exist_ok = True)
    os.makedirs(IMAGES_PATH, exist_ok = True)
    os.makedirs(MASKS_PATH, exist_ok = True)

    for folder in os.listdir(input_path):
        shp_file_path = ""
        tif_file_path = ""

        # Controllo se è una directory
        if (os.path.isdir(os.path.join(input_path, folder))):
            # Ciclo per i file per ottenere lo shape
            for file in os.listdir(os.path.join(input_path, folder)):
                if (file.endswith(".shp")):
                    shp_file_path = os.path.join(input_path, folder, file)
                    break
            
            if (shp_file_path != ""):
                # Ciclo per i file per ottenere i tif
                for file in os.listdir(os.path.join(input_path, folder)):
                    if (file.endswith(".tif") or file.endswith(".tiff")):
                        tif_file_path = os.path.join(input_path, folder, file)
                        crete_binary_mask(tif_file_path, shp_file_path)

        bar_folder.update(1)

    # Elimino la cartella delle maschere binarie tif
    os.system(f"rm -r {MASKS_TIF_PATH}")

    # Creazione delle delle annotazioni COCO
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