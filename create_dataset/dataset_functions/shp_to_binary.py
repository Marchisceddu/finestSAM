import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from PIL import Image
from shapely.geometry import box

# Definizione dei percorsi
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGIN_IMG_PATH = os.path.join(ROOT_PATH, "../../dataset/images")
OUT_TIF_PATH = os.path.join(ROOT_PATH, "../create_dataset/binary_mask/masks_tif")
OUT_PNG_PATH = os.path.join(ROOT_PATH, "../../dataset/masks/shape")

def shp_plot(shapefile_path):
    """
    Traccia i poligoni contenuti in un file shapefile.

    Args:
        shapefile_path: Il percorso del file shapefile da tracciare.
    """
    
    # Leggi il file shapefile
    shp_file_path = shapefile_path
    gdf = gpd.read_file(shp_file_path)

    # Prepara il tracciamento
    fig, ax = plt.subplots()

    # Itera su ogni poligono nel GeoDataFrame
    for idx, geom in gdf.geometry.items():
        if geom.geom_type == 'Polygon':
            # Se è un singolo poligono, estrai le coordinate esterne e tracciale
            x, y = geom.exterior.xy
            ax.plot(x, y, color=plt.cm.viridis(idx / len(gdf)))

    # Aggiungi legenda e titolo
    plt.title("Poligoni dal file shapefile")
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Indice')

    # Mostra il plot
    plt.show()

def convert_tif_to_png(file_path, output_path, file_name = -1):
    """
    Converte un file TIF in PNG.

    Args:
        file_path: Il percorso del file TIF da convertire.
        output_path: Il percorso della cartella di output per i file PNG.
        file_name: Il nome del file PNG di output. Se non specificato, verrà assegnato un nome automatico.
    """

    with rasterio.open(file_path) as src:
        # Leggi il raster
        data = src.read(1)

        # Controlla se il raster contiene valori nulli
        max_value = 1 if (np.max(data) or np.any(np.isnan(data)) == 0) else np.max(data)
        # if np.any(np.isnan(data)):
        #     # Se sono presenti valori nulli, imposta il valore massimo a 1
        #     max_value = 1
        # else:
        #     # Se non ci sono valori nulli, usa il valore massimo originale
        #     max_value = 1 if np.max(data) == 0 else np.max(data)

        # Normalizza i valori del raster
        normalized_data = (data / max_value * 255).astype(np.uint8)

        # Crea l'immagine PIL
        tif_image = Image.fromarray(normalized_data)

        # Conta solo i file nella directory
        if file_name == -1:
            file_name = len(os.listdir(output_path))
        
        # Crea il percorso per il file PNG di output
        output_file_path = os.path.join(output_path, f"{file_name}.png")

        # Salva l'immagine come PNG
        tif_image.save(output_file_path)

def tif_to_png(input_path, output_path):
    """
    Converte un file o una cartella TIF in PNG.

    Args:
        input_path: Il percorso del file o della cartella TIF da convertire.
        output_path: Il percorso della cartella di output per i file PNG.
    """

    # Crea la cartella se non esiste
    os.makedirs(f"{output_path}", exist_ok = True)

    if os.path.isfile(input_path): # Se è un file, lo converte
        convert_tif_to_png(input_path, output_path)
    elif os.path.isdir(input_path): # Se è una cartella, converte tutti i file TIF al suo interno
        bar = tqdm(total = len(os.listdir(input_path)), desc = "Conversione da .tif a .png", position = 2, leave = False)
        idx = 0
        for filename in os.listdir(input_path):
            bar.update(1)
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                convert_tif_to_png(os.path.join(input_path, filename), output_path, file_name = idx)
                idx += 1
    else:
        raise ValueError(f"Percorso non valido: {input_path}")

def shp_to_bm(tif_file_path, shp_file_path, output_folder):
    """
    Crea una maschera binaria per ogni forma georeferenziata in un file shapefile.

    Args:
        tif_file_path: Il percorso del file TIF da cui creare le maschere binarie.
        shp_file_path: Il percorso del file shapefile contenente le forme georeferenziate.
        output_folder: La cartella di output per i file TIFF di maschera binaria.
    """

    # Leggi il file shapefile
    gdf = gpd.read_file(shp_file_path)

    # Carica il file TIFF
    with rasterio.open(tif_file_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        data = src.read(1)  # Leggi il raster

    bar = tqdm(total = len(gdf), desc = "Creazione maschere binarie tif", position = 2, leave = False)
    minx, miny, maxx, maxy = src.bounds
    tif_bounds = box(minx, miny, maxx, maxy)

    # Creare una maschera binaria per ogni forma georeferenziata
    for idx, geom in gdf.geometry.items():
        bar.update(1)
        try:
            if geom.geom_type == 'Polygon':
                # Controllo se la geometria si sovrappone all'estensione dell'immagine
                if geom.intersects(tif_bounds):
                    # Creare una matrice vuota della stessa forma del raster
                    mask = np.zeros_like(data, dtype=np.uint8)

                    # Genera una forma (geometria) per la singola forma
                    shapes = ((geom, 1),)

                    # Creare una maschera per la forma
                    mask = features.rasterize(
                        shapes = shapes,
                        out = mask,
                        fill = 1,
                        transform = transform,
                        all_touched = True,
                        default_value = 0
                    )

                    # Crea la cartella se non esiste
                    os.makedirs(f"{output_folder}", exist_ok = True)

                    # Imposta il percorso per il file TIFF di output
                    output_tif_path = f"{output_folder}/output_mask_{idx}.tif"

                    # Scrivi la maschera nel file TIFF
                    with rasterio.open(output_tif_path, 'w', **profile) as dst:
                        dst.write(mask, 1)
        except AttributeError:
            continue

def crete_binary_mask(tif_file_path, shp_file_path):
    # Trova il nome dell'immagine a cui si riferisce
    image_name = len(os.listdir(ORIGIN_IMG_PATH))
    out_tif = f"{OUT_TIF_PATH}/{image_name}"
    out_png = f"{OUT_PNG_PATH}/{image_name}"

    # Converte tutti il file tif selezionato in file png
    tif_to_png(tif_file_path, ORIGIN_IMG_PATH)

    # Crea una maschera binaria per ogni forma georeferenziata
    shp_to_bm(tif_file_path, shp_file_path, out_tif)

    # Converte tutti i file tif in file png
    tif_to_png(out_tif, out_png)

    print("Maschera binaria completata")