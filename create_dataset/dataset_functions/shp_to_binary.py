import os
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio import features
from shapely.geometry import box
from PIL import Image
from tqdm import tqdm


# Define the paths
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ORIGIN_IMG_PATH = os.path.join(ROOT_PATH, "../../dataset/images")
OUT_TIF_PATH = os.path.join(ROOT_PATH, "../binary_mask")
OUT_PNG_PATH = os.path.join(ROOT_PATH, "../../dataset/masks/shape")


def shp_plot(shapefile_path):
    """
    Draws the polygons of the shapes from a *.shp..

    Args:
        shapefile_path: the path of the shapefile to plot
    """
    
    # Read shapefile
    shp_file_path = shapefile_path
    gdf = gpd.read_file(shp_file_path)

    # Prepare the plot
    fig, ax = plt.subplots()

    # Iterate over the geometries and plot them
    for idx, geom in gdf.geometry.items():
        if geom.geom_type == 'Polygon':
            # If it's a polygon, plot the exterior
            x, y = geom.exterior.xy
            ax.plot(x, y, color=plt.cm.viridis(idx / len(gdf)))

    plt.title("Polygons from the shapefile")
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Index')
    plt.show()







def convert_tif_to_png(file_path, output_path, file_name = -1):
    """
    Convert a TIF file to a PNG file
    
    Args:
        file_path: The path of the TIF file to convert.
        output_path: The output folder path for the PNG files.
        file_name: The output PNG file name. If not specified, an automatic name will be assigned.
    """

    with rasterio.open(file_path) as src:
        # Read the raster data
        data = src.read()

        # Verify if the image is RGB
        is_rgb = data.shape[0] == 3  # If the raster has 3 bands, it is RGB

        # Normalizza i valori del raster
        max_value = np.max(data)
        normalized_data = (data / max_value * 255).astype(np.uint8)

        # Crea l'immagine PIL
        if is_rgb:
            # Se è un'immagine RGB, trasponi gli assi per ottenere l'ordine corretto
            tif_image = Image.fromarray(normalized_data.transpose(1, 2, 0), 'RGB')
        else:
            # Altrimenti, crea un'immagine in scala di grigi
            tif_image = Image.fromarray(normalized_data[0])

        # Conta solo i file nella directory
        if file_name == -1:
            file_name = len(os.listdir(output_path))

        # Crea il percorso per il file PNG di output
        output_file_path = os.path.join(output_path, f"{file_name}.png")

        # Salva l'immagine come PNG
        tif_image.save(output_file_path)

def tif_to_png(input_path, output_path):
    """
    Convert a TIF file or all TIF files in a folder to PNG files.

    Args:
        input_path: The path of the TIF file or folder to convert.
        output_path: The output folder path for the PNG files.
    """

    # Crea la cartella se non esiste
    os.makedirs(f"{output_path}", exist_ok = True)

    if os.path.isfile(input_path): # Se è un file, lo converte
        convert_tif_to_png(input_path, output_path)
    elif os.path.isdir(input_path): # Se è una cartella, converte tutti i file TIF al suo interno
        bar = tqdm(total = len(os.listdir(input_path)), desc = "Converting .tif to .png", position = 1, leave = False)
        idx = 0
        for filename in os.listdir(input_path):
            bar.update(1)
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                convert_tif_to_png(os.path.join(input_path, filename), output_path, file_name = idx)
                idx += 1
    else:
        raise ValueError(f"Path not valid: {input_path}")


def shp_to_bm(tif_file_path, shp_file_path, output_folder):
    """
    Create binary masks from a TIF file (georeferenced) and a shapefile.
    
    Args:
        tif_file_path: The path of the TIF file containing the georeferenced shapes.
        shp_file_path: The path of the shapefile containing the shapes.
        output_folder: The output folder path for the binary masks.
    """

    # Read the shapefile
    gdf = gpd.read_file(shp_file_path)

    # Create the output folder if it does not exist
    os.makedirs(f"{output_folder}", exist_ok = True)

    # Use rasterio to read the TIF file
    with rasterio.open(tif_file_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        data = src.read(1)  # Read the first band

    bar = tqdm(total = len(gdf), desc = "Creating binary tif masks..", position = 1, leave = False)
    minx, miny, maxx, maxy = src.bounds
    tif_bounds = box(minx, miny, maxx, maxy)

    # Create a binary mask for each shape
    for idx, geom in gdf.geometry.items():
        bar.update(1)
        try:
            if geom.geom_type == 'Polygon':
                # Check if the shape intersects the TIF bounds
                #if geom.intersects(tif_bounds):
                    # Create a mask with the same shape as the TIF data
                    mask = np.zeros_like(data, dtype=np.uint8)

                    # Create a shape for the form
                    shapes = ((geom, 1),)

                    # Create the mask
                    mask = features.rasterize(
                        shapes = shapes,
                        out = mask,
                        fill = 1,
                        transform = transform,
                        all_touched = True,
                        default_value = 0
                    )

                    # Set the output path for the binary mask
                    output_tif_path = f"{output_folder}/output_mask_{idx}.tif"

                    # Write the mask to a new TIF file
                    with rasterio.open(output_tif_path, 'w', **profile) as dst:
                        dst.write(mask, 1)
        except AttributeError:
            continue


def crete_binary_mask(tif_file_path, shp_file_path):
    # Find the name of the image
    image_name = len(os.listdir(ORIGIN_IMG_PATH))
    out_tif = f"{OUT_TIF_PATH}/{image_name}"
    out_png = f"{OUT_PNG_PATH}/{image_name}"

    # Convert the TIF file to a PNG file
    tif_to_png(tif_file_path, ORIGIN_IMG_PATH)

    # Create the binary mask from the TIF file and the shapefile
    shp_to_bm(tif_file_path, shp_file_path, out_tif)

    # Convert the binary mask to a PNG file
    tif_to_png(out_tif, out_png)