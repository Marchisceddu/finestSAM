import os
import rasterio
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
from rasterio import features
from macro import OUTPUT_PATH


def tif_to_png(tif, output_folder):
    """
    Convert a TIF file or all TIF files in a folder to PNG files.

    Args:
        tif_folder: The path of the TIF file or folder to convert.
        output_folder: The output folder path for the PNG files.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok = True)

    if os.path.isfile(tif):
        list_file = [tif]
        is_folder = False
    elif os.path.isdir(tif):
        list_file = [os.path.join(tif, file_name) for file_name in os.listdir(tif)]
        is_folder = True
    else:
        raise ValueError(f"Path not valid: {tif}")

    # Iterate over the TIF files
    for file_name, file_path in enumerate(list_file):
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
            if not is_folder:
                file_name = len(os.listdir(output_folder))

            # Salva l'immagine come PNG
            tif_image.save(os.path.join(output_folder, f"{file_name}.png"))

    if not is_folder:
        return file_name


def shp_to_bm(tif, shp, output_folder):
    """
    Create binary masks from a TIF files (georeferenced) and a shapefile.
    
    Args:
        tif: The path of the TIF file (georeferenced).
        shp: The path of the shapefile containing the shapes.
        output_folder: The output folder path for the binary masks.
    """
    # Read the shapefile
    gdf = gpd.read_file(shp)

    # Create the output folder if it does not exist
    os.makedirs(f"{output_folder}", exist_ok = True)

    # Use rasterio to read the TIF file
    with rasterio.open(tif) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs

        # Fix the CRS of the shapefile if it is different from the TIF file
        if(gdf.crs != crs):
            gdf = gdf.to_crs(crs)

        data = src.read(1)  # Read the first band

    minx, miny, maxx, maxy = src.bounds
    tif_bounds = box(minx, miny, maxx, maxy)

    # Create a binary mask for each shape
    for idx, geom in gdf.geometry.items():
                # Check if the shape intersects the TIF bounds
                if geom.intersects(tif_bounds):
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

                    #print(output_folder)
                    # Set the output path for the binary mask
                    output_tif_path = f"{output_folder}/output_mask_{idx}.tif"

                    # Write the mask to a new TIF file
                    with rasterio.open(output_tif_path, 'w', **profile) as dst:
                        dst.write(mask, 1)


def create_binary_mask(shp_folder):
        shp_file_path = ""

        # Check if the folder is a directory
        if (os.path.isdir(shp_folder)):
            # Iterate over the files to get the shp
            for file in os.listdir(shp_folder):

                if (file.endswith(".shp")):
                    shp_file_path = os.path.join(shp_folder, file)
                    break
            
            if (shp_file_path != ""):
                for file in os.listdir(shp_folder):
                    if (file.endswith(".tif") or file.endswith(".tiff")):
                        file_path = os.path.join(shp_folder, file)

                        # Convert the TIF files to PNG files and return the name of the tif file
                        tif_name = tif_to_png(file_path, os.path.join(OUTPUT_PATH, "images"))

                        # Create a temporary folders for each shape wich contains the binary mask   
                        shp_to_bm(file_path, shp_file_path, os.path.join(OUTPUT_PATH, "temp_masks", f"{tif_name}"))

                        # Convert the binary mask in tif format to PNG files
                        tif_to_png(os.path.join(OUTPUT_PATH, "temp_masks", f"{tif_name}"), os.path.join(OUTPUT_PATH, "masks", "shape", f"{tif_name}"))