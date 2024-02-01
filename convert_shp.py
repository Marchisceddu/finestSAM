import numpy as np
import geopandas as gpd
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from PIL import Image
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, MultiPolygon, unary_union

def shp_to_bm_plot(tif_path, shapefile_path):
    # Leggi il file TIFF georeferenziato
    ds = gdal.Open(tif_path)
    array = ds.ReadAsArray()

    # Trasponi l'array se necessario
    if array.shape[0] == 3:  # Se l'immagine ha tre canali (RGB)
        array = np.transpose(array, [1, 2, 0])  # Trasponi l'array

    # Estrai le informazioni sulla geotrasformazione
    geotransform = ds.GetGeoTransform()
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    # Leggi il file shapefile
    gdf = gpd.read_file(shapefile_path)

    # Creare il plot
    fig, ax = plt.subplots()

    # Visualizza l'immagine TIFF
    ax.imshow(array, extent=[origin_x, origin_x + ds.RasterXSize * pixel_width,
                            origin_y + ds.RasterYSize * pixel_height, origin_y])

    # Visualizza le shape sopra l'immagine
    gdf.plot(ax=ax, color='black')

    # Aggiungi la legenda
    legend_elements = [Patch(facecolor='red', edgecolor='red', label='Shapefile')]
    ax.legend(handles=legend_elements, loc='upper right')

    # Mostra il plot
    plt.show()

def shp_to_geojson (shapefile_path, output_path):
    # Carica la shapefile
    gdf = gpd.read_file(shapefile_path)

    # Salva il GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')

def tif_to_png(input_folder, output_folder):
    '''
    Converte tutti i file TIFF in una cartella in file PNG
    '''
    # Controlla se la cartella di output esiste, altrimenti creala
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Itera su tutti i file nella cartella di input
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            tif_path = os.path.join(input_folder, file_name)
            
            # Apre il file TIFF
            ds = gdal.Open(tif_path)
            array = ds.ReadAsArray()
            
            # Trasponi l'array se necessario
            if array.shape[0] == 3:  # Se l'immagine ha tre canali (RGB)
                array = np.transpose(array, [1, 2, 0])  # Trasponi l'array
            
            # Crea il percorso per il file PNG di output
            output_file_path = os.path.join(output_folder, file_name.replace('.tif', '.png'))
            
            # Imposta il comportamento degli avvisi
            np.seterr(divide='ignore', invalid='ignore')
                        
            # Salva l'immagine come PNG utilizzando PIL
            array_uint8 = (array / np.max(array) * 255).astype(np.uint8)
            tif_image = Image.fromarray(array_uint8)
            tif_image.save(output_file_path)

def generate_single_bm(raster_path, shape_path, output_path, file_name):
    """
    Function that generates a binary mask from a vector file (shp or geojson)
    
    raster_path = path to the .tif;
    shape_path = path to the shapefile or GeoJson.
    output_path = Path to save the binary mask.
    file_name = Name of the file.
    """
    
    # Load raster
    with rasterio.open(raster_path, "r") as src:
        raster_meta = src.meta
    
    # Load the shapefile or GeoJSON
    train_df = gpd.read_file(shape_path)
    
    # Verify crs
    if train_df.crs != src.crs:
        print("Raster crs: {}, Vector crs: {}.\nConvert vector and raster to the same CRS.".format(src.crs, train_df.crs))
        
    # Function to generate the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        if polygon.geom_type == 'Polygon':
            polygons = [polygon]
        elif polygon.geom_type == 'MultiPolygon':
            polygons = polygon.geoms

        for poly in polygons:
            for i in np.array(poly.exterior.coords):
                poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly

    
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for _, row in train_df.iterrows():
        if isinstance(row['geometry'], MultiPolygon):
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)

    # Create the unary union of the polygons with buffer(0)
    buffered_polygons = [poly.buffer(0) for poly in poly_shp]
    unified_polygons = unary_union(buffered_polygons)
    
    # Rasterize the polygons
    mask = rasterize(shapes=[unified_polygons],
                     out_shape=im_size)
    
    # Save the mask
    mask = mask.astype("uint16")
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})

    # Save the mask to the output path
    file_name = os.path.join(output_path, file_name)
    with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)

def generate_multi_bm(raster_path, shape_path, output_path, file_name):
    """
    Function that generates a list of binary mask from a vector file (shp or geojson)
    generating a mask for each polygon in the vector file.
    
    raster_path = path to the .tif;
    shape_path = path to the shapefile or GeoJson.
    output_path = Path to save the binary mask.
    file_name = Name of the file.
    """
    
    # Load raster
    with rasterio.open(raster_path, "r") as src:
        raster_meta = src.meta
    
    # Load the shapefile or GeoJSON
    train_df = gpd.read_file(shape_path)
    
    # Verify crs
    if train_df.crs != src.crs:
        print("Raster crs: {}, Vector crs: {}.\nConvert vector and raster to the same CRS.".format(src.crs, train_df.crs))
        
    # Function to generate the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        if polygon.geom_type == 'Polygon':
            polygons = [polygon]
        elif polygon.geom_type == 'MultiPolygon':
            polygons = polygon.geoms

        for poly in polygons:
            for i in np.array(poly.exterior.coords):
                poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly
    
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for _, row in train_df.iterrows():
        if isinstance(row['geometry'], MultiPolygon):
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)

    # Create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over each polygon and create a separate raster mask for each
    for idx, poly in enumerate(poly_shp):
        # Create a unary union of the current polygon with buffer(0)
        buffered_polygon = poly.buffer(0)
        
        # Rasterize the current polygon
        mask = rasterize(shapes=[buffered_polygon], out_shape=im_size)
        
        # Save the mask for the current polygon
        mask = mask.astype("uint16")
        bin_mask_meta = src.meta.copy()
        bin_mask_meta.update({'count': 1})
        
        # Define the output file name for the current polygon
        file_name = os.path.join(output_path, f"mask_polygon_{idx}.tif")
        
        # Save the mask to the output path
        with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
            dst.write(mask * 255, 1)

# Esempi di utilizzo: #
shp_to_bm_plot(tif_path = './barrali_shp_raster/Barrali_EXTRAURBANO.tif',
               shapefile_path = './barrali_shp_raster/111004_barrali_puc_20240118.shp')

shp_to_geojson(shapefile_path = './barrali_shp_raster/111004_barrali_puc_20240118.shp',
                output_path = 'file.geojson')

# converte l'immagine originale da tif in png
tif_to_png(input_folder="./barrali_shp_raster",
            output_folder="./binary_mask/origin_img")

# single bm
generate_single_bm(raster_path="./barrali_shp_raster/Barrali_EXTRAURBANO.tif",
                    shape_path="./barrali_shp_raster/111004_barrali_puc_20240118.shp",
                    output_path="./binary_mask/out_single",
                    file_name="111004_barrali_puc_20240118_mask.tif")

tif_to_png(input_folder="./binary_mask/out_single",
            output_folder="./binary_mask/out_single_png")

# multi bm
generate_multi_bm(raster_path="./barrali_shp_raster/Barrali_EXTRAURBANO.tif",
                shape_path="./barrali_shp_raster/111004_barrali_puc_20240118.shp",
                output_path="./binary_mask/out_multiple",
                file_name="111004_barrali_puc_20240118_mask.tif")

tif_to_png(input_folder="./binary_mask/out_multiple",
            output_folder="./binary_mask/out_multiple_png")


