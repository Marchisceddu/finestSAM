# # ./barrali_shp_raster/Barrali_EXTRAURBANO.tif
# # ./barrali_shp_raster/111004_barrali_puc_20240118.shp
# import numpy as np
# import geopandas as gpd
# from osgeo import gdal
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch

# # Leggi il file TIFF georeferenziato
# tif_path = './barrali_shp_raster/Barrali_EXTRAURBANO.tif'
# ds = gdal.Open(tif_path)
# array = ds.ReadAsArray()

# # Trasponi l'array se necessario
# if array.shape[0] == 3:  # Se l'immagine ha tre canali (RGB)
#     array = np.transpose(array, [1, 2, 0])  # Trasponi l'array

# # Estrai le informazioni sulla geotrasformazione
# geotransform = ds.GetGeoTransform()
# origin_x = geotransform[0]
# origin_y = geotransform[3]
# pixel_width = geotransform[1]
# pixel_height = geotransform[5]

# # Leggi il file shapefile
# shapefile_path = './barrali_shp_raster/111004_barrali_puc_20240118.shp'
# gdf = gpd.read_file(shapefile_path)

# # Creare il plot
# fig, ax = plt.subplots()

# # Visualizza l'immagine TIFF
# ax.imshow(array, extent=[origin_x, origin_x + ds.RasterXSize * pixel_width,
#                          origin_y + ds.RasterYSize * pixel_height, origin_y])

# # Visualizza le shape sopra l'immagine
# gdf.plot(ax=ax, color='black')

# # Aggiungi la legenda
# legend_elements = [Patch(facecolor='red', edgecolor='red', label='Shapefile')]
# ax.legend(handles=legend_elements, loc='upper right')

# # Mostra il plot
# plt.show()





# test 2
import numpy as np
import geopandas as gpd
from osgeo import gdal
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Leggi il file TIFF georeferenziato
tif_path = './barrali_shp_raster/Barrali_EXTRAURBANO.tif'
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
shapefile_path = './barrali_shp_raster/111004_barrali_puc_20240118.shp'
gdf = gpd.read_file(shapefile_path)

# Creare il plot
fig, ax = plt.subplots()

# Visualizza l'immagine TIFF
ax.imshow(array, extent=[origin_x, origin_x + ds.RasterXSize * pixel_width,
                         origin_y + ds.RasterYSize * pixel_height, origin_y])

# Visualizza le shape sopra l'immagine
gdf.plot(ax = ax, color='red')

# Aggiungi la legenda
legend_elements = [Patch(facecolor='red', edgecolor='red', label='Shapefile')]
ax.legend(handles=legend_elements, loc='upper right')

# Converti l'array in uint8
array_uint8 = (array / np.max(array) * 255).astype(np.uint8)

# Salva l'immagine TIFF come PNG
tif_image = Image.fromarray(array_uint8)
tif_image.save('immagine_con_shape.png')

# Crea la maschera binaria delle shape
mask = np.zeros((array.shape[0], array.shape[1]), dtype=np.uint8)
for geom in gdf.geometry:
    if geom.geom_type == 'Polygon':
        coords = np.column_stack((geom.exterior.xy[0], geom.exterior.xy[1]))
        coords = (coords - [origin_x, origin_y]) / [pixel_width, pixel_height]
        coords = np.round(coords).astype(int)
        rr, cc = coords[:, 1], coords[:, 0]
        rr = np.clip(rr, 0, array.shape[0] - 1)
        cc = np.clip(cc, 0, array.shape[1] - 1)
        mask[rr, cc] = 255
    elif geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:  # Itera sui singoli poligoni all'interno del MultiPolygon
            coords = np.column_stack((polygon.exterior.xy[0], polygon.exterior.xy[1]))
            coords = (coords - [origin_x, origin_y]) / [pixel_width, pixel_height]
            coords = np.round(coords).astype(int)
            rr, cc = coords[:, 1], coords[:, 0]
            rr = np.clip(rr, 0, array.shape[0] - 1)
            cc = np.clip(cc, 0, array.shape[1] - 1)
            mask[rr, cc] = 255

# Salva la maschera come PNG
mask_image = Image.fromarray(mask, mode='L')
mask_image.save('mask_shape.png')

# Mostra il plot
# plt.show()
