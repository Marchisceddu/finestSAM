from osgeo import ogr

def convert_shp_to_geojson(input_shapefile, output_geojson):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(input_shapefile, 0)
    layer = dataSource.GetLayer()

    # Create GeoJSON
    out_driver = ogr.GetDriverByName('GeoJSON')
    out_ds = out_driver.CreateDataSource(output_geojson)
    out_layer = out_ds.CreateLayer('layer1', geom_type=ogr.wkbPolygon)
    out_layer.CreateFields(layer.schema)

    # Write features to GeoJSON
    for feature in layer:
        out_layer.CreateFeature(feature)

    # Close data sources
    dataSource = None
    out_ds = None

# Usage example
input_shapefile = './shape/91066_ORTUERI_PUC_20240112_STE.shp'
output_geojson = 'output.geojson'
convert_shp_to_geojson(input_shapefile, output_geojson)
