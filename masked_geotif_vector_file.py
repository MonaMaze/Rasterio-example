# Load all libraries
import os
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.plot
import rasterio.mask
from rasterio.enums import MergeAlg
from rasterio.plot import show
from numpy import int16
from pyproj import CRS, Proj
import fiona

# Reading RUV shapefile
ruv_shp = gpd.read_file('shapefile/RUV.shp')
with fiona.open("shapefile/RUV.shp", "r") as shapefile:
    features = [feature["geometry"] for feature in shapefile]

# Select the dates folders in LU folder
for date_folder in os.listdir('LU'):
    # Select the geotif images in each folder
    folder = os.path.join('LU', date_folder)
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        # Mask the original tif file
        with rasterio.open(img_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
            out_profile = src.profile.copy()
        
        # Save the masked result as a tif file
        res_path = os.path.join('masked', date_folder)
        os.makedirs(res_path, exist_ok=True)
        out_profile.update({"driver": "GTiff", 'width': out_image.shape[2],'height': out_image.shape[1], 'transform': out_transform})
        with rasterio.open(os.path.join(res_path, 'masked_' + file_name), 'w', **out_profile) as dst:
            dst.write(out_image)
        mas_shr = rasterio.open(os.path.join(res_path, 'masked_' + file_name))
            
# Convert shapefile to geotiff image with the Class_id value
# create tuples of geometry, value pairs, where value is the attribute value you want to burn
geom_value = ((geom,value) for geom, value in zip(ruv_shp.geometry, ruv_shp['Class_id']))
# Rasterize vector using the shape and transform of the raster
rasterized = rasterio.features.rasterize(geom_value, out_shape = (mas_shr.shape[0], mas_shr.shape[1]), transform = mas_shr.transform, all_touched = True,
                                         fill = 0, merge_alg = MergeAlg.replace, dtype = int16)
# Save the rasterized vector
with rasterio.open("images/vector_shr.tif", "w", driver = "GTiff", crs = mas_shr.crs, transform = mas_shr.transform, dtype = rasterio.uint8, count = 1,
                   width = mas_shr.width, height = mas_shr.height) as dst:
    dst.write(rasterized, indexes = 1)