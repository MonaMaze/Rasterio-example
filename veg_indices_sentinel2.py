# Load all libraries
import dask.distributed
import dask.utils
import numpy as np
import planetary_computer as pc
import xarray as xr
from IPython.display import display
from pystac_client import Client
import matplotlib.pyplot as plt
from odc.stac import configure_rio, stac_load
import rasterio
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from pathlib import Path
import scipy
import scipy.ndimage
import os
import pyproj
from pyproj import CRS, Proj

# Configuration of bands. * used as wildcard
cfg = {
    "sentinel-2-l2a": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0},
            "SCL": {"data_type": "uint8", "nodata": 0},
            "visual": {"data_type": "uint8", "nodata": 0},
        },
    },
    "*": {"warnings": "ignore"},
}

# Start Dask Client: for improving load speed significantly (optional)
client = dask.distributed.Client()
configure_rio(cloud_defaults=True, client=client)
# Query STAC API for Sentinel-2 dataset
# Looking for Sentinel dataset
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

query = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime=("2021-06", "2021-08"),
    query={"s2:mgrs_tile": dict(eq="36RUV"), "eo:cloud_cover": {"lt": 4}}, #36RVV
)

# Load the required bands
xx = stac_load(
    items,
    bands=["red", "green", "blue", "B05", "nir", "B11", "B12", "SCL"],
    #chunks={"x": 2048, "y": 2048},
    stac_cfg=cfg,
    patch_url=pc.sign,
    resolution=10,
)
src = CRS.from_epsg(32636)

# Define a function for calculating the noramlized indices
def normalized_index(img, b1, b2, x, eps=0.0001):
    band1 = np.where((img[b1][x]==0) & (img[b2][x]==0), np.nan, img[b1][x])
    band2 = np.where((img[b1][x]==0) & (img[b2][x]==0), np.nan, img[b2][x])
    return (band1 - band2) / (band1 + band2)

# Calculating Vegetation indices
for i in len(items):
    # Getting the properties of each image
    resdict = query.get_all_items_as_dict()['features'][i]
    trans = resdict['assets']['B04']['proj:transform']
    dt = resdict['properties']['datetime'][0:10]
    
    # Calculating the vegetation indices for each image and save it
    # NDVI index
    ndvi = normalized_index(xx, "nir", "red", i)
    # Save NDVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndvi_tif = rasterio.open(os.path.join(img_path, 'ndvi_'+ dt +'.tif'), 'w', driver='GTiff', height = ndvi.shape[0], width = ndvi.shape[1], count=1, dtype=str(ndvi.dtype), crs=src, transform=trans)
    ndvi_tif.write(ndvi, 1)
    ndvi_tif.close()
    
    # KNDVI index
    kndvi = np.tanh(((xx['nir'][i] - xx['red'][i]) / (2 * (0.5 * (xx['nir'][i] + xx['red'][i]))))**2)
    # Save kNDVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    kndvi_tif = rasterio.open(os.path.join(img_path, 'kndvi_'+ dt +'.tif'), 'w', driver='GTiff', height = kndvi.shape[0], width = kndvi.shape[1], count=1, dtype=str(kndvi.dtype), crs=src, transform=trans)
    kndvi_tif.write(kndvi, 1)
    kndvi_tif.close()
    
    # NDBI index
    ndbi = normalized_index(xx, "B11", "nir", i)
    # Save kNDVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndbi_tif = rasterio.open(os.path.join(img_path, 'ndbi_'+ dt +'.tif'), 'w', driver='GTiff', height = ndbi.shape[0], width = ndbi.shape[1], count=1, dtype=str(ndbi.dtype), crs=src, transform=trans)
    ndbi_tif.write(ndbi, 1)
    ndbi_tif.close()
    
    # DBSI index
    dbsi = normalized_index(xx, "B11", "green", i) - ndvi
    # Save DBSI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    dbsi_tif = rasterio.open(os.path.join(img_path, 'dsbi_'+ dt +'.tif'), 'w', driver='GTiff', height = dbsi.shape[0], width = dbsi.shape[1], count=1, dtype=str(dbsi.dtype), crs=src, transform=trans)
    dbsi_tif.write(dbsi, 1)
    dbsi_tif.close()
    
    # SAVI index
    savi = 1.5 * (xx['nir'][i] - xx['red'][i]) * (xx['nir'][i] + xx['red'][i] + 0.5)    
    # Save SAVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    savi_tif = rasterio.open(os.path.join(img_path, 'savi_'+ dt +'.tif'), 'w', driver='GTiff', height = savi.shape[0], width = savi.shape[1], count=1, dtype=str(savi.dtype), crs=src, transform=trans)
    savi_tif.write(savi, 1)
    savi_tif.close()
    
    # OSAVI index
    osavi = 1.16 * (xx['nir'][i] - xx['red'][i]) * (xx['nir'][i] + xx['red'][i] + 0.16)    
    # Save OSAVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    osavi_tif = rasterio.open(os.path.join(img_path, 'osavi_'+ dt +'.tif'), 'w', driver='GTiff', height = osavi.shape[0], width = osavi.shape[1], count=1, dtype=str(osavi.dtype), crs=src, transform=trans)
    osavi_tif.write(osavi, 1)
    osavi_tif.close()
    
    # NDWI index
    ndwi = normalized_index(xx, "green", "nir", i)    
    # Save NDWI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndwi_tif = rasterio.open(os.path.join(img_path, 'ndwi_'+ dt +'.tif'), 'w', driver='GTiff', height = ndwi.shape[0], width = ndwi.shape[1], count=1, dtype=str(ndwi.dtype), crs=src, transform=trans)
    ndwi_tif.write(ndwi, 1)
    ndwi_tif.close()
    
    # NDMI index
    ndmi = normalized_index(xx, "nir", "B11", i)    
    # Save NDMI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndmi_tif = rasterio.open(os.path.join(img_path, 'ndmi_'+ dt +'.tif'), 'w', driver='GTiff', height = ndmi.shape[0], width = ndmi.shape[1], count=1, dtype=str(ndmi.dtype), crs=src, transform=trans)
    ndmi_tif.write(ndmi, 1)
    ndmi_tif.close()
    
    # MNDWI index
    mndmi = normalized_index(xx, "green", "B11", i)    
    # Save MNDMI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    mndmi_tif = rasterio.open(os.path.join(img_path, 'mndmi_'+ dt +'.tif'), 'w', driver='GTiff', height = mndmi.shape[0], width = mndmi.shape[1], count=1, dtype=str(mndmi.dtype), crs=src, transform=trans)
    mndmi_tif.write(mndmi, 1)
    mndmi_tif.close()
    
    # WRI index
    wri = (xx['green'][i] + xx['red'][i]) / (xx['nir'][i] + xx['B11'][i])    
    # Save WRI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    wri_tif = rasterio.open(os.path.join(img_path, 'wri_'+ dt +'.tif'), 'w', driver='GTiff', height = wri.shape[0], width = wri.shape[1], count=1, dtype=str(wri.dtype), crs=src, transform=trans)
    wri_tif.write(wri, 1)
    wri_tif.close()
    
    # AWEI index
    awei = xx['blue'][i] + 2.5 * xx['green'][i] - 1.5 * (xx['nir'][i] + xx['B11'][i]) - 0.25 * xx['B12'][i]    
    # Save AWEI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    awei_tif = rasterio.open(os.path.join(img_path, 'awei_'+ dt +'.tif'), 'w', driver='GTiff', height = awei.shape[0], width = awei.shape[1], count=1, dtype=str(awei.dtype), crs=src, transform=trans)
    awei_tif.write(awei, 1)
    awei_tif.close()
    
    # STR index (Needs downscaling first)
    # From B11
    str11 = ((1 - xx['B11'][i])**2) / (2 * xx['B11'][i])    
    # Save STR calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    str11_tif = rasterio.open(os.path.join(img_path, 'str11_'+ dt +'.tif'), 'w', driver='GTiff', height = str11.shape[0], width = str11.shape[1], count=1, dtype=str(str11.dtype), crs=src, transform=trans)
    str11_tif.write(str11, 1)
    str11_tif.close()
    
    # From B12
    str11 = ((1 - xx['B12'][i])**2) / (2 * xx['B12'][i])    
    # Save STR calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    str12_tif = rasterio.open(os.path.join(img_path, 'str12_'+ dt +'.tif'), 'w', driver='GTiff', height = str12.shape[0], width = str12.shape[1], count=1, dtype=str(str12.dtype), crs=src, transform=trans)
    str12_tif.write(str12, 1)
    str12_tif.close()
    
    # Average from B11 and B12
    stravg = (str11 + str12) / 2    
    # Save STR calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    stravg_tif = rasterio.open(os.path.join(img_path, 'stravg_'+ dt +'.tif'), 'w', driver='GTiff', height = stravg.shape[0], width = stravg.shape[1], count=1, dtype=str(stravg.dtype), crs=src, transform=trans)
    stravg_tif.write(stravg, 1)
    stravg_tif.close()
    
    # RVI index
    # With red band
    rvi1 = xx['nir'][i] / xx['red'][i]    
    # Save RVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    rvi1_tif = rasterio.open(os.path.join(img_path, 'rvi1_'+ dt +'.tif'), 'w', driver='GTiff', height = rvi1.shape[0], width = rvi1.shape[1], count=1, dtype=str(rvi1.dtype), crs=src, transform=trans)
    rvi1_tif.write(rvi1, 1)
    rvi1_tif.close()
    
    # With green band
    rvi2 = xx['nir'][i] / xx['green'][i]    
    # Save RVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    rvi2_tif = rasterio.open(os.path.join(img_path, 'rvi2_'+ dt +'.tif'), 'w', driver='GTiff', height = rvi2.shape[0], width = rvi2.shape[1], count=1, dtype=str(rvi2.dtype), crs=src, transform=trans)
    rvi2_tif.write(rvi2, 1)
    rvi2_tif.close()
    
    # TVI 
    tvi = 60 * (xx['nir'][i] - xx['green'][i]) - 100 * (xx['red'][i] - xx['green'][i])    
    # Save TVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    tvi_tif = rasterio.open(os.path.join(img_path, 'tvi_'+ dt +'.tif'), 'w', driver='GTiff', height = tvi.shape[0], width = tvi.shape[1], count=1, dtype=str(tvi.dtype), crs=src, transform=trans)
    tvi_tif.write(tvi, 1)
    tvi_tif.close()
    
    # EVI 
    evi = 2.5 * ((xx['nir'][i] - xx['red'][i]) / (xx['nir'][i] + 6 * xx['red'][i] - 7.5 * xx['blue'][i] + 1))    
    # Save EVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    evi_tif = rasterio.open(os.path.join(img_path, 'evi_'+ dt +'.tif'), 'w', driver='GTiff', height = evi.shape[0], width = evi.shape[1], count=1, dtype=str(evi.dtype), crs=src, transform=trans)
    evi_tif.write(evi, 1)
    evi_tif.close()
    
    # GI 
    gi = xx['green'][i] / xx['red'][i]    
    # Save GI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    gi_tif = rasterio.open(os.path.join(img_path, 'gi_'+ dt +'.tif'), 'w', driver='GTiff', height = gi.shape[0], width = gi.shape[1], count=1, dtype=str(gi.dtype), crs=src, transform=trans)
    gi_tif.write(gi, 1)
    gi_tif.close()
    
    # CGI
    cgi = xx['green'][i] / xx['red'][i] -1    
    # Save CGI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    cgi_tif = rasterio.open(os.path.join(img_path, 'cgi_'+ dt +'.tif'), 'w', driver='GTiff', height = cgi.shape[0], width = cgi.shape[1], count=1, dtype=str(cgi.dtype), crs=src, transform=trans)
    cgi_tif.write(cgi, 1)
    cgi_tif.close()
    
    # GNDVI
    gndvi = normalized_index(xx, "nir", "green", i)    
    # Save GNDVI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    gndvi_tif = rasterio.open(os.path.join(img_path, 'gndvi_'+ dt +'.tif'), 'w', driver='GTiff', height = gndvi.shape[0], width = gndvi.shape[1], count=1, dtype=str(gndvi.dtype), crs=src, transform=trans)
    gndvi_tif.write(gndvi, 1)
    gndvi_tif.close()
    
    # SRPI
    srpi = xx['blue'][i] / xx['red'][i]    
    # Save SRPI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(os.path.join(img_path, 'srpi_'+ dt +'.tif'), exist_ok=True)
    srpi_tif = rasterio.open(img_path, 'w', driver='GTiff', height = srpi.shape[0], width = srpi.shape[1], count=1, dtype=str(srpi.dtype), crs=src,       transform=trans)
    srpi_tif.write(srpi, 1)
    srpi_tif.close()
    
    # NDPI
    ndpi = normalized_index(xx, "B11", "green", i)    
    # Save NDPI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndpi_tif = rasterio.open(os.path.join(img_path, 'ndpi_'+ dt +'.tif'), 'w', driver='GTiff', height = ndpi.shape[0], width = ndpi.shape[1], count=1, dtype=str(ndpi.dtype), crs=src, transform=trans)
    ndpi_tif.write(ndpi, 1)
    ndpi_tif.close()
    
    # SIPI
    sipi = normalized_index(xx, "nir", "blue", i)    
    # Save SIPI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(os.path.join(img_path, exist_ok=True)
    sipi_tif = rasterio.open(os.path.join(img_path, 'sipi_'+ dt +'.tif'), 'w', driver='GTiff', height = sipi.shape[0], width = sipi.shape[1], count=1, dtype=str(sipi.dtype), crs=src, transform=trans)
    sipi_tif.write(sipi, 1)
    sipi_tif.close()
    
    # NPCI
    npci = normalized_index(xx, "red", "blue", i)    
    # Save NPCI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    npci_tif = rasterio.open(os.path.join(img_path, 'npci_'+ dt +'.tif'), 'w', driver='GTiff', height = npci.shape[0], width = npci.shape[1], count=1, dtype=str(npci.dtype), crs=src, transform=trans)
    npci_tif.write(npci, 1)
    npci_tif.close()
    
    # NDCI
    ndci = normalized_index(xx, "B05", "red", i)    
    # Save NDCI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndci_tif = rasterio.open(os.path.join(img_path, 'ndci_'+ dt +'.tif'), 'w', driver='GTiff', height = ndci.shape[0], width = ndci.shape[1], count=1, dtype=str(ndci.dtype), crs=src, transform=trans)
    ndci_tif.write(ndci, 1)
    ndci_tif.close()
    
    # PSRI
    psri = (xx['blue'][i] - xx['red'][i]) / xx['green'][i]    
    # Save PSRI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    psri_tif = rasterio.open(os.path.join(img_path, 'psri_'+ dt +'.tif'), 'w', driver='GTiff', height = psri.shape[0], width = psri.shape[1], count=1, dtype=str(psri.dtype), crs=src, transform=trans)
    psri_tif.write(psri, 1)
    psri_tif.close()
    
    # NDVIgb
    ndvigb = normalized_index(xx, "green", "blue", i)    
    # Save NDVIgb calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    ndvigb_tif = rasterio.open(os.path.join(img_path, 'ndvigb_'+ dt +'.tif'), 'w', driver='GTiff', height = ndvigb.shape[0], width = ndvigb.shape[1], count=1, dtype=str(ndvigb.dtype), crs=src, transform=trans)
    ndvigb_tif.write(ndvigb, 1)
    ndvigb_tif.close()
    
    # VARI
    vari = (xx['green'][i] - xx['red'][i]) / (xx['green'][i] + xx['red'][i] - xx['blue'][i])    
    # Save VARI calculated index as geotiff
    img_path = os.path.join('sentinel2', dt)
    os.makedirs(img_path, exist_ok=True)
    vari_tif = rasterio.open(os.path.join(img_path, 'vari_'+ dt +'.tif'), 'w', driver='GTiff', height = vari.shape[0], width = vari.shape[1], count=1, dtype=str(vari.dtype), crs=src,       transform=trans)
    vari_tif.write(vari, 1)
    vari_tif.close()
    
    # Save red band as geotiff
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    red_tif = rasterio.open(os.path.join(img_path, 'red_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['red'][i].shape[i], width=xx['red'][i].shape[1], count=1, dtype=str(xx['red'][i].dtype), crs=src, transform=trans)
    red_tif.write(xx['red'][i], 1)
    red_tif.close()
    
    # Save blue band as geotiff
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    blue_tif = rasterio.open(os.path.join(img_path, 'blue_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['blue'][i].shape[i], width=xx['blue'][i].shape[1], count=1, dtype=str(xx['blue'][i].dtype), crs=src, transform=trans)
    blue_tif.write(xx['blue'][i], 1)
    blue_tif.close()
    
    # Save green band as geotiff
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    green_tif = rasterio.open(os.path.join(img_path, 'green_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['green'][i].shape[i], width=xx['green'][i].shape[1], count=1, dtype=str(xx['green'][i].dtype), crs=src, transform=trans)
    green_tif.write(xx['green'][i], 1)
    green_tif.close()
    
    # Save nir band as geotiff
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    nir_tif = rasterio.open(os.path.join(img_path, 'nir_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['nir'][i].shape[i], width=xx['nir'][i].shape[1], count=1, dtype=str(xx['nir'][i].dtype), crs=src, transform=trans)
    nir_tif.write(xx['nir'][i], 1)
    nir_tif.close()
    
    # Save B11 band as geotiff (needs downscaling first)
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    B11_tif = rasterio.open(os.path.join(img_path, 'B11_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['B11'][i].shape[i], width=xx['B11'][i].shape[1], count=1, dtype=str(xx['B11'][i].dtype), crs=src, transform=trans)
    B11_tif.write(xx['B11'][i], 1)
    B11_tif.close()
    
    # Save B12 band as geotiff (needs downscaling first)
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    B12_tif = rasterio.open(os.path.join(img_path, 'B12_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['B12'][i].shape[i], width=xx['B12'][i].shape[1], count=1, dtype=str(xx['B12'][i].dtype), crs=src, transform=trans)
    B12_tif.write(xx['B12'][i], 1)
    B12_tif.close()
    
    # Save B05 band as geotiff (needs downscaling first)
    img_path = os.path.join('sentinel2',  dt)
    os.makedirs(img_path, exist_ok=True)
    B05_tif = rasterio.open(os.path.join(img_path, 'B05_'+ dt +'.tif'), 'w', driver='GTiff', height=xx['B05'][i].shape[i], width=xx['B05'][i].shape[1], count=1, dtype=str(xx['B05'][i].dtype), crs=src, transform=trans)
    B05_tif.write(xx['B05'][i], 1)
    B05_tif.close()