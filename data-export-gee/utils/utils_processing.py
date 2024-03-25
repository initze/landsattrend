import os
import shutil

import ee
import numpy as np

import os
from google.cloud import storage
service_account = "pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com"
path_to_file = os.path.join(os.getcwd(), 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
credentials = ee.ServiceAccountCredentials(service_account, path_to_file)
storage_client = storage.Client.from_service_account_json(
    path_to_file)
ee.Initialize(credentials)


def get_utmzone_from_lon(lon):
    return int(31 + np.floor(lon / 6))


def crs_from_utmzone(utm):
    return f'EPSG:326{utm:02d}'


def epsg_from_utmzone(utm):
    return f'326{utm:02d}'


def prefix_from_utmzone(utm):
    return f'trendimage_Z{utm:02d}'


def create_dem_data():
    # Create DEM data from various sources
    alosdem = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").select(['DSM'], ['elevation'])
    proj = alosdem.first().select(0).projection()
    # Alos slope calculation needs special projection treatment
    alos_slope = ee.Terrain.slope(alosdem.mosaic().setDefaultProjection(proj)).select([0], ['slope'])
    alosdem = alosdem.mosaic().addBands(alos_slope).select(['elevation', 'slope']).toFloat()

    nasadem = ee.Image("NASA/NASADEM_HGT/001").select(['elevation'])
    nasadem = nasadem.addBands(ee.Terrain.slope(nasadem)).select(['elevation', 'slope']).toFloat()

    arcticDEM = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic").select(['elevation'])
    arcticDEM = arcticDEM.addBands(ee.Terrain.slope(arcticDEM)).select(['elevation', 'slope']).toFloat()
    dem = ee.ImageCollection([arcticDEM, alosdem, nasadem]).mosaic()
    return dem


def get_lon_from_utmzone(zone, distance):
    start = (zone - 31) * 6
    return list(range(start, start + 6, distance))


def epsgprefix_from_utmzone(utm):
    epsg = epsg_from_utmzone(utm)
    return f'trendimage_{epsg}'


def make_fileprefix(epsg, period_start, period_end, lon, lat):
    fileprefix = f'trendimage_{period_start}-{period_end}_{epsg}_{lon}_{lat}'
    return fileprefix


def make_move(row):
    if not row.filepath_out.exists():
        print(f'creating new directory {row.filepath_out}')
        os.makedirs(row.filepath_out)
    try:
        print(f'Moving file {row.filename}')
        shutil.move(str(row.filepath), str(row.filepath_out))
    except:
        print(f'Skipping file {row.filename}')


def get_water_mask(dilation_size=90):
    jrc_GSW = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')
    masked = jrc_GSW.select('occurrence').gt(0).select([0], ['water_mask']).unmask()
    water_mask = masked.focalMax(kernelType='circle', radius=dilation_size, units='meters')
    return water_mask
