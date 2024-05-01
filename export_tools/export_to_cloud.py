import ee
import sys
import os
import time
import datetime
import numpy as np
import sys
import argparse
from google.cloud import storage
from modules import high_level_functions


service_account = "pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com"
path_to_file = os.path.join(os.getcwd(), 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
credentials = ee.ServiceAccountCredentials(service_account, path_to_file)
storage_client = storage.Client.from_service_account_json(
    path_to_file)
ee.Initialize(credentials)

from google.cloud import storage
from utils.utils_processing import *

storage_client = storage.Client.from_service_account_json(
    path_to_file)

#
exported_files = 'exported_files.txt'
# CLOUD bucket parameters -different to the one on the bottom
outputBucket = 'pdg-landsattrend' #Change for your Cloud Storage bucket

# DOWNLOAD PARAMETERS
data_cols = ['TCB_slope',
             'TCB_offset',
             'TCG_slope',
             'TCG_offset',
             'TCW_slope',
             'TCW_offset',
             'NDVI_slope',
             'NDVI_offset',
             'NDMI_slope',
             'NDMI_offset',
             'NDWI_slope',
             'NDWI_offset',
             'elevation',
             'slope']

MAXCLOUD = 70
STARTYEAR = 2000
ENDYEAR = 2020
STARTMONTH = 7
ENDMONTH = 8
SCALE = 30

PROCESS_SITE = "TEST"

parser=argparse.ArgumentParser()

parser.add_argument("--startyear", help="The start year")
parser.add_argument("--endyear", help="The end year")
parser.add_argument("--process_site", help="The PROCESS_SITE")

args=parser.parse_args()

print(f"Dict format: {vars(args)}")

if 'process_site' in vars(args):
    if vars(args)['process_site'] is not None:
        print("We have a process site")
        PROCESS_SITE = vars(args)['process_site']
if 'startyear' in vars(args):
    if vars(args)['startyear'] is not None:
        print("We have a start year")
        startyear_value = int(vars(args)['startyear'])
        STARTYEAR = startyear_value
if 'endyear' in vars(args):
    if vars(args)['endyear'] is not None:
        print("We have an end year")
        endyear_value = int(vars(args)['endyear'])
        ENDYEAR = endyear_value

X_SIZE = 3
Y_SIZE = 1
X_OVERLAP = 0.1
X_SIZE = X_SIZE + (2*X_OVERLAP)

config_trend = {
  'date_filter_yr' : ee.Filter.calendarRange(STARTYEAR, ENDYEAR, 'year'),
  'date_filter_mth' : ee.Filter.calendarRange(STARTMONTH, ENDMONTH, 'month'),
  'meta_filter_cld' : ee.Filter.lt('CLOUD_COVER', MAXCLOUD),
  'select_bands_visible' : ["B1", "B2","B3","B4"],
  'select_indices' : ["TCB", "TCG", "TCW", "NDVI", "NDMI", "NDWI"],
  'select_TCtrend_bands' : ["TCB_slope", "TCG_slope", "TCW_slope"],
  'geom' : None
}


def get_zone(lon):
    utm = get_utmzone_from_lon(lon)
    zone = epsg_from_utmzone(utm)
    return zone


def is_file_exported_cloud(filename, cloudFolder, outputBucket):
    bucket = storage_client.get_bucket(outputBucket)
    name_in_bucket = cloudFolder + '/' + filename
    blob_list = bucket.list_blobs()
    for each_blob in blob_list:
        current_name = each_blob.name
        if name_in_bucket in current_name:
            return True
    return False


def get_filename_to_upload(X_MIN, Y_MIN, ZONE):
    print('getting filename for', X_MIN, Y_MIN, ZONE)
    file_name = f'trendimage_' + str(STARTYEAR) + '-' + str(ENDYEAR) + '_' + ZONE + '_' + str(X_MIN) + '_' + str(Y_MIN)
    file_name = file_name + '.tif'
    return file_name


def run_preprocess(config_trend, geom, X_MIN, Y_MIN, outputBucket, cloudFolder, mask_non_water=True, run_task=True):

    CURRENT_ZONE = get_zone(X_MIN)
    print('in run preprocess')
    print('outputBucket', outputBucket)
    config_trend['geom'] = geom
    trend = high_level_functions.runTCTrend(config_trend)
    data = trend['data']

    #### setup data
    dem = create_dem_data()
    data_export = data.addBands(dem).toFloat().select(data_cols)
    if mask_non_water:
        # new version set to 90 m
        water_mask = get_water_mask(dilation_size=90)
        data_export = data_export.updateMask(water_mask)

    file_name = f'trendimage_' + str(STARTYEAR) + '-' + str(ENDYEAR) + '_' + CURRENT_ZONE + '_' + str(X_MIN) + '_' + str(Y_MIN)
    cloudFolder = cloudFolder + '/' + str(STARTYEAR) + '-' + str(ENDYEAR)
    if run_task:
        # Export the image, specifying scale and region.
        task = ee.batch.Export.image.toCloudStorage(**{
            'image': data_export,
            'description': file_name,
            'scale': 30,
            'region': geom,
            'crs': f'EPSG:{CURRENT_ZONE}',
            'fileFormat': 'GeoTIFF',
            'fileNamePrefix': cloudFolder + '/' + file_name,
            'bucket': outputBucket,
            'maxPixels': 1e12,
            'formatOptions': {'cloudOptimized': True}
        })
        task.start()
        file_name = file_name + '.tif'
        with open(exported_files, 'a') as f:
            f.write(file_name + '\n')
        return [task, file_name]
    else:
        return [None, None]


def check_for_files(files_to_upload, outputBucket):
    bucket = storage_client.get_bucket(outputBucket)

    blob_list = bucket.list_blobs()
    blob_names = []
    for each_blob in blob_list:
        blob_name = each_blob.name
        blob_names.append(blob_name)
    all_files_uploaded = True
    for file in files_to_upload:
        if file not in blob_names:
            all_files_uploaded = False
    return all_files_uploaded

# DUPLICATE?
"""
def get_filenames_to_upload(Y_MIN_START, Y_MIN_END, X_MIN_START, X_MIN_END, outputBucket, ZONE):
    file_names = []
    for Y_MIN in range(Y_MIN_START, Y_MIN_END):
        for X_MIN in range(X_MIN_START, X_MIN_END, 3):
            file_name = get_filename_to_upload(X_MIN, Y_MIN, ZONE)
            file_names.append(file_name)
    return file_names
"""


def get_filenames_to_upload(Y_MIN_START, Y_MIN_END, X_MIN_START, X_MIN_END):
    filenames_to_upload = []
    for Y_MIN in range(Y_MIN_START, Y_MIN_END):
        for X_MIN in range(X_MIN_START, X_MIN_END, 3):
            CURRENT_ZONE = get_zone(X_MIN)
            filename_to_upload = get_filename_to_upload(X_MIN, Y_MIN, CURRENT_ZONE)
            filenames_to_upload.append(filename_to_upload)
    return filenames_to_upload


def run_export(Y_MIN_START, Y_MIN_END, X_MIN_START, X_MIN_END, outputBucket, dataset_files = [], X_OVERLAP=0.1, X_SIZE=3, Y_SIZE=1, cloudFolder='TEST', run_task=True):

    tasks = []
    file_names = []

    print('running export to output bucket', outputBucket)

    for Y_MIN in range(Y_MIN_START, Y_MIN_END, Y_SIZE):
        for X_MIN in range(X_MIN_START, X_MIN_END, X_SIZE):
            geom = ee.Geometry.Rectangle(coords=[X_MIN-X_OVERLAP, Y_MIN, X_MIN+X_SIZE, Y_MIN+Y_SIZE], proj=ee.Projection('EPSG:4326'))
            config_trend['geom'] = geom
            CURRENT_ZONE = get_zone(X_MIN)
            filename_to_upload = get_filename_to_upload(X_MIN, Y_MIN, CURRENT_ZONE)
            file_already_exported = is_file_exported_cloud(filename_to_upload,cloudFolder, outputBucket)
            print(filename_to_upload, file_already_exported)
            if run_task is False:
                print(filename_to_upload)
            if run_task:
                if not file_already_exported:
                    print('exporting file: ' + filename_to_upload + ' to google cloud bucket')
                    with open('export_eurasia4.txt', 'a') as f:
                        f.write(filename_to_upload + '\n')
                    result = run_preprocess(config_trend, geom, X_MIN, Y_MIN, outputBucket, cloudFolder)
                    tasks.append(result[0])
                    file_names.append(result[1])
                else:
                    print('file: ' + filename_to_upload + ' has already been uploaded')
    return [tasks, file_names]


if __name__ == "__main__":

    regions = {
        'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
        'ALASKA': {'Y_MIN_START':55,'Y_MIN_END':72,'X_MIN_START':-168, 'X_MIN_END':-138},
        'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
        'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
        'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
        'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
    }

    bucket_name = "pdg-landsattrend"

    if PROCESS_SITE == 'ALL':
        all_sites = list(regions.keys())
        for each_site in all_sites:
            start_zone = get_zone(regions[each_site]['X_MIN_START'])
            end_zone = get_zone(regions[each_site]['X_MIN_END'])
            try:
                run_export(Y_MIN_START=regions[each_site]['Y_MIN_START'], \
                           Y_MIN_END=regions[each_site]['Y_MIN_END'], \
                           X_MIN_START=regions[each_site]['X_MIN_START'], \
                           X_MIN_END=regions[each_site]['X_MIN_END'], \
                           outputBucket=bucket_name, \
                           cloudFolder=each_site)
            except Exception as e:
                print(e)
    else:
        start_zone = get_zone(regions[PROCESS_SITE]['X_MIN_START'])
        end_zone = get_zone(regions[PROCESS_SITE]['X_MIN_END'])
        try:
            run_export(Y_MIN_START=regions[PROCESS_SITE]['Y_MIN_START'], \
                       Y_MIN_END=regions[PROCESS_SITE]['Y_MIN_END'], \
                       X_MIN_START=regions[PROCESS_SITE]['X_MIN_START'], \
                       X_MIN_END=regions[PROCESS_SITE]['X_MIN_END'], \
                       outputBucket=bucket_name, \
                       cloudFolder=PROCESS_SITE)
        except Exception as e:
            print(e)