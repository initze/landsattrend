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

current_dir = os.getcwd()
print('current dir', current_dir)

service_account = "pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com"
path_to_file = os.path.join(os.getcwd(), 'export_tools', 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
credentials = ee.ServiceAccountCredentials(service_account, path_to_file)
storage_client = storage.Client.from_service_account_json(
    path_to_file)
ee.Initialize(credentials)

from google.cloud import storage
from utils.utils_processing import *
import generate_zones

storage_client = storage.Client.from_service_account_json(
    path_to_file)

#
exported_files = 'exported_files.txt'
# CLOUD bucket parameters -different to the one on the bottom
outputBucket = 'pdg-landsattrend'

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}

def download_from_cloud(current_site_name, current_start_year, current_end_year):
    zones_to_run = []
    CURRENT_PROCESS_SITE = current_site_name
    start_zone = generate_zones.get_zone(regions[CURRENT_PROCESS_SITE]['X_MIN_START'])
    end_zone = generate_zones.get_zone(regions[CURRENT_PROCESS_SITE]['X_MIN_END'])
    year_substring = str(current_start_year) + '-' + str(current_end_year)
    substrings_to_match = []
    start_zone_int = int(start_zone)
    end_zone_int = int(end_zone)
    zones_ints = np.arange(start_zone_int, end_zone_int + 1)
    print(zones_ints)
    for zone in zones_ints:
        zone_string = str(zone)
        zones_to_run.append(zone_string)
        substring = year_substring + '_' + zone_string
        substrings_to_match.append(substring)
    bucket = storage_client.get_bucket(outputBucket)
    blob_list = bucket.list_blobs()
    for each_blob in blob_list:
        current_name = each_blob.name
        matches = any(substring in current_name for substring in substrings_to_match)
        if matches:
            print("We should download this", current_name)
            current_name_parts = current_name.split('/')
            current_base_name = current_name_parts[-1]
            current_base_name_parts = current_base_name.split('_')
            current_years = current_base_name_parts[1]
            current_zone = current_base_name_parts[2]
            path_to_download = os.path.join(os.getcwd(), 'data', current_zone, current_years, 'tiles', current_base_name)
            each_blob.download_to_filename(path_to_download)
    return zones_to_run
