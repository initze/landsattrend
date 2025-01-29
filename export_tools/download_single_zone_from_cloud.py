import ee
import sys
import os
import time
import datetime
import numpy as np
import sys
import argparse
from google.cloud import storage
from pathlib import Path


current_dir = os.getcwd()
print('current dir', current_dir)

service_account = "pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com"
path_to_file = os.path.join(os.getcwd(), 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
print("THE PATH TO CRED FILE IS ")
print(path_to_file)
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

DOWNLOAD_PATH = '/work/hdd/bbou/toddn/landsat-delta/landsattrend/data/'

def download_from_cloud(current_zone, current_start_year, current_end_year):

    bucket = storage_client.get_bucket(outputBucket)
    blob_list = bucket.list_blobs()
    substring = current_start_year + '-' + current_end_year + '_' + current_zone
    year_span = current_start_year + '-' + current_end_year
    for blob in blob_list:
        current_name = blob.name
        if substring in current_name:
            print(current_name)
            print("We should download this file.")
            current_download_path_dir = os.path.join(DOWNLOAD_PATH, current_zone, year_span, 'tiles')
            if os.path.exists(current_download_path_dir) and os.path.isdir(current_download_path_dir):
                print("Directory already exists", current_download_path_dir)
            else:
                print('create dir', current_download_path_dir)
                Path(current_download_path_dir).mkdir(parents=True, exist_ok=True)
            path_for_file = os.path.join(current_download_path_dir, current_name)
            print("Downloading to", path_for_file)
            blob.download_to_filename(path_for_file)

if __name__ == "__main__":
    zone = sys.argv[1]
    start_year = sys.argv[2]
    end_year = sys.argv[3]

    download_from_cloud(zone,start_year, end_year)
    print("Done donwloading")
