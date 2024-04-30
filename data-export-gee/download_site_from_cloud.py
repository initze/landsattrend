import ee#, eemont
import sys
import os
from pathlib import Path
import generate_zones
import argparse

service_account = 'pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com'
path_to_file = os.path.join(os.getcwd(),'data-export-gee', 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
credentials = ee.ServiceAccountCredentials(service_account, path_to_file)
ee.Initialize(credentials)
print(f"After authentication")


print(f"After url and key")
landsat_space_id = '63051408e4b0fe3d54a9864e'
alaska_collection_id = '63603f14e4b03d731ea3df55'

# ee.Authenticate()
# ee.Initialize()
from google.cloud import storage

def generate_path_for_file(filename, path_to_data):
    filename_parts = filename.split('_')
    basename = os.path.basename(filename)
    timespan = filename_parts[1]
    zone = filename_parts[2]
    path_for_file_zone = os.path.join(path_to_data, zone, timespan, 'tiles')
    if not os.path.exists(path_for_file_zone):
        print('creating path', path_for_file_zone)
        Path(path_for_file_zone).mkdir(parents=True, exist_ok=True)
    file_target_path = os.path.join(path_for_file_zone, basename)
    return file_target_path

def download_zone(site_name, start_year, end_year, path_to_data):
    storage_client = storage.Client.from_service_account_json(
        path_to_file)

    bucket = storage_client.get_bucket('pdg-landsattrend')

    substring_to_find = str(start_year) + '-' + str(end_year) + '_' + site_name

    blob_list = bucket.list_blobs()
    blobs_to_download = []
    for blob in blob_list:
        if substring_to_find in blob.name:
            blobs_to_download.append(blob)
            download_location = generate_path_for_file(blob.name, path_to_data)
            if os.path.exists(download_location):
                print(download_location, 'is already downloaded')
            else:
                print(download_location, 'downloading now...')
                blob.download_to_filename(download_location)


if __name__ == "__main__":
    download_zone('32608', 2000, 2020, path_to_data='/Users/helium/ncsa/pdg/landsattrend2/landsattrend/data')