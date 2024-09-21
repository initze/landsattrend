import ee#, eemont
import sys
import os
from pathlib import Path
import argparse

service_account = 'pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com'
path_to_file = os.path.join(os.getcwd(), 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
credentials = ee.ServiceAccountCredentials(service_account, path_to_file)
ee.Initialize(credentials)
print(f"After authentication")


print(f"After url and key")
landsat_space_id = '63051408e4b0fe3d54a9864e'
alaska_collection_id = '63603f14e4b03d731ea3df55'

# ee.Authenticate()
# ee.Initialize()
from google.cloud import storage

DEFAULT_DOWNLOAD_DIR = '/scratch/bbou/landsat-delta/landsattrend/data'


def get_download_dir(zone):
    download_directory = '/scratch/bbou/toddn/landsat-delta/landsattrend/data/' + zone + '/2000-2020/tiles'
    return download_directory

def get_download_location(filename, dir=DEFAULT_DOWNLOAD_DIR):
    components = filename.split('_')
    timespan = components[1]
    zone = components[2]
    download_directory = dir + "/" + zone + '/' + timespan + '/tiles'
    if not os.path.exists(download_directory):
        print(f"Creating download dir {download_directory}")
        os.makedirs(download_directory)
    download_location = os.path.join(download_directory, filename)
    return download_location

def generate_path_for_file(filename, path_to_data):
    filename_parts = filename.split('_')
    timespan = filename_parts[1]
    zone = filename_parts[2]
    path_for_file_zone = os.path.join(path_to_data, zone, timespan, 'tiles')
    if not os.path.exists(path_for_file_zone):
        print('creating path', path_for_file_zone)
        Path(path_for_file_zone).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        path_to_file)
    parser = argparse.ArgumentParser()

    #
    parser.add_argument("--startyear", help="The start year", default="2000")
    parser.add_argument("--endyear", help="The end year", default="2020")
    parser.add_argument("--zone", help="The UTM zone", default="32607")
    parser.add_argument("--download_dir", help="The download dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.parse_args()

    args = parser.parse_args()
    print(args, 'are args')
    if 'zone' in vars(args):
        if vars(args)['zone'] is not None:
            print("We have a process site")
            ZONE = vars(args)['zone']
    if 'startyear' in vars(args):
        if vars(args)['startyear'] is not None:
            print("We have a start year")
            startyear_value = str(vars(args)['startyear'])
            STARTYEAR = startyear_value
    if 'endyear' in vars(args):
        if vars(args)['endyear'] is not None:
            print("We have an end year")
            endyear_value = str(vars(args)['endyear'])
            ENDYEAR = endyear_value
    if 'download_dir' in vars(args):
        if vars(args)['download_dir'] is not None:
            DOWNLOAD_DIR = vars(args)['download_dir']

    SERACH_STRING = STARTYEAR + '-' + ENDYEAR + '_' + ZONE
    print(f"The serach string is {SERACH_STRING}")
    # Make an authenticated API request
    bucket = storage_client.get_bucket('pdg-landsattrend')

    blob_list = bucket.list_blobs()


    # download the files
    file_blobs = []
    bucket_file_names = []
    for blob in blob_list:
        current_full_filename = blob.name
        current_full_filename_parts = current_full_filename.split('/')
        current_filename = current_full_filename_parts[-1]
        if SERACH_STRING in current_filename:
            print("we will download", current_filename)
            download_location = get_download_location(current_filename, dir=DEFAULT_DOWNLOAD_DIR)
            blob = bucket.get_blob(blob.name)
            print('downloading to', download_location)
            blob.download_to_filename(download_location)
            print(f"Does the download location exist?")
            print(os.path.exists(download_location))
            print(f"How large is the file?")
            print(os.stat(download_location).st_size)
    print("Finished downloading from cloud")
