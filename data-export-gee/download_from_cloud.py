import ee#, eemont
import sys
import os
from pathlib import Path
import generate_zones
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

DOWNLOAD_DIR = '/scratch/bbou/landsat-delta/landsattrend/data'


def get_download_dir(zone):
    download_directory = '/scratch/bbou/toddn/landsat-delta/landsattrend/data/' + zone + '/2000-2020/tiles'
    return download_directory

def get_download_location(filename, dir=DOWNLOAD_DIR):
    components = filename.split('_')
    timespan = components[1]
    zone = components[2]
    download_directory = dir + "/" + zone + '/' + timespan + '/tiles'
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
    parser.add_argument("--startyear", help="The start year")
    parser.add_argument("--endyear", help="The end year")
    parser.add_argument("--process_site", help="The PROCESS_SITE")
    parser.add_argument("--download_dir", help="The download dir")
    parser.parse_args()

    args = parser.parse_args()
    print(args, 'are args')
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
    if 'download_dir' in vars(args):
        if vars(args)['download_dir'] is not None:
            DOWNLOAD_DIR = vars(args)['download_dir']

    regions = {
        'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
        'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
        'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
        'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
        'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
        'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
    }

    # Make an authenticated API request
    bucket = storage_client.get_bucket('pdg-landsattrend')

    blob_list = bucket.list_blobs()

    if PROCESS_SITE == 'ALL':
        files_to_download = []
        print('we are downloading for ALL REGIONS')
        for region in regions:
            CURRENT_PROCESS_SITE = region
            start_zone = generate_zones.get_zone(regions[CURRENT_PROCESS_SITE]['X_MIN_START'])
            end_zone = generate_zones.get_zone(regions[CURRENT_PROCESS_SITE]['X_MIN_END'])

            print('the download dir', DOWNLOAD_DIR)
            print('does the download dir exists?')
            does_exist = os.path.exists(DOWNLOAD_DIR)
            print(does_exist)
            print('start zone and end zone', start_zone, end_zone)
            print('start year and end year', STARTYEAR, ENDYEAR)
            print('region', PROCESS_SITE)
            print('getting files to download...')

            new_files_to_download = generate_zones.get_filenames_to_upload(Y_MIN_START=regions[CURRENT_PROCESS_SITE]["Y_MIN_START"],
                                                                       Y_MIN_END=regions[CURRENT_PROCESS_SITE]["Y_MIN_END"],
                                                                       X_MIN_START=regions[CURRENT_PROCESS_SITE]["X_MIN_START"],
                                                                       X_MIN_END=regions[CURRENT_PROCESS_SITE]["X_MIN_END"],
                                                                       STARTYEAR=STARTYEAR,
                                                                       ENDYEAR=ENDYEAR)
            files_to_download = files_to_download + new_files_to_download
    else:
        start_zone = generate_zones.get_zone(regions[PROCESS_SITE]['X_MIN_START'])
        end_zone = generate_zones.get_zone(regions[PROCESS_SITE]['X_MIN_END'])

        print('the download dir', DOWNLOAD_DIR)
        print('does the download dir exists?')
        does_exist = os.path.exists(DOWNLOAD_DIR)
        print(does_exist)
        print('start zone and end zone', start_zone, end_zone)
        print('start year and end year', STARTYEAR, ENDYEAR)
        print('region', PROCESS_SITE)
        print('getting files to download...')

        files_to_download = generate_zones.get_filenames_to_upload(Y_MIN_START=regions[PROCESS_SITE]["Y_MIN_START"],
                                                                    Y_MIN_END=regions[PROCESS_SITE]["Y_MIN_END"],
                                                                    X_MIN_START=regions[PROCESS_SITE]["X_MIN_START"],
                                                                    X_MIN_END=regions[PROCESS_SITE]["X_MIN_END"],
                                                                    STARTYEAR=STARTYEAR,
                                                                    ENDYEAR=ENDYEAR)
    print('we will download', len(files_to_download))
    # create folders for the specific files
    for file in files_to_download:
        print('downloading file', file)
        generate_path_for_file(file, path_to_data=DOWNLOAD_DIR)

    # download the files
    file_blobs = []
    bucket_file_names = []
    for blob in blob_list:
        current_full_filename = blob.name
        current_full_filename_parts = current_full_filename.split('/')
        current_filename = current_full_filename_parts[-1]
        if current_filename in files_to_download:
            print("we will download", current_filename)
            download_location = get_download_location(current_filename, dir=DOWNLOAD_DIR)
            blob = bucket.get_blob(blob.name)
            print('downloading to', download_location)
            blob.download_to_filename(download_location)
    print("Finished downloading from cloud")
