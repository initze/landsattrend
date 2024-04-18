import ray
import os
import gcsfs
from google.cloud import storage
from utils.utils_processing import *
import time
from landsattrend.lake_analysis import LakeMaker
import os, platform
import shutil
import sys
import ray

@ray.remote
def download_data(FS, ENTRY, num_cpus, num_gpus):
    print(FS, ENTRY)
    print('downloading')

# storage_client = storage.Client.from_service_account_json(
#     path_to_file)

#
# TODO rewrite this and all the steps using ray datasets, model after maple_V3
@ray.remote
def run_lake_analysis(PROCESS_ROOT, CURRENT_SITE_NAME, CLASS_PERIOD, num_cpus, num_gpus):
    process_dir = os.path.join(PROCESS_ROOT, 'process', CLASS_PERIOD)
    print('the process dir is', process_dir)
    site_name = CURRENT_SITE_NAME
    CLASS_MODEL = os.path.join(PROCESS_ROOT, 'models', 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z')
    LAKE_FILTER_MODEL = os.path.join(PROCESS_ROOT, 'models', '20180820_lakefilter_12039samples_py3.z')
    DEM_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'dem', 'DEM.vrt')
    FOREST_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'forestfire', 'forestfire.vrt')

    if platform.system() == 'Windows':
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
        os.environ['GDAL_PATH'] = os.path.join(os.environ['CONDA_PREFIX'], 'Scripts')
    else:
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'bin')
        os.environ['GDAL_PATH'] = os.environ['GDAL_BIN']
    print('the process root is', PROCESS_ROOT)
    tiles_directory = os.path.join(PROCESS_ROOT, 'data', site_name, CLASS_PERIOD, 'tiles')
    tif_files = os.listdir(tiles_directory)

    if '.DS_Store' in tif_files:
        tif_files.remove('.DS_Store')
    print('Available Images:\n')
    for t in tif_files:
        print(t)

    l = LakeMaker(site_name, os.path.join(process_dir, site_name), tiles_directory, classperiod=CLASS_PERIOD)
    print("\nStart Classification")
    l.classify(CLASS_MODEL)

    print("\nPreparing additional Data")
    l.prepare_aux_data(DEM_LOCATION, FOREST_LOCATION)

    print("\nCreating Masks")
    l.make_masks()

    print("\nCalculating Stats")
    l.make_stats()

    print("\nSaving DataFrame to Disk")
    l.save_df()

    # errors come somewhere here
    print("\nFiltering non-lake objects")
    l.filter_data(LAKE_FILTER_MODEL)
    print("\nSaving DataFrame to Disk")
    l.save_filtered_data()
    print("\nTransforming data to metric values")
    l.finalize_calculations()
    print("\nSaving DataFrame to Disk")
    l.save_results()
    print("\nSaving ResultGrid at 3km resolution")
    l.export_gridded_results([100, 250])


## https://stackoverflow.com/questions/20478369/how-do-you-get-or-generate-a-url-to-the-object-in-a-bucket
exported_files = 'exported_files.txt'
# CLOUD bucket parameters -different to the one on the bottom
outputBucket = 'pdg-landsattrend' #Change for your Cloud Storage bucket

START_YEAR = 2000
END_YEAR = 2020
YEAR_SPAN = str(START_YEAR) + '-' + str(END_YEAR)
REGION = 'TEST'
ZONE_1 = '32655'
ZONE_2 = '32656'

def get_all_files_from_bucket(file_system, bucket_name):
    total_contents = []
    contents = file_system.listdir(bucket_name)
    for content in contents:
        if content['type'] == 'directory':
            current_contents = get_all_files_from_bucket(file_system, content['name'])
            if current_contents is not None:
                total_contents += current_contents
        if content['type'] == 'file':
            total_contents.append(content)
    return total_contents

if __name__ == "__main__":
    ray.init()
    path_to_token = os.path.join(os.getcwd(), 'data-export-gee', 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
    print(os.path.exists(path_to_token))

    fs = gcsfs.GCSFileSystem(project='pdg-landsattrend', token=path_to_token)
    contents = fs.listdir('pdg-landsattrend')
    all_contents = get_all_files_from_bucket(fs, 'pdg-landsattrend')

    entries_to_run = []

    zone_1 = str(START_YEAR) + '-' + str(END_YEAR) + '_' + ZONE_1
    zone_2 = str(START_YEAR) + '-' + str(END_YEAR) + '_' + ZONE_2

    ZONES = [ZONE_1, ZONE_2]

    for entry in all_contents:
        name = entry['name']
        if entry['type'] == 'file':
            if zone_2 in name or zone_1 in name:
                entries_to_run.append(entry)
        else:
            print(entry, 'was not a file')

    print('got fs')

    # download the files from one zone for demo purposes
    # note - you can comment this out if they are already downloaded
    # TODO need to check if the files exist locally and are the same size
    print('now we need to download the data')
    for entry in entries_to_run:
        file_path = entry['name']
        file_path_parts = file_path.split('/')
        file_name = file_path_parts[-1]
        file_name_parts = file_name.split('_')
        years = file_name_parts[1]
        zone = file_name_parts[2]
        path_to_write = os.path.join(os.getcwd(), 'data', zone, years, 'tiles')
        file_path_to_write = os.path.join(path_to_write, file_name)
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write, exist_ok=True)
        if not os.path.exists(file_path_to_write):
            with fs.open(file_path, 'rb') as f:
                data = f.read()
                if not os.path.exists(file_path_to_write):
                    print("we will download file", file_name)
                    with open(file_path_to_write, 'wb') as f2:
                        f2.write(data)
        else:
            print('we already downloaded file', file_name)

    # TODO for zone in zones run entry
    print("we downloaded zones now")
    print("processing")
    ray_futures = []
    current_process_root = os.path.join(os.getcwd())
    for ZONE in ZONES:
        future = run_lake_analysis.remote(PROCESS_ROOT=current_process_root,
                                            CURRENT_SITE_NAME=ZONE, CLASS_PERIOD=YEAR_SPAN, num_cpus=1,
                                            num_gpus=2)
        ray_futures.append(future)

    print("checking futures at regular intervals")
    for i in range(0, 1000):
        print("we have futures")
        print(ray.get(ray_futures))
        time.sleep(30)
    print("done sleeping")
