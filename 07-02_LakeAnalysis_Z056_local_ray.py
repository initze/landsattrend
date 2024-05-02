from landsattrend.lake_analysis import LakeMaker
import os, platform
import shutil
import sys
import ray
import argparse
import time

from export_tools.download_site_from_cloud import download_zone

STARTYEAR = 0
ENDYEAR = 0
PROCESS_ROOT = ""
CURRENT_SITE_NAME = ""
CLASS_PERIOD = ""
SITE_FILE_LIST= ""

# SET THESE FROM ARGPARSE
parser = argparse.ArgumentParser()

parser.add_argument("--process_root", help="The process root for the script, the data dir location")
parser.add_argument("--startyear", help="The start year")
parser.add_argument("--endyear", help="The end year")
parser.add_argument("--current_site_name", help="The CURRENT_SITE_NAMES a comma delimited list")
parser.add_argument("--site_file_list", help="A file with the list of sites to run, useful for larger runs")

args, unknown = parser.parse_known_args()
print(f"Dict format: {vars(args)}")

if 'current_site_name' in vars(args):
    if vars(args)['current_site_name'] is not None:
        print("We have a process site")
        CURRENT_SITE_NAME = vars(args)["current_site_name"]
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
if 'process_root' in vars(args):
    if vars(args)['process_root'] is not None:
        print("We have a process root")
        PROCESS_ROOT = vars(args)["process_root"]
if 'site_file_list' in vars(args):
    if vars(args)['site_file_list'] is not None:
        SITE_FILE_LIST = vars(args)["site_file_list"]

if STARTYEAR != 0 and ENDYEAR != 0:
    CLASS_PERIOD = str(STARTYEAR) + '-' + str(ENDYEAR)





def set_conda_gdal_paths():
    if platform.system() == 'Windows':
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
        os.environ['GDAL_PATH'] = os.path.join(os.environ['CONDA_PREFIX'], 'Scripts')
    else:
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'bin')
        os.environ['GDAL_PATH'] = os.environ['GDAL_BIN']


@ray.remote
def run_lake_analysis(PROCESS_ROOT, CURRENT_SITE_NAME, CLASS_PERIOD, num_cpus, num_gpus):
    process_dir = os.path.join(PROCESS_ROOT, 'process', CLASS_PERIOD)
    print('the process dir is', process_dir)
    site_name = CURRENT_SITE_NAME
    CLASS_MODEL = os.path.join(PROCESS_ROOT, 'models', 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z')
    LAKE_FILTER_MODEL = os.path.join(PROCESS_ROOT, 'models', '20180820_lakefilter_12039samples_py3.z')
    DEM_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'dem', 'DEM.vrt')
    FOREST_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'forestfire', 'forestfire.vrt')

    set_conda_gdal_paths()
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
    return True

if __name__ == "__main__":
    ray.init()
    sites_to_run = []
    if SITE_FILE_LIST is not None:
        with open(SITE_FILE_LIST, 'r') as f:
            lines = f.readlines()
            for line in lines:
                current_site = line.rstrip('\n')
                sites_to_run.append(current_site)
    # TODO check that files exist locally, or download them
    sites_to_download = []
    for site in sites_to_run:
        path_to_site = os.path.join(PROCESS_ROOT, 'data', site, CLASS_PERIOD, 'tiles')
        has_contents = False
        if os.path.exists(path_to_site):
            contents = os.listdir(path_to_site)
            if len(contents) > 1:
                has_contents = True
        if not has_contents:
            path_to_data = os.path.join(PROCESS_ROOT, 'data')
            download_zone(site_name=site,start_year=STARTYEAR, end_year=ENDYEAR,path_to_data=path_to_data)

    # TODO if they are not in bucket, then export those zones and wait
    ray_futures = []
    for site in sites_to_run:
        current_future = run_lake_analysis.remote(PROCESS_ROOT=PROCESS_ROOT,
                             CURRENT_SITE_NAME=site, CLASS_PERIOD=CLASS_PERIOD, num_cpus=1, num_gpus=2)
        ray_futures.append(current_future)
    print(ray.get(ray_futures))

    # TODO a loop to check on these tqsks needs to be here
    print('running in ray now')