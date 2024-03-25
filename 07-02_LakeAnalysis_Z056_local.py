from landsattrend.lake_analysis import LakeMaker
import os, platform
import shutil
import sys
import argparse

STARTYEAR = 0
ENDYEAR = 0
PROCESS_ROOT = ""
CURRENT_SITE_NAME = ""
CLASS_PERIOD = ""

# SET THESE FROM ARGPARSE
parser = argparse.ArgumentParser()

parser.add_argument("--process_root", help="The process root for the script, the data dir location")
parser.add_argument("--startyear", help="The start year")
parser.add_argument("--endyear", help="The end year")
parser.add_argument("--current_site_name", help="The CURRENT_SITE_NAME")

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

if STARTYEAR != 0 and ENDYEAR != 0:
    CLASS_PERIOD = str(STARTYEAR) + '-' + str(ENDYEAR)




def set_conda_gdal_paths():
    if platform.system() == 'Windows':
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
        os.environ['GDAL_PATH'] = os.path.join(os.environ['CONDA_PREFIX'], 'Scripts')
    else:
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'bin')
        os.environ['GDAL_PATH'] = os.environ['GDAL_BIN']

process_dir = os.path.join(PROCESS_ROOT, 'process', CLASS_PERIOD)
print('the process dir is', process_dir)
site_name = CURRENT_SITE_NAME
CLASS_MODEL = os.path.join(PROCESS_ROOT, 'models', 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z')
LAKE_FILTER_MODEL = os.path.join(PROCESS_ROOT, 'models', '20180820_lakefilter_12039samples_py3.z')

DEM_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'dem', 'DEM.vrt')
FOREST_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'forestfire', 'forestfire.vrt')

def main():
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


if __name__ == "__main__":
    if STARTYEAR == 0 or ENDYEAR == 0 or PROCESS_ROOT == "" or CURRENT_SITE_NAME == "":
        print("Not enough arguments to run script, ending")
    else:
        main()
