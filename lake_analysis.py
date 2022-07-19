from landsattrend.lake_analysis import LakeMaker
import os, platform
import shutil

PROCESS_ROOT = os.getcwd()

def set_conda_gdal_paths():
    if platform.system() == 'Windows':
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
        os.environ['GDAL_PATH'] = os.path.join(os.environ['CONDA_PREFIX'], 'Scripts')
    else:
        os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_PREFIX'], 'bin')
        os.environ['GDAL_PATH'] = os.environ['GDAL_BIN']

process_dir = os.path.join(PROCESS_ROOT, 'process')
site_name = '32604'
CLASS_PERIOD = '2000-2020'
CLASS_MODEL = os.path.join(PROCESS_ROOT, 'models', 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z')
LAKE_FILTER_MODEL = os.path.join(PROCESS_ROOT, 'models', '20180820_lakefilter_12039samples_py3.z')

DEM_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'dem', 'DEM.vrt')
FOREST_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'forestfire', 'forestfire.vrt')

def run_lake_analysis(path_to_tiles, current_class_period, current_site_name):
    print('in run lake analysis method')
    print('CLASS_MODEL', CLASS_MODEL)
    print(os.path.exists(CLASS_MODEL))
    print('LAKE_FILTER_MODEL', LAKE_FILTER_MODEL)
    print(os.path.exists(LAKE_FILTER_MODEL))
    print('DEM_LOCATION', DEM_LOCATION)
    print(os.path.exists(DEM_LOCATION))
    print('FOREST_LOCATION', FOREST_LOCATION)
    print(os.path.exists(FOREST_LOCATION))
    set_conda_gdal_paths()
    tiles_directory = path_to_tiles
    tif_files = os.listdir(tiles_directory)

    print('Available Images:\n')
    for t in tif_files:
        if t == '.DS_Store':
            path_to_ds_store = os.path.join(tiles_directory, t)
            os.remove(path_to_ds_store)
            print(t)

    tif_files = os.listdir(tiles_directory)

    print('process dir is', process_dir)
    print('current site name', current_site_name)
    print('current tiles dir', tiles_directory)
    print('is it really a dir?')
    print(os.path.isdir(tiles_directory))
    print('current class period', current_class_period)
    l = LakeMaker(site_name, os.path.join(process_dir, current_site_name), tiles_directory, classperiod=current_class_period)
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

# if __name__ == "__main__":
#     path_to_tiles = os.path.join(os.getcwd(),'home','data','32604','2000-2020','tiles')
#     run_lake_analysis(path_to_tiles=path_to_tiles, current_class_period='2000-2020',current_site_name='32604')
