from landsattrend.lake_analysis import LakeMaker
import os, platform
import shutil

PROCESS_ROOT = os.getcwd()

def add_file_to_tiles(path_to_file):
    filename = path_to_file.split('/')[-1]
    filename_parts = filename.split('_')
    current_site_name = filename_parts[2]
    current_class_period = filename_parts[1]
    current_lat = filename_parts[3].lstrip('-')
    current_lon = filename_parts[4].rstrip('.tif')

    current_tiles_directory = os.path.join(PROCESS_ROOT, 'data', current_site_name, current_class_period, 'tiles')
    new_filename = 'trendimage_'+current_site_name+'_'+current_lat+'_'+current_lon+'.tif'
    destination = os.path.join(current_tiles_directory, new_filename)
    shutil.copy(path_to_file, destination)


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

def run_lake_analuysis(path_to_tiles, current_class_period, current_site_name):
    set_conda_gdal_paths()
    tiles_directory = path_to_tiles
    tif_files = os.listdir(tiles_directory)

    print('Available Images:\n')
    for t in tif_files:
        print(t)

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

def main():
    set_conda_gdal_paths()
    tiles_directory = os.path.join(PROCESS_ROOT, 'data', site_name, CLASS_PERIOD, 'tiles')
    tif_files = os.listdir(tiles_directory)

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
    main()
