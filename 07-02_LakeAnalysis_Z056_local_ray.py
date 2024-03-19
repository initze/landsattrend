from landsattrend.lake_analysis import LakeMaker
import os, platform
import shutil
import sys
import ray

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

if __name__ == "__main__":
    ray.init()
    # futures = [run_lake_analysis.remote(PROCESS_ROOT='/Users/helium/ncsa/pdg/landsattrend2/landsattrend',
    #                          CURRENT_SITE_NAME='32602', CLASS_PERIOD='2000-2020', num_cpus=1, num_gpus=2)]
    futures = [run_lake_analysis.remote(PROCESS_ROOT='/Users/helium/ncsa/pdg/landsattrend2/landsattrend',
                             CURRENT_SITE_NAME='32602', CLASS_PERIOD='2000-2020', num_cpus=1, num_gpus=2),
    run_lake_analysis.remote(PROCESS_ROOT='/Users/helium/ncsa/pdg/landsattrend2/landsattrend',
                             CURRENT_SITE_NAME='32603', CLASS_PERIOD='2000-2020', num_cpus=1, num_gpus=2)]

    print(ray.get(futures))
    print('here at end')

