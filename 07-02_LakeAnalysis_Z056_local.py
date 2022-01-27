from landsattrend.lake_analysis import LakeMaker
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import geopandas as gpd

PROCESS_ROOT = os.getcwd()

tiles = ['150_62']
process_dir = os.path.join(PROCESS_ROOT, 'process')
site_name = 'Z056-Kolyma'

CLASS_PERIOD = '2000-2020'
CLASS_MODEL = os.path.join(PROCESS_ROOT, 'models', 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z')
LAKE_FILTER_MODEL = os.path.join(PROCESS_ROOT, 'models', '20180820_lakefilter_12039samples_py3.z')

os.environ['GDAL_BIN'] = ''
os.environ['GDAL_PATH'] = ''
#os.environ['GDAL_BIN'] = os.path.join(os.environ['CONDA_Prefix'], 'Library','bin')
#os.environ['GDAL_PATH'] = os.path.join(os.environ['CONDA_Prefix'], 'Scripts') #in linux its the same as $GDAL_BIN

DEM_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'dem', 'DEM.vrt')
FOREST_LOCATION = os.path.join(PROCESS_ROOT, r'aux_data', 'forestfire', 'forestfire.vrt')

def main():

    tiles_directory = os.path.join(PROCESS_ROOT, 'data', site_name, CLASS_PERIOD, 'tiles')
    tif_files = os.listdir(tiles_directory)

    print('Available Images:\n')
    for t in tif_files:
        print(t)

    l = LakeMaker(site_name, os.path.join(process_dir, site_name), classperiod='2000-2020')
    print("\nStart Classification")
    l.classify(CLASS_MODEL, tiles)

    print("\nPreparing additional Data")
    # l.prepare_aux_data(r'E:\18_Paper02_LakeAnalysis\02_AuxData\04_DEM\DEM.vrt',r'E:\18_Paper02_LakeAnalysis\02_AuxData\01_ForestLoss\forestfire.vrt')
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
