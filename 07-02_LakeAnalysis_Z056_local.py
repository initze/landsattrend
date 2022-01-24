from landsattrend.lake_analysis import LakeMaker
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import geopandas as gpd

PROCESSING_DIR = os.getcwd()
tiles = [ '150_62']
zone = 'Z056-Kolyma'
directory_location = os.path.join(PROCESSING_DIR, 'process')
site_name = 'Z056-Kolyma'

CLASS_PERIOD = '2000-2020'
CLASS_MODEL = os.path.join(PROCESSING_DIR, 'models', 'PDG_6idx2feat_elslope_model_py38_sklearn0232_v04.z')
LAKE_FILTER_MODEL = os.path.join(PROCESSING_DIR, 'models', '20180820_lakefilter_12039samples_py3.z')

def main():

    tiles_directory = os.path.join(os.getcwd(), 'data', site_name, CLASS_PERIOD, 'tiles')
    tif_files = os.listdir(tiles_directory)
    for t in tif_files:
        print(t)

    l = LakeMaker(zone, os.path.join(directory_location, site_name), classperiod=CLASS_PERIOD)
    print("\nStart Classification")
    l.classify(CLASS_MODEL, tiles)
    """
	print("\nPreparing additional Data")
    l.prepare_aux_data(r'E:\18_Paper02_LakeAnalysis\02_AuxData\04_DEM\DEM.vrt',r'E:\18_Paper02_LakeAnalysis\02_AuxData\01_ForestLoss\forestfire.vrt')
    print("\nCreating Masks")
    l.make_masks()
    print("\nCalculating Stats")
    l.make_stats()
    print("\nSaving DataFrame to Disk")
    l.save_df()
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
    #print("\nSaving ResultGrid at 7.5km resolution")
    #l.export_gridded_results(250)
	"""


if __name__ == "__main__":
    main()
