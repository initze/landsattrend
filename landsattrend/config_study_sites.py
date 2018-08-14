__author__ = 'initze'
import os
import pandas as pd


DATA_DIR = r'N:\petacarbnob1\petacarb\initze\01_Data_Archive\02_Processed_Masked'
DEM_DIR = r'D:\01_RasterData\01_Landsat\04_DEM'
PROCESSING_DIR_01 = r'D:\01_RasterData\01_Landsat\02_Processed'
PROCESSING_DIR_02 = r'K:\01_RasterData\01_Landsat\02_Processed'
VECTOR_DIR = r'D:\05_Vector'
RESULT_DIR = r'E:\06_Trendimages'

# load study sites from configuration file
csvdir = os.path.join(os.environ['HOMEDRIVE'], os.environ['HOMEPATH'], 'landsattrend')
csvpath = os.path.join(csvdir, 'config_study_sites.csv')
study_sites = pd.DataFrame.from_csv(csvpath).T.to_dict()

wrs2_path = os.path.join(VECTOR_DIR, 'wrs2_descending.shp')