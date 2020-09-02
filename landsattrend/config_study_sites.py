__author__ = 'initze'
import os
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'dataX')
DEM_DIR = os.path.join(BASE_DIR, 'aux_data', 'dem')
PROCESSING_DIR_01 = os.path.join(BASE_DIR, 'processing')
PROCESSING_DIR_02 = os.path.join(BASE_DIR, 'processing')
VECTOR_DIR = os.path.join(BASE_DIR, 'vector')
RESULT_DIR = os.path.join(BASE_DIR, 'data')

# load study sites from configuration file
csvdir = os.path.join(os.getcwd(),'config')
csvpath = os.path.join(csvdir, 'config_study_sites.csv')
study_sites = pd.DataFrame.from_csv(csvpath).T.to_dict()

wrs2_path = os.path.join(VECTOR_DIR, 'wrs2_descending.shp')