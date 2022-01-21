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
study_sites_df = pd.read_csv(csvpath)
#adapt paths
study_sites_df['fishnet_file'] = study_sites_df.apply(lambda x: os.path.join(VECTOR_DIR, os.path.basename(x['fishnet_file'])), axis=1)
study_sites_df['dem_dir'] = study_sites_df.apply(lambda x: os.path.join(DEM_DIR, os.path.basename(x['dem_dir'])), axis=1)
study_sites_df['result_dir'] = study_sites_df.apply(lambda x: os.path.join(RESULT_DIR, os.path.basename(x['result_dir'])), axis=1)

fishnet_file_DIR = study_sites_df['fishnet_file'][0]
dem_dir_DIR = study_sites_df['dem_dir'][0]
results_DIR = study_sites_df['result_dir'][0]

study_sites = study_sites_df.T.to_dict()
wrs2_path = os.path.join(VECTOR_DIR, 'wrs2_descending.shp')
print('done')