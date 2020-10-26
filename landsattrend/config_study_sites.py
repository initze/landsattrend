__author__ = 'initze'
import os
import pandas as pd
import logging

logging.getLogger('pyclowder').setLevel(logging.DEBUG)
logging.getLogger('__main__').setLevel(logging.DEBUG)

def check_director(path_to_dir):

    is_dir = os.path.isdir(path_to_dir)
    if is_dir:
        print(path_to_dir, 'is a directory')
        print('contents')
        print(os.listdir(path_to_dir))
        return True
    else:
        print(path_to_dir, 'not a directory')
        return False

DIRS = []

BASE_DIR = os.getcwd()
DIRS.append(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'dataX')
DIRS.append(DATA_DIR)
DEM_DIR = os.path.join(BASE_DIR, 'aux_data', 'dem')
DIRS.append(DEM_DIR)
PROCESSING_DIR_01 = os.path.join(BASE_DIR, 'processing')
PROCESSING_DIR_02 = os.path.join(BASE_DIR, 'processing')
VECTOR_DIR = os.path.join(BASE_DIR, 'vector')
DIRS.append(VECTOR_DIR)
# TODO changes if we use docker, then in home
RESULT_DIR = os.path.join(BASE_DIR,'home', 'data')


for each in DIRS:
    exists = check_director(each)
    logging.info(str(each))
    logging.info("Exists? : " + str(exists))
    print(str(each))
    print("Exists? : " + str(exists))
    print('checked')

# load study sites from configuration file
csvdir = os.path.join(os.getcwd(),'config')
csvpath = os.path.join(csvdir, 'config_study_sites.csv')
study_sites_df = pd.read_csv(csvpath)
#adapt paths

result_DIR = study_sites_df["result_dir"]

study_sites_df['fishnet_file'] = study_sites_df.apply(lambda x: os.path.join(VECTOR_DIR, os.path.basename(x['fishnet_file'])), axis=1)
study_sites_df['dem_dir'] = study_sites_df.apply(lambda x: os.path.join(DEM_DIR, os.path.basename(x['dem_dir'])), axis=1)
study_sites_df['result_dir'] = study_sites_df.apply(lambda x: os.path.join(RESULT_DIR, os.path.basename(x['result_dir'])), axis=1)

study_sites = study_sites_df.T.to_dict()
wrs2_path = os.path.join(VECTOR_DIR, 'wrs2_descending.shp')