import os
import sys
import pyclowder
import time
import generate_zones
import upload_utils
import argparse

PROCESS_DIR = '/scratch/bbou/toddn/landsat-delta/landsattrend/process'
LANDSATTREND_SPACE_ID = '646d02d2e4b05d174c9fab1c'

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}

parser=argparse.ArgumentParser()

parser.add_argument("--startyear", help="The start year")
parser.add_argument("--endyear", help="The end year")
parser.add_argument("--process_site", help="The PROCESS_SITE")
parser.add_argument("--download_dir", help="The download dir")
parser.add_argument("--clowder_url", help="clowder url")
parser.add_argument("--api_key", help="the api key")

parser.parse_args()

args=parser.parse_args()

CLOWDER_URL = None
ZONE = None
STARTYEAR = None
ENDYEAR = None

if 'zone' in vars(args):
    if vars(args)['zone'] is not None:
        print("We have a process site")
        ZONE = vars(args)['zone']
if 'startyear' in vars(args):
    if vars(args)['startyear'] is not None:
        print("We have a start year")
        startyear_value = int(vars(args)['startyear'])
        STARTYEAR = startyear_value
if vars(args)['endyear'] is not None:
    print("We have an end year")
    endyear_value = int(vars(args)['endyear'])
    ENDYEAR = endyear_value
    if vars(args)['endyear'] is not None:
        print("We have an end year")
        endyear_value = int(vars(args)['endyear'])
        ENDYEAR = endyear_value

if 'clowder_url' in vars(args):
    if vars(args)['clowder_url'] is not None:
        CLOWDER_URL = vars(args)['clowder_url']
if 'api_key' in vars(args):
    if vars(args)['api_key'] is not None:
        API_KEY = vars(args)['api_key']
print('ZONE', ZONE)
print("STARTYEAR", STARTYEAR)
print("ENDYEAR", ENDYEAR)

timespan = str(STARTYEAR) + '-' + str(ENDYEAR)

current_zone = ZONE


path_to_zone_process = os.path.join(PROCESS_DIR, timespan, current_zone)
contents = os.listdir(path_to_zone_process)
# check that the dataset is created, it should already exist
dataset_name = current_zone + '_' + timespan
current_dataset = upload_utils.create_or_get_dataset(dataset_name=dataset_name, space_id=LANDSATTREND_SPACE_ID)
current_dataset_id = current_dataset['id']
for folder in contents:
    folder_path = os.path.join(path_to_zone_process, folder)
    folder_contents = os.listdir(folder_path)
    for folder_item in folder_contents:
        path_to_item = os.path.join(folder_path, folder_item)
        new_file_id = upload_utils.upload_a_file_to_dataset_with_folder(filepath=path_to_item, dataset_id=current_dataset_id, folder_name=folder,url=CLOWDER_URL)
        print('uploaded', path_to_item, new_file_id)
print('finished')
