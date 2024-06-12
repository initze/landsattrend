import os
import sys
import pyclowder
import time
import generate_zones
import upload_utils
import argparse

DATA_DIR = '/scratch/bbou/toddn/landsat-delta/landsattrend/data'
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
parser.add_argument("--zone", help="The PROCESS_SITE")
parser.add_argument("--download_dir", help="The download dir")
parser.add_argument("--clowder_url", help="clowder url")
parser.add_argument("--api_key", help="the api key")

parser.parse_args()

args=parser.parse_args()

CLOWDER_URL = None
API_KEY = None
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
current_collection = upload_utils.find_collection_if_exists_for_timespan(timespan=timespan)
if current_collection is None:
    current_collection = upload_utils.create_collection(collection_name=timespan, space_id=LANDSATTREND_SPACE_ID)


path_to_current_tiles = os.path.join(DATA_DIR, current_zone, timespan, 'tiles')
print('path to current tiles', print(path_to_current_tiles))
print('current tiles')
current_dataset_name = current_zone + '_' + timespan
print('dataset name', current_dataset_name)
current_dataset = upload_utils.create_or_get_dataset(dataset_name=current_dataset_name, space_id=LANDSATTREND_SPACE_ID)
current_dataset_id = current_dataset['id']
# add dataset to collection
upload_utils.add_dataset_to_collection(current_dataset_id, collection_id=current_collection['id'])
current_tiles = os.listdir(path_to_current_tiles)
for tile in current_tiles:
    path_to_current_tile = os.path.join(path_to_current_tiles, tile)
    print('trying to upload this file', path_to_current_tile)
    current_file_id = upload_utils.upload_a_file_to_dataset(filepath=path_to_current_tile,dataset_id=current_dataset_id, clowder_url=CLOWDER_URL, user_api=API_KEY)
    print('uploaded file', current_file_id)




