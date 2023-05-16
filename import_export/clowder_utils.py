import requests
import pyclowder
import pyclowder.datasets
import os
import sys
import datetime
from requests_toolbelt.multipart.encoder import MultipartEncoder


url = sys.argv[1]
key = sys.argv[2]
landsat_space_id = '63051408e4b0fe3d54a9864e'

path_to_data = os.path.join(os.getcwd())

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 150, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}

def get_region_for_filename(filename, regions=regions):
    filename_parts = filename.split('_')
    X_coord = int(filename_parts[3])
    Y_coord = int(filename_parts[4].rstrip('.tif'))
    region_names = regions.keys()
    for region_name in region_names:
        Y_MIN_START = regions[region_name]['Y_MIN_START']
        Y_MIN_END = regions[region_name]['Y_MIN_END']
        X_MIN_START = regions[region_name]['X_MIN_START']
        X_MIN_END = regions[region_name]['X_MIN_END']
        if X_MIN_START <= X_coord <= X_MIN_END:
            if Y_MIN_START <= Y_coord <= Y_MIN_END:
                return region_name
    return None

