import os
import os
import argparse, sys
import shutil

import numpy as np

def get_utmzone_from_lon(lon):
    return int(31 + np.floor(lon / 6))


def crs_from_utmzone(utm):
    return f'EPSG:326{utm:02d}'


def epsg_from_utmzone(utm):
    return f'326{utm:02d}'

def get_zone(lon):
    utm = get_utmzone_from_lon(lon)
    zone = epsg_from_utmzone(utm)
    return zone

regions = {
        'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
        'ALASKA': {'Y_MIN_START':55,'Y_MIN_END':72,'X_MIN_START':-168, 'X_MIN_END':-138},
        'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
        'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
        'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
        'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
    }

all_regions = list(regions.keys())

zones = []

for region in all_regions:
    print(f"Making file for region {region}")
    start_zone = get_zone(regions[region]['X_MIN_START'])
    end_zone = get_zone(regions[region]['X_MIN_END'])
    start_and_end = [start_zone, end_zone]
    current_zone_range = np.arange(int(min(start_and_end)), int(max(start_and_end)))
    for zone in current_zone_range:
        zones.append(str(zone))
    if start_zone not in zones:
        zones.append(start_zone)
    if end_zone not in zones:
        zones.append(end_zone)
    zones.sort()

print(f"Checking for lake change file for all zones.")


PATH_TO_PROCESS = '/scratch/bbou/toddn/landsat-delta/landsattrend/process'

zones_to_rerun = []

for zone in zones:
    path_to_zone_final = os.path.join(PATH_TO_PROCESS, '2000-2020', zone, '05_Lake_Dataset_Raster_02_final')
    if os.path.exists(path_to_zone_final):
        zone_files = os.listdir(path_to_zone_final)
        if 'lake_change.gpkg' in zone_files:
            print(f"We have a lake change file")
            path_to_geopackage = os.path.join(path_to_zone_final, 'lake_change.gpkg')
            size_of_geopackage = os.stat(path_to_geopackage).st_size
            print(f"Size of geopackage is {size_of_geopackage}")
            if size_of_geopackage == 0:
                print(f"The geopackage is of size 0")
                zones_to_rerun.append(zone)
        else:
            print(f"No geopackage file in results for zone {zone}")
            zones_to_rerun.append(zone)
    else:
        print(f"We are missing the final folder for zone {zone}")
        zones_to_rerun.append(zone)

with open('RERUN_zones.txt', 'w') as f:
    for rerun_zone in zones_to_rerun:
        f.write(rerun_zone + '\n')
print(f"Finished we have a file for rerun zones")