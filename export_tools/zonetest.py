import numpy as np

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}

def get_utmzone_from_lon(lon):
    return int(31 + np.floor(lon/ 6))

def crs_from_utmzone(utm):
    return f'EPSG:326{utm:02d}'

def epsg_from_utmzone(utm):
    return f'326{utm:02d}'

def prefix_from_utmzone(utm):
    return f'trendimage_Z{utm:02d}'

def get_zone(lon):
    utm = get_utmzone_from_lon(lon)
    zone = epsg_from_utmzone(utm)
    return zone

all_zones = []

zone = get_zone(-169)
for region in regions:
    X_MIN_START = regions[region]['X_MIN_START']
    X_MIN_END = regions[region]['X_MIN_END']
    lons = np.arange(X_MIN_START, X_MIN_END)
    for lon in lons:
        current_zone = get_zone(lon)
        if current_zone not in all_zones:
            all_zones.append(current_zone)
    print('got start and end')
print('got zone')
all_zones.sort()
with open('all_zones.txt', 'w') as f:
    for each in all_zones:
        f.write(each + '\n')
print('we sorted')