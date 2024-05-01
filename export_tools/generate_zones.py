import numpy as np

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

def get_zones(min_lat, max_lat, min_lon, max_lon):
    zones = dict()
    for lon in range(min_lon, max_lon):
        current_zone_name = get_zone(lon)
        if current_zone_name not in zones:
            zone = dict()
            zone['min_lon'] = lon
            zone['max_lon'] = lon
            zone['min_lat'] = min_lat
            zone['max_lat'] = max_lat
            zones[current_zone_name] = zone
        else:
            zones[current_zone_name]['max_lon'] = lon
    return zones

def get_filename_to_upload(X_MIN, Y_MIN, ZONE, STARTYEAR, ENDYEAR):
    print('getting filename for', X_MIN, Y_MIN, ZONE)
    file_name = f'trendimage_' + str(STARTYEAR) + '-' + str(ENDYEAR) + '_' + ZONE + '_' + str(X_MIN) + '_' + str(Y_MIN)
    file_name = file_name + '.tif'
    return file_name

def get_filenames_to_upload(Y_MIN_START, Y_MIN_END, X_MIN_START, X_MIN_END, STARTYEAR, ENDYEAR):
    filenames_to_upload = []
    for Y_MIN in range(Y_MIN_START, Y_MIN_END):
        for X_MIN in range(X_MIN_START, X_MIN_END, 3):
            CURRENT_ZONE = get_zone(X_MIN)
            filename_to_upload = get_filename_to_upload(X_MIN, Y_MIN, CURRENT_ZONE, STARTYEAR, ENDYEAR)
            filenames_to_upload.append(filename_to_upload)
    return filenames_to_upload