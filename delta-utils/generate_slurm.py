import os
import sys
import numpy as np
import argparse

# PARSING ARGUMENTS IF ANY

parser = argparse.ArgumentParser()

parser.add_argument("--process_root", help="The process root for the script, the data dir location")
parser.add_argument("--startyear", help="The start year")
parser.add_argument("--endyear", help="The end year")
parser.add_argument("--process_site", help="The PROCESS_SITE")

args, unknown = parser.parse_known_args()

print(f"Dict format: {vars(args)}")


def get_utmzone_from_lon(lon):
    return int(31 + np.floor(lon / 6))


def epsg_from_utmzone(utm):
    return f'326{utm:02d}'


def get_zone(lon):
    utm = get_utmzone_from_lon(lon)
    zone = epsg_from_utmzone(utm)
    return zone


sites_to_run = []

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 153, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}

region_names = list(regions.keys())
for region in region_names:
    X_START = regions[region]['X_MIN_START']
    X_END = regions[region]['X_MIN_END']
    for x in range(X_START, X_END):
        current_site = get_zone(x)
        if current_site not in sites_to_run:
            sites_to_run.append(current_site)

for site in sites_to_run:
    site_name = str(site)
    old_command = 'python /scratch/bbou/toddn/landsat-delta/landsattrend/07-02_LakeAnalysis_Z056_local.py ' \
                  '--process_root=PROCESS_ROOT' \
                  '--startyear=STARTYEAR' \
                  '--endyear=ENDYEAR' \
                  '--current_site_name=CURRENT_SITE_NAME'



    parser.add_argument("--process_root", help="The process root for the script, the data dir location")
    parser.add_argument("--startyear", help="The start year")
    parser.add_argument("--endyear", help="The end year")
    parser.add_argument("--current_site_name", help="The CURRENT_SITE_NAME")

    with open('start_gpu_job.sbatch', 'r') as f:
        content = f.read()
    new_content = None

    if old_command in content:
        new_command = old_command.replace('CURRENT_SITE_NAME', site_name)
        new_content = content.replace(old_command, new_command)

    if new_content:
        file_name = 'start_gpu_job_' + site_name + '.sbatch'
        with open(file_name, 'w') as f2:
            f2.write(new_content)
