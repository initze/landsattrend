import os
import sys
import numpy as np

region = sys.argv[1]
start_year = sys.argv[2]
end_year = sys.argv[3]

year_span = start_year + '-' + end_year

current_dir = os.getcwd()

slurm_jobs_dir = os.path.join(os.getcwd(), 'slurm-processing/jobs')

print('creating directory')
os.makedirs(slurm_jobs_dir, exist_ok=True)

sample_slurm_file = 'start_gpu_job_test_file.sbatch.py'

command = 'python /scratch/bbou/toddn/landsat-delta/landsattrend/07-02_LakeAnalysis_Z056_local.py --current_site_name=SITENAME --startyear=STARTYEAR --endyear=ENDYEAR --process_root=/scratch/bbou/toddn/landsat-delta/landsattrend'

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

X_START = regions[region]['X_MIN_START']
X_END = regions[region]['X_MIN_END']

print('create a directory for these slurm files')


region_names = list(regions.keys())
for x in range(X_START, X_END):
    current_site = get_zone(x)
    if current_site not in sites_to_run:
        sites_to_run.append(current_site)

for site in sites_to_run:
    site_name = str(site)

    with open('start_gpu_job_test_file.sbatch', 'r') as f:
        content = f.read()
    new_content = None

    if command in content:
        old_command = str(command)
        old_command = old_command.replace('SITENAME', site_name)
        old_command = old_command.replace('STARTYEAR', start_year)
        old_command = old_command.replace('ENDYEAR', end_year)
        new_content = content.replace(command, old_command)

    if new_content:
        file_name = 'start_gpu_job_' + site_name + '_' + year_span + '.sbatch'
        file_path = os.path.join(slurm_jobs_dir, file_name)
        with open(file_path, 'w') as f2:
            f2.write(new_content)
