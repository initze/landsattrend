import os
import sys
import numpy as np

site_name = sys.argv[1]
start_year = sys.argv[2]
end_year = sys.argv[3]

year_span = start_year + '-' + end_year

current_dir = os.getcwd()

slurm_jobs_dir = os.getcwd()


sample_slurm_file = os.path.join(os.getcwd(),'start_gpu_job_test_file.sbatch')
print('does the file exist?', sample_slurm_file)
print(os.path.exists(sample_slurm_file))

command = 'python /scratch/bbou/toddn/landsat-delta/landsattrend/07-02_LakeAnalysis_Z056_local.py --current_site_name=SITENAME --startyear=STARTYEAR --endyear=ENDYEAR --process_root=/scratch/bbou/toddn/landsat-delta/landsattrend'

def get_utmzone_from_lon(lon):
    return int(31 + np.floor(lon / 6))


def epsg_from_utmzone(utm):
    return f'326{utm:02d}'

def get_zone(lon):
    utm = get_utmzone_from_lon(lon)
    zone = epsg_from_utmzone(utm)
    return zone

with open(sample_slurm_file, 'r') as f:
    content = f.read()
new_content = None
print('the content is', content)
print('the command is', command)
if command in content:
    print('command is in the content')
    old_command = str(command)
    print('the old command', old_command)
    old_command = old_command.replace('SITENAME', site_name)
    old_command = old_command.replace('STARTYEAR', start_year)
    old_command = old_command.replace('ENDYEAR', end_year)
    new_content = content.replace(command, old_command)
    print("the new content", new_content)

if new_content:
    file_name = 'start_gpu_job_' + site_name + '_' + year_span + '.sbatch'
    file_path = os.path.join(slurm_jobs_dir, file_name)
    print('making file with name', file_name, file_path)
    with open(file_path, 'w') as f2:
        f2.write(new_content)
