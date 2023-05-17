import os
import sys

site_list = sys.argv[1]
site_names = site_list.split(',')
print('generating file for site name', site_names)

for site_name in site_names:
    old_command = 'python /scratch/bbou/toddn/landsat-delta/landsattrend/07-02_LakeAnalysis_Z056_local.py SITE_NAME'

    with open('start_gpu_job.sbatch', 'r') as f:
        content = f.read()
    new_content = None

    if old_command in content:
        new_command = old_command.replace('SITE_NAME', site_name)
        new_content = content.replace(old_command, new_command)

    if new_content:
        file_name = 'start_gpu_job_' + site_name + '.sbatch'
        with open(file_name, 'w') as f2:
            f2.write(new_content)
