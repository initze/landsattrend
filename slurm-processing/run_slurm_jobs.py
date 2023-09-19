import os
import sys

slurm_jobs_dir = os.path.join(os.getcwd(), 'jobs')

slurm_job_files = os.listdir(slurm_jobs_dir)

for i in range(0,2):
    current_job_file = slurm_job_files[i]
    command = 'sbatch ' + current_job_file
    print('doing command', command)
    os.system(command)

print('done')