import os

all_files = os.listdir(os.getcwd())

existing_sbatch_files = ['start_gpu_job.sbatch', 'start_gpu_job.sbatch', 'start_gpu_job_test.sbatch']

for file in all_files:
    print('checking file', file)
    if file.endswith('.sbatch'):
        print('this ends with sbatch', file)
        if file not in existing_sbatch_files:
            current_path = os.path.join(os.getcwd(), file)
            print('deleting file', current_path)
            os.remove(current_path)