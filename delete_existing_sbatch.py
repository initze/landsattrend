import os

all_files = os.listdir(os.getcwd())

for file in all_files:
    print('checking file', file)
    if file.endswith('.sbatch'):
        print('this ends with sbatch', file)
        if file != 'start_gpu_job.sbatch' and file != 'start_gpu_job_test.batch':
            if file != 'start_gpu_test_file.sbatch':
                current_path = os.path.join(os.getcwd(), file)
                print('deleting file', current_path)
                os.remove(current_path)