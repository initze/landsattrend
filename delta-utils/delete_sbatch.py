import os

current_path = '/projects/bbou/toddn/landsattrend'

contents = os.listdir(current_path)
for content in contents:
    if content.endswith('.sbatch'):
        if content != 'start_gpu_job.sbatch':
            path_to_content = os.path.join(current_path, content)
            os.path.remove(path_to_content)