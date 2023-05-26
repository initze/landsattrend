import os
import shutil

path_to_landsattrend = '/scratch/bbou/toddn/landsat-delta/landsattrend'

path_to_projects = '/projects/bbou/toddn'

all_files_in_dir = os.listdir(path_to_landsattrend)
for file in all_files_in_dir:
    print(file)