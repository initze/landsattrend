import os
import sys
import shutil

site_name = sys.argv[1]

path_to_data = os.path.join('/projects/bbou/toddn/landsattrend/process', site_name)
destination = os.path.join('/scratch/bbou/toddn/landsat-delta/landsattrend/process/', site_name)
print(path_to_data)
print('move to')
print(destination)
shutil.copytree(path_to_data, destination)
print('done')