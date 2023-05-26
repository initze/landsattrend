import os
import sys
import shutil

site_name = sys.argv[1]

path_to_data = os.path.join('/scratch/bbou/toddn/landsat-delta/landsattrend/data/', site_name)
print(path_to_data)

destination = os.path.join('/projects/bbou/toddn/landsatrend/data', site_name)
print(destination)
shutil.copytree(path_to_data, destination)