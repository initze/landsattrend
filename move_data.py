import os
import shutil

from pathlib import Path
path_to_data = os.path.join(os.getcwd(), 'data')

path_to_files_to_move = '/Users/helium/Desktop/landsattrend-test/data/tiles'


files_to_move = os.listdir(path_to_files_to_move)

print('got files to move')

for file in files_to_move:
    filename_parts = file.split('_')
    year_span = filename_parts[1]
    zone = filename_parts[2]
    path_to_create = os.path.join(path_to_data, zone, year_span, 'tiles')
    Path(path_to_create).mkdir(parents=True, exist_ok=True)
    print('create this path')
    path_to_file = os.path.join(path_to_files_to_move, file)
    target = os.path.join(path_to_create, file)
    shutil.copy2(path_to_file, target)
    print('copied')