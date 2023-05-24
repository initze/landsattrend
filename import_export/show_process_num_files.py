import os
import sys


current_zone = sys.argv[1]

current_dir = os.path.join(os.getcwd())
data_dir = current_dir.replace('import_export', 'data')
process_dir = current_dir.replace('import_export', 'process')
current_input = os.path.join(data_dir, current_zone, '2000-2020', 'tiles')
current_output = os.path.join(process_dir, current_zone)

print('getting the number of tiles')

input_files = os.listdir(current_input)
print(len(input_files), 'tiles')

print('checking each part of the process dir')
process_folders = os.listdir(current_output)

for folder in process_folders:
    print('current folder is', folder)
    path_to_folder = os.path.join(current_output, folder)
    files_inside = os.listdir(path_to_folder)
    print(folder, 'has', len(files_inside), 'files inside')
