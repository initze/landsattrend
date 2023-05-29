import os
import sys
import pyclowder
import pyclowder.datasets

url = sys.argv[1]
key = sys.argv[2]
current_zone = sys.argv[3]
client = pyclowder.datasets.ClowderClient(host= url, key=key)
print('starting')
landsat_space_id = '646d02d2e4b05d174c9fab1c'

def get_matching_dataset_in_space(space_id, dataset_name):
    datasets_in_space = client.get('/spaces/' + space_id + '/datasets')
    matching_dataset = None
    for ds in datasets_in_space:
        if ds['name'] == dataset_name:
            matching_dataset = ds
            return matching_dataset
    return matching_dataset

matching_dataset = get_matching_dataset_in_space(space_id=landsat_space_id, dataset_name=current_zone)
matching_dataset_id = matching_dataset['id']

files_in_dataset = client.get('/datasets/' + matching_dataset_id + '/files')
dataset_folders = client.get('/datasets/' + matching_dataset_id + '/folders')
folder_file_dict = dict()
file_dict = dict()
filenames_in_dataset = []
for file in files_in_dataset:
    file_name = file['filename']
    filenames_in_dataset.append(file_name)
    file_size = int(file['size'])
    file_dict[file_name] = file_size

print(file_dict.keys())
current_dir = os.path.join(os.getcwd())
data_dir = current_dir.replace('import_export', 'data')
process_dir = current_dir.replace('import_export', 'process')
current_input = os.path.join(data_dir, current_zone, '2000-2020', 'tiles')
current_output = os.path.join(process_dir, current_zone)

print('getting the number of tiles')

current_files = os.listdir(current_input)
files_uploaded_correctly = []
files_too_small = []
files_too_big = []
files_not_uploaded = []
files_not_on_disk = []

paths_to_check = []

files_in_input = os.listdir(current_input)


# add the inputs
for f in files_in_input:
    path_to_f = os.path.join(current_input, f)
    paths_to_check.append(path_to_f)

# add the inputs
output_folders = os.listdir(current_output)
for folder in output_folders:
    path_to_folder = os.path.join(current_output, folder)
    files_in_folder = os.listdir(path_to_folder)
    for file in files_in_folder:
        path_to_f = os.path.join(path_to_folder, file)
        paths_to_check.append(path_to_f)

files_not_uploaded = []
files_uploaded = []

# FIRST check just IN or OUT
for p in paths_to_check:
    base_filename = os.path.basename(p)
    if base_filename in filenames_in_dataset:
        files_uploaded.append(p)
    else:
        files_not_uploaded.append(p)

print('these files were not uploaded')
for f in files_not_uploaded:
    print(f)
