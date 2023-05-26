import os
import sys
import pyclowder
import pyclowder.datasets

current_zone = sys.argv[1]
url = sys.argv[2]
key = sys.argv[3]
client = pyclowder.datasets.ClowderClient(host= url, key=key)

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
for file in files_in_dataset:
    print(file)
    file_name = file['filename']
    file_size = int(file['size'])
    print(file_size)
    file_dict[file_name] = file_size

current_dir = os.path.join(os.getcwd())
data_dir = current_dir.replace('import_export', 'data')
process_dir = current_dir.replace('import_export', 'process')
current_input = os.path.join(data_dir, current_zone, '2000-2020', 'tiles')
current_output = os.path.join(process_dir, current_zone)

print('getting the number of tiles')

input_files = os.listdir(current_input)
files_uploaded_correctly = []
files_uploaded_wrong_size = []
files_not_uploaded = []
for input_file in input_files:
    if input_file in file_dict:
        input_file_path = os.path.join(current_input, input_file)
        input_file_size = os.path.getsize(input_file_path)
        print(input_file, input_file_size, type(input_file_size), print(file_dict[input_file]))
        if input_file_size == file_dict[input_file]:
            files_uploaded_correctly.append(input_file)
        else:
            files_uploaded_wrong_size.append(input_file)
    else:
        files_not_uploaded.append(input_file)


print('checking each part of the process dir')
process_folders = os.listdir(current_output)

for folder in process_folders:
    print('current folder is', folder)
    path_to_folder = os.path.join(current_output, folder)
    files_inside = os.listdir(path_to_folder)
    for output_file in files_inside:
        if output_file in file_dict:
            output_file_path = os.path.join(path_to_folder, output_file)
            output_file_size = os.path.getsize(output_file_path)
            if output_file_size == file_dict[output_file]:
                files_uploaded_correctly.append(output_file)
            else:
                files_uploaded_wrong_size.append(output_file_path)
        else:
            files_not_uploaded.append(output_file_path)

print('printing summary')
print('files uploaded that are the right size', len(files_uploaded_correctly))
print('files uploaded, wrong size', len(files_uploaded_wrong_size))
if len(files_uploaded_wrong_size) > 0:
    for file in files_uploaded_wrong_size:
        print(file)
        print('on disk, in clowder')
        basename = os.path.basename(file)
        print(os.path.getsize(file), file_dict[basename])
print('files not uploaded at all', len(files_not_uploaded))
