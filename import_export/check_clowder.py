import os
import sys
import pyclowder
import pyclowder.datasets

url = sys.argv[1]
key = sys.argv[2]
current_zone = sys.argv[3]
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
    file_name = file['filename']
    file_size = int(file['size'])
    file_dict[file_name] = file_size

current_dir = os.path.join(os.getcwd())
data_dir = current_dir.replace('import_export', 'data')
process_dir = current_dir.replace('import_export', 'process')
current_input = os.path.join(data_dir, current_zone, '2000-2020', 'tiles')
current_output = os.path.join(process_dir, current_zone)

print('getting the number of tiles')

input_files = os.listdir(current_input)
files_uploaded_correctly = []
files_too_small = []
files_too_big = []
files_not_uploaded = []
files_not_on_disk = []
for input_file in input_files:
    if input_file in file_dict:
        input_file_path = os.path.join(current_input, input_file)
        if os.path.exists(input_file_path):
            input_file_size = os.path.getsize(input_file_path)
            if file_dict[input_file] == input_file_size:
                files_uploaded_correctly.append(input_file)
            elif file_dict[input_file] < input_file_size:
                files_too_small.append(input_file_path)
            else:
                files_too_big.append(input_file_path)
        else:
            files_not_on_disk.append(input_file)

    else:
        files_not_uploaded.append(input_file)


print('checking each part of the process dir')
process_folders = os.listdir(current_output)

for folder in process_folders:
    print('current folder is', folder)
    path_to_folder = os.path.join(current_output, folder)
    output_files = os.listdir(path_to_folder)
    for output_file in output_files:
        if output_file in file_dict:
            output_file_path = os.path.join(current_output, output_file)
            if os.path.exists(output_file_path):
                output_file_size = os.path.getsize(output_file_path)
                if file_dict[output_file] == output_file_size:
                    files_uploaded_correctly.append(output_file_path)
                elif file_dict[output_file] < output_file_size:
                    files_too_small.append(output_file_path)
                else:
                    files_too_big.append(output_file_path)
            else:
                files_not_on_disk.append(output_file)
        else:
            files_not_uploaded.append(output_file_path)

print('printing summary')
print('files uploaded that are the right size', len(files_uploaded_correctly))
print('files that are larger on clowder', len(files_too_big))
print('the following files are too small, they cannot have all the bytes')
for each in files_too_small:
    print(each)

print('the following files were not uploaded')
for each in files_not_uploaded:
    print(each)
print('the files not on disk')
for f in files_not_on_disk:
    print(f)
print('done')