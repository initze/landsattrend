import os
import sys
import pyclowder
import pyclowder.datasets
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
url = sys.argv[1]
key = sys.argv[2]
current_zone = sys.argv[3]
client = pyclowder.datasets.ClowderClient(host= url, key=key)
print('starting')
landsat_space_id = '646d02d2e4b05d174c9fab1c'

def create_or_get_folder(dataset_id, folder_name):
    # get datasets in space
    ds_folders = client.get('/datasets/' + dataset_id + '/folders')
    for folder in ds_folders:
        if folder is not None:
            current_folder_name = folder['name'].lstrip('/')
            if current_folder_name == folder_name:
                return folder
    data = dict()
    data["name"] = folder_name
    data["parentType"] = "dataset"
    data["parentId"] = dataset_id
    new_folder = client.post("/datasets/" + dataset_id + '/newFolder', content=data, params=data)
    return new_folder

def upload_a_file_to_dataset(filepath, dataset_id, clowder_url, user_api):
    url = '%s/api/uploadToDataset/%s?key=%s' % (clowder_url, dataset_id, user_api)
    file_exists = os.path.exists(filepath)
    if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            m = MultipartEncoder(
                fields={'file': (filename, open(filepath, 'rb'))}
            )
            try:
                result = requests.post(url, data=m, headers={'Content-Type': m.content_type},
                                        verify=False)

                uploadedfileid = result.json()['id']
                return uploadedfileid
            except Exception as e:
                print('failed to upload file, error')
                print(e)
    else:
        print(f"unable to upload file %s (not found) {filepath}")
    return None


def upload_a_file_to_dataset_with_folder(filepath, dataset_id, folder_id, clowder_url, user_api):
    url = '%s/api/uploadToDataset/%s?key=%s&folder_id=%s' % (clowder_url, dataset_id, user_api, folder_id)
    print('the url')
    print(url)
    file_exists = os.path.exists(filepath)
    print('starting upload')
    if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            m = MultipartEncoder(
                fields={'file': (filename, open(filepath, 'rb')),
                        'folder_id':folder_id}
            )
            try:
                result = requests.post(url, data=m, headers={'Content-Type': m.content_type},
                                        verify=False)

                print(result)
                uploadedfileid = result.json()['id']
                print("uploaded file", uploadedfileid)
            except Exception as e:
                print('failed to upload file, error')
                print(e)
    else:
        print("unable to upload file %s (not found)", filepath)



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

bigger_on_clowder = []
smaller_on_clowder = []
print('print files uploaded that are not same size on clowder')
for f in files_uploaded:
    current_file_size = os.path.getsize(f)
    base_filename = os.path.basename(f)
    size_on_clowder = file_dict[base_filename]
    if size_on_clowder != current_file_size:
        message = base_filename + ',' + str(current_file_size) + ',' + str(size_on_clowder)
        if size_on_clowder > current_file_size:
            bigger_on_clowder.append(message)
        else:
            smaller_on_clowder.append(message)
print('files bigger on clowder')
for b in bigger_on_clowder:
    print(b)

print('files smaller on clowder')
for s in smaller_on_clowder:
    print(s)

print('these files were not uploaded')
print(len(files_not_uploaded))
for f in files_not_uploaded:
    print(f)