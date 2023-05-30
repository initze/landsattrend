import os
import sys
import pyclowder
import pyclowder.datasets
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
url = sys.argv[1]
key = sys.argv[2]
current_zone = sys.argv[3]
missing_file = current_zone + '.txt'
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

def upload_data_path(path_to_file, space_id, dataset_name, url, key):
    dataset_id = get_matching_dataset_in_space(space_id, dataset_name)
    if dataset_id:
        file_id = upload_a_file_to_dataset(filepath=path_to_file, dataset_id=dataset_id,clowder_url=url, user_api=key)
        return file_id
    return None

def upload_process_path(path_to_file, space_id, dataset_name, url, key):
    path_parts = path_to_file.split('/')
    foldername = path_parts[-2]
    dataset_id = get_matching_dataset_in_space(space_id, dataset_name)
    if dataset_id:
        folder_id = create_or_get_folder(dataset_id, folder_name=foldername)
        file_id = upload_a_file_to_dataset_with_folder(filepath=path_to_file, dataset_id=dataset_id, folder_id=folder_id,clowder_url=url, user_api=key)
        return file_id
    return None

def upload_path(path_to_file, space_id, dataset_name, url, key):
    file_id = None
    if 'data' in path_to_file:
        file_id = upload_data_path(path_to_file=path_to_file, space_id=space_id, dataset_name=dataset_name, url=url, key=key)
    if 'process' in path_to_file:
        file_id = upload_process_path(path_to_file=path_to_file, space_id=space_id, dataset_name=dataset_name, url=url, key=key)
    return file_id


matching_dataset = get_matching_dataset_in_space(space_id=landsat_space_id, dataset_name=current_zone)
matching_dataset_id = matching_dataset['id']

files_in_dataset = client.get('/datasets/' + matching_dataset_id + '/files')
dataset_folders = client.get('/datasets/' + matching_dataset_id + '/folders')

current_dir = os.getcwd()
path_to_file = os.path.join(current_dir, missing_file)

with open(path_to_file, 'r') as f:
    lines = f.readlines()

index_of_line = lines.index('these files were not uploaded\n')
start_index = index_of_line + 2

paths_to_upload = []

for i in range(start_index, len(lines)):
    current_path = lines[i].rstrip('\n')
    paths_to_upload.append(current_path)

print('uploading paths')
for p in paths_to_upload:
    print(p)
    file_id = upload_path(path_to_file=p, space_id=landsat_space_id, dataset_name=matching_dataset, url=url, key=key)
    print(file_id)

