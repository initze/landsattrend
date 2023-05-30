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

matching_dataset = get_matching_dataset_in_space(space_id=landsat_space_id, dataset_name=current_zone)
matching_dataset_id = matching_dataset['id']

files_in_dataset = client.get('/datasets/' + matching_dataset_id + '/files')
dataset_folders = client.get('/datasets/' + matching_dataset_id + '/folders')

current_dir = os.getcwd()
path_to_file = os.path.join(current_dir, missing_file)

with open(path_to_file, 'r') as f:
    lines = f.readlines()

index_of_line = lines.index('these files were not uploaded')
print(index_of_line, 'is index')


