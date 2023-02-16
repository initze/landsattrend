import requests
from datetime import datetime
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
import pyclowder.datasets
import sys

clowder_host = sys.argv[1]
clowder_key = sys.argv[2]
dataset_id = sys.argv[3]
path_to_results = sys.argv[4]

client = pyclowder.datasets.ClowderClient(host=clowder_host, key=clowder_key)

def upload_a_file_to_dataset_with_folder(filepath, dataset_id, folder_id):
    url = '%sapi/uploadToDataset/%s?key=%s&folder_id=%s' % (clowder_host, dataset_id, clowder_key, folder_id)
    print('the url')
    print(url)
    file_exists = os.path.exists(filepath)
    print('starting upload')
    print(str(datetime.now()))
    before = datetime.now()
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
                print(str(datetime.now()))
                after = datetime.now()
                duration = after - before
                print(str(duration))
            except Exception as e:
                print('failed to upload file, error')
                print(e)
                print(str(datetime.now()))
    else:
        print("unable to upload file %s (not found)", filepath)

def upload_a_file_to_dataset(filepath, dataset_id, clowder_url, user_api):
    url = '%s/api/uploadToDataset/%s?key=%s' % (clowder_url, dataset_id, user_api)
    file_exists = os.path.exists(filepath)
    before = datetime.now()
    if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            m = MultipartEncoder(
                fields={'file': (filename, open(filepath, 'rb'))}
            )
            try:
                result = requests.post(url, data=m, headers={'Content-Type': m.content_type},
                                        verify=True)

                uploadedfileid = result.json()['id']
                return uploadedfileid
            except Exception as e:
                print('failed to upload file, error')
                print(e)
                print(str(datetime.now()))
    else:
        print("unable to upload file %s (not found)", filepath)
    return None

subfolders = os.listdir(path_to_results)

# create a folder for PROCESS

data = dict()
data["name"] = 'process'
data["parentId"] = dataset_id
data["parentType"] = "dataset"
process_folder = client.post('/datasets/' + dataset_id + '/newFolder', content=data, params=data)
print('created dataset with name process, dataset id is', process_folder)

# create a folder for each dir under process, add files
for subfolder in subfolders:
    subfolder_data = dict()
    subfolder_data["name"] = subfolder
    subfolder_data["parentId"] = process_folder
    subfolder_data["parentType"] = "folder"
    current_folder = client.post('/datasets/' + dataset_id + '/newFolder', content=subfolder_data, params=subfolder_data)
    # go through each file in the subfolder
    path_to_subfolder = os.path.join(path_to_results, subfolder)
    subfolder_contents = os.listdir(path_to_subfolder)
    for item in subfolder_contents:
        path_to_item = os.path.join(path_to_subfolder, item)
        print('uploading', path_to_item, 'dataset', dataset_id, 'folder', current_folder)
        upload_a_file_to_dataset_with_folder(path_to_item, dataset_id, current_folder)






