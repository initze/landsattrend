import requests
from datetime import datetime
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
import pyclowder.datasets
import sys
import json

landsat_space_id = '646d02d2e4b05d174c9fab1c'
alaska_collection_id = '63603f14e4b03d731ea3df5'

path_to_process = '/scratch/bbou/toddn/landsat-delta/landsattrend/process'


clowder_url = 'https://pdg.clowderframework.org'

key = sys.argv[1]
zone_file_path = sys.argv[2]
year_span = sys.argv[3]

base_headers = {'X-API-key': key}
headers = {**base_headers, 'Content-type': 'application/json',
           'accept': 'application/json'}

def search_dataset_folders(dataset_id, folder_name, url):
    folder_url = f"{url}/api/datasets/{dataset_id}/folders"
    dataset_folders = requests.get(folder_url, headers=headers)
    result = dataset_folders.json()
    for folder in result:
        current_folder_name = folder['name'].lstrip('/')
        if current_folder_name == folder_name:
            return folder
    return None

def search_dataset_by_name(dataset_name, url, space_id):
    matching_datasets = []
    search_dataset_url = f"{url}/api/search"
    search = requests.get(search_dataset_url, params={'query': dataset_name, 'resource_type': 'dataset'},
                          headers=headers)
    search_results = search.json()['results']
    print(f"The search results for dataset {dataset_name}")
    print(len(search_results))
    if len(search_results) > 0:
        for result in search_results:
            result_spaces = result['spaces']
            if space_id in result_spaces:
                matching_datasets.append(result)
    if len(matching_datasets) > 0:
        return matching_datasets[0]
    else:
        return None

def create_dataset(url, space, zone_name):
    dataset_name = f"{zone_name}"
    search_results = search_dataset_by_name(dataset_name=dataset_name, url=url, space_id=space)
    if search_results is None:
        create_dataset_url = f"{url}/api/datasets/createempty"
        payload = json.dumps({'name': dataset_name,
                              'description': f"Landsat data for zone {zone_name}",
                              'access': "PRIVATE",
                              'space': [space],
                              'collection': ""})
        r = requests.post(create_dataset_url,
                          data=payload,
                          headers=headers)
        r.raise_for_status()
        return r.json()["id"]
    else:
        dataset_id = search_results['id']
        return dataset_id

def is_file_in_dataset(path_to_file, dataset_id, url, key):
    file_already_in_dataset = False
    list_files_url = f"{url}/api/datasets/{dataset_id}/listAllFiles?key={key}"
    response = requests.get(list_files_url, headers=headers)
    dataset_files = response.json()
    current_upload_filename = os.path.basename(path_to_file)
    for file in dataset_files:
        if file['filename'] == current_upload_filename:
            print(f"We already uploaded the file {current_upload_filename}")
            current_size = file['size']
            current_size_clowder = int(current_size)
            size_of_upload = os.stat(path_to_file).st_size
            print(f"Size of file on clowder is {current_size_clowder}")
            print(f"Size of file locally is {size_of_upload}")
            if size_of_upload == current_size_clowder:
                file_already_in_dataset = True
    return file_already_in_dataset



def upload_a_file_to_dataset_with_folder(filepath, dataset_id, folder_name, url):
    folder = search_dataset_folders(dataset_id=dataset_id, folder_name=folder_name, url=url)
    print("result of search dataset folders is", folder)
    if folder is None:
        folder_post_url = f"{url}/api/datasets/{dataset_id}/newFolder?key={key}"
        payload = json.dumps({'name': folder_name,
                              'parentId': dataset_id,
                              'parentType': "dataset"})
        r = requests.post(folder_post_url, data=payload, headers=headers)
        r.raise_for_status()
        folder_id = r.json()["id"]
    else:
        folder_id = folder['id']

    url = '%s/api/uploadToDataset/%s?key=%s&folder_id=%s' % (url, dataset_id, key, folder_id)
    if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            m = MultipartEncoder(
                fields={'file': (filename, open(filepath, 'rb')),
                        'folder_id':folder_id}
            )
            try:
                result = requests.post(url, data=m, headers={'Content-Type': m.content_type},
                                        verify=False)
                print('upload result is', result)
                uploadedfileid = result.json()['id']
            except Exception as e:
                print('failed to upload file, error')
                print(e)
    else:
        print("unable to upload file %s (not found)", filepath)
    return uploadedfileid

def upload_a_file_to_dataset_with_folder_id(filepath, dataset_id, parentId, url):
    url = '%s/api/uploadToDataset/%s?key=%s&folder_id=%s' % (url, dataset_id, key, parentId)
    if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            m = MultipartEncoder(
                fields={'file': (filename, open(filepath, 'rb')),
                        'folder_id':parentId}
            )
            try:
                result = requests.post(url, data=m, headers={'Content-Type': m.content_type},
                                        verify=False)
                print('upload result is', result)
                uploadedfileid = result.json()['id']
                return uploadedfileid
            except Exception as e:
                return None
                print('failed to upload file, error')
                print(e)
    else:
        print("unable to upload file %s (not found)", filepath)
    return None

def upload_a_file_to_dataset(filepath, dataset_id, url):
    uploadedfileid = None
    upload_url = '%s/api/uploadToDataset/%s?key=%s' % (url, dataset_id, key)
    file_exists = os.path.exists(filepath)
    if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            m = MultipartEncoder(
                fields={'file': (filename, open(filepath, 'rb'))}
            )
            try:
                result = requests.post(upload_url, data=m, headers={'Content-Type': m.content_type},
                                        verify=False)

                uploadedfileid = result.json()['id']
                print('upload result is', result)
            except Exception as e:
                print('failed to upload file, error')
                print(e)
    else:
        print("unable to upload file %s (not found)", filepath)
    return uploadedfileid

if __name__ == "__main__":

    zones_to_upload = []

    with open(zone_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            zones_to_upload.append(line.rstrip('\n'))

    print(f"Zones to upload are {zones_to_upload}")

    path_to_current_process = os.path.join(path_to_process, year_span)

    for i in range(0, len(zones_to_upload)):
        print('doing zone', zones_to_upload[i])
        zone_name = zones_to_upload[i]
        # create dataset for that zone in space
        zone_dataset = create_dataset(url=clowder_url, space=landsat_space_id, zone_name=zone_name)
        path_to_zone_results = os.path.join(path_to_current_process, zone_name)
        zone_folders = os.listdir(path_to_zone_results)
        print('uploading these files for this zone')
        # process_folder = search_dataset_folders(dataset_id=zone_dataset, folder_name='process', url=clowder_url)
        # print('the process folder id is', process_folder, 'it already existed')
        # if process_folder is None:
        #     folder_post_url = f"{clowder_url}/api/datasets/{zone_dataset}/newFolder?key={key}"
        #     print('folder post URL', folder_post_url)
        #     payload = json.dumps({'name': 'process',
        #                           'parentId': zone_dataset,
        #                           'parentType': "dataset"})
        #     r = requests.post(folder_post_url, data=payload, headers=headers)
        #     r.raise_for_status()
        #     process_folder_id = r.json()["id"]
        #     print('process folder id is', process_folder_id,'we created it')
        # else:
        #     process_folder_id = process_folder['id']
        for folder in zone_folders:
            path_to_folder = os.path.join(path_to_zone_results, folder)
            folder_post_url = f"{clowder_url}/api/datasets/{zone_dataset}/newFolder?key={key}"
            print('folder post URL', folder_post_url)
            payload = json.dumps({'name': folder,
                                    'parentId': zone_dataset,
                                    'parentType': "dataset"})
            r = requests.post(folder_post_url, data=payload, headers=headers)
            r.raise_for_status()
            folder_id = r.json()["id"]
            files = os.listdir(path_to_folder)
            for file in files:
                path_to_file = os.path.join(path_to_folder, file)
                file_already_uploaded = is_file_in_dataset(path_to_file=path_to_file,
                                                           dataset_id=zone_dataset, key=key, url=clowder_url)
                print(f"Is the file {file} already uploaded? {file_already_uploaded}")
                if not file_already_uploaded:
                    print('uploading', path_to_file)
                    try:
                        new_file_id = upload_a_file_to_dataset_with_folder_id(path_to_file, zone_dataset, folder_id, clowder_url)
                        print('the new file id is', new_file_id)
                    except Exception as e:
                        print(f"Error uploading file")
                        print(e)
            print('done')

