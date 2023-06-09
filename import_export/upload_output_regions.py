import requests
import pyclowder
import numpy as np
import pyclowder.datasets
import os
import sys
from requests_toolbelt.multipart.encoder import MultipartEncoder


url = sys.argv[1]
key = sys.argv[2]
current_region = sys.argv[3]
path_to_process = sys.argv[4]
landsat_space_id = sys.argv[5]
print('arguments are', sys.argv[:])


# sample path is
# /scratch/bbou/toddn/landsat-delta/landsattrend/process

# these are the regions you can get

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 150, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}


def get_utmzone_from_lon(lon):
    return int(31 + np.floor(lon / 6))


def epsg_from_utmzone(utm):
    return f'326{utm:02d}'


def get_zone(lon):
    utm = get_utmzone_from_lon(lon)
    zone = epsg_from_utmzone(utm)
    return zone

def get_zones_for_region(region_name):
    region_boundaries = regions[region_name]
    X_START = region_boundaries['X_MIN_START']
    X_END = region_boundaries['X_MIN_END']
    start_zone = get_zone(X_START)
    end_zone = get_zone(X_END)
    region_zones = []
    for i in range(int(start_zone), int(end_zone)+1):
        current_zone = str(i)
        region_zones.append(current_zone)

    return region_zones

client = pyclowder.datasets.ClowderClient(host=url, key=key)

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


def create_or_get_dataset_in_collection(dataset_name, collection_id, space_id):
    print(f"get or create dataset {dataset_name} in collection {collection_id} in space{space_id}")
    datasets_in_collection = client.get('/collections/'+collection_id+'/datasets')

    ds = None

    if len(datasets_in_collection) > 0:
        for dataset in datasets_in_collection:
            if dataset["name"] == dataset_name:
                dataset_spaces = dataset["spaces"]
                if space_id in dataset_spaces:
                    return dataset
                else:
                    data = dict()
                    data['space'] = space_id
                    result = client.post('/spaces/'+space_id+'/addDatasetToSpace/'+dataset["id"], content=data)
                    return dataset
    if ds is None:
        data = dict()
        data["name"] = dataset_name
        data["description"] = ''
        if space_id:
            data["space"] = [space_id]
        if collection_id:
            data["collection"] = [collection_id]
        result = client.post("/datasets/createempty", content=data, params=data)
        return result

def create_or_get_dataset(dataset_name, space_id):
    # get datasets in space
    datasets_in_space = client.get('/spaces/' + space_id + '/datasets')
    matching_dataset = None
    for ds in datasets_in_space:
        if ds['name'] == dataset_name:
            matching_dataset = ds
    if matching_dataset is None:
        data = dict()
        data["name"] = dataset_name
        data["description"] = dataset_name
        if space_id:
            data["space"] = [space_id]
        result = client.post("/datasets/createempty", content=data, params=data)
        return result
    else:
        return matching_dataset

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

def get_region_for_filename(filename, regions=regions):
    filename_parts = filename.split('_')
    X_coord = int(filename_parts[3])
    Y_coord = int(filename_parts[4].rstrip('.tif'))
    region_names = regions.keys()
    for region_name in region_names:
        Y_MIN_START = regions[region_name]['Y_MIN_START']
        Y_MIN_END = regions[region_name]['Y_MIN_END']
        X_MIN_START = regions[region_name]['X_MIN_START']
        X_MIN_END = regions[region_name]['X_MIN_END']
        if X_MIN_START <= X_coord <= X_MIN_END:
            if Y_MIN_START <= Y_coord <= Y_MIN_END:
                return region_name
    return None

def get_region_for_filename(filename, regions=regions):
    filename_parts = filename.split('_')
    X_coord = int(filename_parts[3])
    Y_coord = int(filename_parts[4].rstrip('.tif'))
    region_names = regions.keys()
    for region_name in region_names:
        Y_MIN_START = regions[region_name]['Y_MIN_START']
        Y_MIN_END = regions[region_name]['Y_MIN_END']
        X_MIN_START = regions[region_name]['X_MIN_START']
        X_MIN_END = regions[region_name]['X_MIN_END']
        if X_MIN_START <= X_coord <= X_MIN_END:
            if Y_MIN_START <= Y_coord <= Y_MIN_END:
                return region_name
    return None

def process_input_dir(path_to_input):
    parts = path_to_input.split('/')
    site = parts[-1]
    site_dataset = create_or_get_dataset(site, space_id=landsat_space_id)
    print(site_dataset, 'is the current dataset')
    site_dataset_id = site_dataset['id']
    files = os.listdir(path_to_input)
    for f in files:
        path_to_file = os.path.join(path_to_input, f)
        file_id = upload_a_file_to_dataset(filepath=path_to_file, dataset_id=site_dataset_id, clowder_url=url, user_api=key)
        print('uploaded file', file_id)

def process_input_dir(site_name, path_to_input):
    site_dataset = create_or_get_dataset(site_name, space_id=landsat_space_id)
    print(site_dataset, 'is the current dataset')
    site_dataset_id = site_dataset['id']
    files = os.listdir(path_to_input)
    for f in files:
        path_to_file = os.path.join(path_to_input, f)
        file_id = upload_a_file_to_dataset(filepath=path_to_file, dataset_id=site_dataset_id, clowder_url=url,
                                           user_api=key)
        print('uploaded file', file_id)

def process_output_dir(site_name, path_to_output):
    site_dataset = create_or_get_dataset(site_name, space_id=landsat_space_id)
    print(site_dataset, 'is the current dataset')
    site_dataset_id = site_dataset['id']
    folders = os.listdir(path_to_output)
    for folder in folders:
        path_to_folder = os.path.join(path_to_output, folder)
        clowder_folder = create_or_get_folder(dataset_id=site_dataset_id, folder_name=folder)
        clowder_folder_id = clowder_folder['id']
        files = os.listdir(path_to_folder)
        for f in files:
            path_to_file = os.path.join(path_to_folder, f)
            print('the path to file is', path_to_file)
            file_id = upload_a_file_to_dataset_with_folder(filepath=path_to_file, dataset_id=site_dataset_id, folder_id=clowder_folder_id, clowder_url=url, user_api=key)
            print('uploaded file', file_id)

if __name__ == '__main__':
    region_zones = get_zones_for_region(region_name=current_region)
    print(region_zones)
    for zone in region_zones:
        current_output = os.path.join(path_to_process, zone)
        if os.path.exists(current_output):
            process_output_dir(site_name=zone, path_to_output=current_output)


