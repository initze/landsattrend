import requests
import pyclowder
import pyclowder.datasets
import os
import sys
import datetime
from requests_toolbelt.multipart.encoder import MultipartEncoder


url = sys.argv[1]
key = sys.argv[2]
landsat_space_id = '63051408e4b0fe3d54a9864e'

path_to_data = os.path.join(os.getcwd())

regions = {
    'TEST': {'Y_MIN_START': 62, 'Y_MIN_END': 64, 'X_MIN_START': 150, 'X_MIN_END': 156},
    'ALASKA': {'Y_MIN_START': 55, 'Y_MIN_END': 72, 'X_MIN_START': -168, 'X_MIN_END': -138},
    'CANADA': {'Y_MIN_START': 50, 'Y_MIN_END': 80, 'X_MIN_START': -141, 'X_MIN_END': -54},
    'EURASIA1': {'Y_MIN_START': 55, 'Y_MIN_END': 71, 'X_MIN_START': 18, 'X_MIN_END': 63},
    'EURASIA2': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': 66, 'X_MIN_END': 177},
    'EURASIA3': {'Y_MIN_START': 55, 'Y_MIN_END': 80, 'X_MIN_START': -180, 'X_MIN_END': -169},
}

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


def find_dataset_if_exists(url, key, dataset_name):
    client = pyclowder.datasets.ClowderClient(host=url, key=key)
    matching_datasets = []
    search_results = client.get('/search', params={'query': dataset_name, 'resource_type': 'dataset'})
    results = search_results['results']
    if len(results) > 0:
        return results
    else:
        return None

def find_collection_if_exists(url, key, collection_name, space_id=""):
    client = pyclowder.datasets.ClowderClient(host=url, key=key)
    search_results = client.get('/search', params={'query': collection_name, 'resource_type': 'collection'})
    results = search_results['results']
    if len(results) == 0:
        return None
    else:
        for result in results:
            current_name = result['collectionname']
            if current_name == collection_name:
                return result
    return None

def create_or_get_dataset(dataset_name, collection_id, space_id):
    client = pyclowder.datasets.ClowderClient(host=url, key=key)
    print('get or create dataset', dataset_name, collection_id, space_id)
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
                                        verify=False)

                uploadedfileid = result.json()['id']
                return uploadedfileid
            except Exception as e:
                print('failed to upload file, error')
                print(e)
                print(str(datetime.now()))
    else:
        print("unable to upload file %s (not found)", filepath)
    return None


def upload_site_name(site_name, collection_id=None):
    # TODO create a dataset for the zone
    current_dir = os.getcwd()
    data_dir = current_dir.replace('import_export', 'data')
    zone_dir = os.path.join(data_dir, site_name)
    files_to_upload = []
    for path, subdirs, files in os.walk(zone_dir):
        for name in files:
            filepath_to_upload = os.path.join(path, name)
            print(filepath_to_upload)
            files_to_upload.append(filepath_to_upload)
            region = get_region_for_filename(name)
            # does the collection exist for this region?
            result = find_collection_if_exists(url, key,region,space_id=landsat_space_id)
            if result is None:
                # TODO create the collection
                print('no collection, we should create it')
            print('result')


    # TODO upload the files

    # TODO put the dataseet in the right collection
    print('done')

def upload_process(zone_name, dataset_id):
    pass

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir = current_dir.replace('import_export', 'data')
    upload_site_name('32656')