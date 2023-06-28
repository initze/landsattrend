import os
import requests
import pyclowder.datasets
import logging
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import pathlib
import sys
import json
import datetime
import time

clowder_host = sys.argv[1]
clowder_key = sys.argv[2]
Landsat_space_ID = '646d02d2e4b05d174c9fab1c'

client = pyclowder.datasets.ClowderClient(host= clowder_host, key=clowder_key)

def get_datasets_in_space(space_id):
    datasets = client.get('/spaces/' + space_id + '/datasets')
    return datasets

def delete_file(file_id):
    delete_url = clowder_host + '/api/files/' + file_id + '?key=' + clowder_key
    r = requests.delete(url=delete_url)
    return r

def get_file_names_in_dataset(dataset_id):
    dataset_files = client.get('/datasets/' + dataset_id + '/files')
    dataset_file_names = []
    for each in dataset_files:
        current_name = each['filename']
        dataset_file_names.append(current_name)
    return dataset_file_names

def get_files_in_datasets(dataset_id):
    dataset_files = client.get('/datasets/' + dataset_id + '/files')
    return dataset_files

def get_matching_files(filename, all_dataset_files):
    matching_files = []
    for file in all_dataset_files:
        if file['filename'] == filename:
            matching_files.append(file)
    return matching_files

if __name__ == '__main__':
    space_datasets = get_datasets_in_space(space_id=Landsat_space_ID)
    for i in range(0, len(space_datasets)):
        print('doing index', i)
        dataset = space_datasets[i]
        dataset_files = get_files_in_datasets(dataset_id=dataset['id'])
        dataset_file_names = get_file_names_in_dataset(dataset_id=dataset['id'])
        for current_file_name in dataset_file_names:
            # time.sleep(1)
            matching_files = get_matching_files(current_file_name, dataset_files)
            print('found matching files')
            largest_file = None
            largest_file_id = None
            max_size = 0
            if len(matching_files) == 1:
                print('no need to check anything')
            else:
                for file in matching_files:
                    file_size = int(file['size'])
                    if file_size > max_size:
                        largest_file = file
                        largest_file_id = file['id']
                # delete all the smaller files
                if largest_file_id != None:
                    for file in matching_files:
                        if file['id'] != largest_file_id:
                            delete_file(file_id=file['id'])
                            print('deleted the file', file['id'])
                else:
                    print('error, somehow no largest file')
