
import pyclowder.datasets
import os
import requests
import sys
import subprocess

url = sys.argv[1]
userkey = sys.argv[2]
dataset_id = sys.argv[3]




client = pyclowder.datasets.ClowderClient(host=url, key=userkey)

dataset = client.get('/datasets/'+dataset_id)
name = dataset['name']
zone = name
dataset_files = client.get('/datasets/' + dataset_id + '/files')

# create download directory

download_directory = '/scratch/bbki/toddn/landsat-delta/landsattrend/data/' + zone + '/2000-2020/tiles'
print('download directory is', download_directory)

dir_files = os.listdir(download_directory)

print('the download directory has', len(dir_files), 'files')

dir_files = os.listdir(download_directory)
print(len(dir_files))

not_in_dirlist = []

print('the dataset files', len(dataset_files))

unique_dataset_file_name = []


for i in range(0, len(dataset_files)):
    current_file = dataset_files[i]
    current_filename = current_file['filename']
    unique_dataset_file_name.append(current_filename)
    current_path = os.path.join(download_directory, current_filename)
    exists = os.path.isfile(current_path)
    print('checking if', current_filename, 'exists', exists)
    print('is it in the original list?')
    in_the_dir = current_filename in dir_files
    print(in_the_dir, 'is it in the dir')

unique_dataset_file_name = set(unique_dataset_file_name)
unique_dataset_file_name = list(unique_dataset_file_name)

print('the number of unique names', len(unique_dataset_file_name))



