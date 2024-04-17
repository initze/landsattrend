import ray
import os
import gcsfs
from google.cloud import storage
from utils.utils_processing import *

# storage_client = storage.Client.from_service_account_json(
#     path_to_file)

#
exported_files = 'exported_files.txt'
# CLOUD bucket parameters -different to the one on the bottom
outputBucket = 'pdg-landsattrend' #Change for your Cloud Storage bucket

def get_all_files_from_bucket(file_system, bucket_name):
    total_contents = []
    contents = file_system.listdir(bucket_name)
    for content in contents:
        print(type(content))
        if content['type'] == 'directory':
            current_contents = get_all_files_from_bucket(file_system, content['name'])
            if current_contents is not None:
                total_contents += current_contents
        if content['type'] == 'file':
            total_contents.append(content)
        print(content)
    return total_contents


path_to_token = os.path.join(os.getcwd(), 'data-export-gee', 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
print(os.path.exists(path_to_token))

fs = gcsfs.GCSFileSystem(project='pdg-landsattrend', token=path_to_token)
contents = fs.listdir('pdg-landsattrend')
all_contents = get_all_files_from_bucket(fs, 'pdg-landsattrend')
print('got fs')