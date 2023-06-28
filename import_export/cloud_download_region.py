import os
import ee
import sys
from google.cloud import storage
from pathlib import Path
folder = sys.argv[1]
data_directory = sys.argv[2]

service_account = "pdg-landsattrend@uiuc-ncsa-permafrost.iam.gserviceaccount.com"
path_to_file = os.path.join(os.getcwd(), 'project-keys', 'uiuc-ncsa-permafrost-44d44c10c9c7.json')
credentials = ee.ServiceAccountCredentials(service_account, path_to_file)
storage_client = storage.Client.from_service_account_json(
    path_to_file)
ee.Initialize(credentials)

download_directory = os.path.join(os.getcwd())

def download_file(bucketName, filename, download_location, check_bucket=False, overwrite=False):
    bucket = storage_client.get_bucket(bucketName)
    if check_bucket:
        # TODO this should not happen
        print('we are not sure the file is really there')
    else:
        blob = bucket.get_blob(filename)
        filename_parts = filename.split('/')
        base_filename = filename_parts[-1]
        filename_parts = base_filename.split('_')
        site_name = filename_parts[2]
        class_range = filename_parts[1]
        path_to_tiles = os.path.join(download_location, site_name, class_range, 'tiles')
        if not os.path.exists(path_to_tiles):
            path = Path(path_to_tiles)
            path.mkdir(parents=True)
        path_to_file = os.path.join(path_to_tiles, base_filename)
        if not os.path.exists(path_to_file):
            print('downloading the file to', path_to_file)
            blob.download_to_filename(path_to_file)
        else:
            size_in_bytes = bucket.get_blob(filename).size
            size_of_file_on_disk = os.path.getsize(path_to_file)
            if size_in_bytes > size_of_file_on_disk:
                print('downloading the file to', path_to_file)
                blob.download_to_filename(path_to_file)
            else:
                print('no need to download', filename)

def download_folder(bucketName, folder, download_location):
    bucket = storage_client.get_bucket(bucketName)
    blob_list = bucket.list_blobs()
    filenames_to_download = []
    for blob in blob_list:
        current_filename = blob.name
        if '/' in current_filename:
            if current_filename.startswith(folder+'/') and current_filename != folder+'/':
                download_file(bucketName, current_filename, download_location, check_bucket=False)
    return blob_list

if __name__ == "__main__":
    print(download_directory)
    print(data_directory)
    folder_on_cloud = sys.argv[1]
    download_folder('pdg-landsattrend', folder, data_directory)

