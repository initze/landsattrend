
import pyclowder.datasets
import os
import requests
import sys
import subprocess

url = sys.argv[1]
userkey = sys.argv[2]
dataset_id = sys.argv[3]
index = int(sys.argv[4])


client = pyclowder.datasets.ClowderClient(host=url, key=userkey)

def download_file_to_location(file_url, file_location):
    r = requests.get(file_url, stream=True)
    if r.ok:
        with open(file_location, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:
        pass



dataset = client.get('/datasets/'+dataset_id)
name = dataset['name']
zone = name
dataset_files = client.get('/datasets/' + dataset_id + '/files')

# create download directory

download_directory = '/scratch/bbki/toddn/landsat-delta/landsattrend/data/' + zone + '/2000-2020/tiles'

if not os.path.isdir(download_directory):
    subprocess.run(['mkdir', download_directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

counter = 0
print('doing range', index, index+10)
for i in range(0, len(dataset_files)):
    print('index is now', i)
    if i < len(dataset_files):
        print('total files is', len(dataset_files))
        print('doing index', i)
        f = dataset_files[i]
        print('we are doing files', i)
        print('current file', f['filename'])
        file_id = f['id']
        filename = f['filename']
        location_to_download = os.path.join(download_directory, filename)
        try:
            print('the location it should be is', location_to_download)
            print("downloading the file now")
            download_url = url + '/api/files/' + file_id + '/blob?key=' + userkey
            download_file = requests.get(download_url)
            download_file_to_location(download_url, location_to_download)
        except Exception as e:
            print(e)

print('done')

