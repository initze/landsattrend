import logging
import subprocess
import zipfile
from pyclowder.extractors import Extractor
import pyclowder.files
import pyclowder.datasets
from pathlib import Path
import os
import platform
import os
import sys
import requests
import pathlib
import shutil
import time
import json
import extractor_lake_analysis
import distutils

PROCESS_ROOT = os.getcwd()

HOME_DIR = os.path.join(os.getcwd(), 'home')

def add_file_to_tiles(path_to_file):
    filename = path_to_file.split('/')[-1]
    filename_parts = filename.split('_')
    current_site_name = filename_parts[2]
    current_class_period = filename_parts[1]
    current_lat = filename_parts[3].lstrip('-')
    current_lon = filename_parts[4].rstrip('.tif')

    current_tiles_directory = os.path.join(PROCESS_ROOT, 'home','data', current_site_name, current_class_period, 'tiles')
    if not os.path.isdir(current_tiles_directory):
        try:
            tiles_path = Path(current_tiles_directory)
            tiles_path.mkdir(parents=True)
        except Exception as e:
            print(e)
    if os.path.isdir(current_tiles_directory):
        new_filename = 'trendimage_'+current_site_name+'_'+current_lat+'_'+current_lon+'.tif'
        destination = os.path.join(current_tiles_directory, new_filename)
        shutil.copy(path_to_file, destination)
        try:
            os.remove(path_to_file)
        except Exception as e:
            print(e)

def move_files_in_dataset(path_to_unzipped_dataset):
    print('here')
    contents = os.listdir(path_to_unzipped_dataset)
    for item in contents:
        if item.endswith('.tif'):
            try:
                add_file_to_tiles(os.path.join(path_to_unzipped_dataset, item))
            except Exception as e:
                print(e)
        else:
            try:
                os.remove(os.path.join(path_to_unzipped_dataset, item))
            except Exception as e:
                print(e)


def delete_unnecessary_files(path_to_unzipped_dataset):
    contents = os.listdir(path_to_unzipped_dataset)
    for item in contents:
        if os.path.isdir(os.path.join(path_to_unzipped_dataset, item)):
            if item == 'metadata':
                try:
                    os.rmtree(os.path.join(path_to_unzipped_dataset, item))
                except Exception as e:
                    print(e)
        else:
            try:
                os.remove(os.path.join(path_to_unzipped_dataset, item))
            except Exception as e:
                print(e)



class LandsattrendExtractor(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        # add any additional arguments to parser
        # self.parser.add_argument('--max', '-m', type=int, nargs='?', default=-1,
        #                          help='maximum number (default=-1)')

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the dataset

        logger = logging.getLogger(__name__)

        dataset_id = resource["id"]
        dataset_name = resource["name"]
        files = resource["files"]

        logger.info("in process message")

        # dataset_download_location = os.path.join(HOME_DIR, dataset_name)
        #
        # download = pyclowder.datasets.download(connector, host, secret_key, dataset_id)
        # with zipfile.ZipFile(download, 'r') as zip:
        #     zip.extractall(dataset_download_location)

        # delete everything except the files
        path_to_dataset = os.path.join(HOME_DIR, dataset_name)
        dataset_contents = os.listdir(path_to_dataset)
        path_to_dataset_data = os.path.join(HOME_DIR, dataset_name, 'data')
        path_to_dataset_metadata = os.path.join(HOME_DIR, dataset_name, 'metadata')

        delete_unnecessary_files(path_to_dataset)






        # move files to somewhere else

        print('dataset is unzipped')
        print(os.listdir(HOME_DIR))
        return 0
        # path_to_landsattrend_data = os.path.join(dataset_download_location, 'data', 'data', 'Z056-Kolyma')
        # path_to_data_folder_in_dataset = os.path.join(dataset_download_location, 'data', 'data')
        # logger.info("path to landsattrend data")
        # logger.info(str(path_to_landsattrend_data))
        # if os.path.isdir(path_to_landsattrend_data):
        #     logger.info("We have landsattrend data")
        #     if os.path.isdir('/home/data'):
        #         logger.info("we have old data, removing")
        #         shutil.rmtree('/home/data')
        #     else:
        #         shutil.move(path_to_data_folder_in_dataset, '/home/data')
        # logger.info("run_lake_analysis now, after pause")
        # contents_of_tiles = os.listdir('/home/data/Z056-Kolyma/1999-2019/tiles')
        # logger.info("contents of tiles")
        # logger.info(str(contents_of_tiles))
        # time.sleep(60*6)
        # return 0
        # tiles_in_folder = get_tiles_from_files('/home/data/Z056-Kolyma/1999-2019/tiles')
        # logger.info('tiles_in_folder')
        # logger.info(str(tiles_in_folder))
        # if len(tiles_in_folder) > 1:
        #     location_of_final_dataset = run_lake_analysis.process_tiles(tiles_in_folder)
        #     logger.info("final dataset location")
        #     logger.info(str(location_of_final_dataset))
        #
        #     upload_ids = []
        #
        #     contents_of_final_dataset = os.listdir(str(location_of_final_dataset))
        #     for each_file in contents_of_final_dataset:
        #         try:
        #             current_file_path = os.path.join(location_of_final_dataset, each_file)
        #             current_id = pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, filepath=current_file_path)
        #             upload_ids.append(current_id)
        #         except Exception as e:
        #             logger.info("Could not upload : " + str(current_file_path))
        #
        #     logger.info("We have uploaded : " + str(len(upload_ids)))
        #     logger.info("These files have ids : " + str(upload_ids))
        #      # TODO upload to dataset as files, then move to new folder
        #
        #     params = dict()
        #     headers = {'content-type': 'application/json'}
        #     params['key'] = secret_key
        #     params['parentId'] = dataset_id
        #     params['parentType'] = 'dataset'
        #     params['name'] = 'process'
        #
        #     # creating folder
        #
        #     url = host+'api/datasets/'+dataset_id+'/newFolder'
        #
        #     response = requests.post(url, data=json.dumps(params), headers=headers, params=params,
        #                              auth=None, timeout=1000, verify=False)
        #     response.raise_for_status()
        #
        #
        #
        #     as_json = response.json()
        #     folder_id = as_json["id"]
        #     for each_id in upload_ids:
        #         try:
        #             logger.info("trying to move " + str(each_id))
        #             current_id = each_id
        #             url = host+'api/datasets/'+dataset_id+'/moveFile/'+folder_id+'/'+current_id
        #             requests.post(url, data=json.dumps(params), headers=headers, params=params,
        #                           auth=None, timeout=1000, verify=False)
        #         except Exception as e:
        #             logger.info("Could not move file to folder : " + str(current_id))
        #     logger.info("Finished moving files to folder")
        #
        #     result = {"ran landsattrend extractor": "true", "tiles": str(tiles_in_folder)}
        #     metadata = self.get_metadata(result, 'dataset', dataset_id, host)
        #     pyclowder.datasets.upload_metadata(connector, host, secret_key, dataset_id, metadata)
        #
        # else:
        #     logger.info("Not enough tiles to run extractor")
        #
        # try:
        #     path_to_dataset_download = os.path.join('home', dataset_name)
        #     path_to_data = os.path.join('home', 'data')
        #     logger.info('contents of home')
        #     logger.info(str(os.listdir('/home')))
        #     if os.path.isdir(path_to_dataset_download):
        #         logger.info('removing dataset download : ' + path_to_dataset_download)
        #         shutil.rmtree(path_to_dataset_download)
        #         shutil.rmtree(path_to_data)
        # except Exception as e:
        #     logger.info("Could not delete dataset download")




if __name__ == "__main__":


    path_to_dataset = os.path.join(os.getcwd(), 'home', 'landsat test','data')
    print(os.path.isdir(path_to_dataset))
    move_files_in_dataset(path_to_dataset)
    print('before extractor starts, contents of home')
    print(os.listdir(HOME_DIR))
    extractor = LandsattrendExtractor()
    extractor.start()


    print('nothing yet')