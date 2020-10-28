#!/usr/bin/env python

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
import run_lake_analysis
import distutils

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_tiles_from_files(path_to_files):
    tile_values = []

    all_files = os.listdir(path_to_files)
    for each in all_files:
        base_filename = Path(each).stem
        logging.info("base_filename")
        logging.info(str(base_filename))
        index_of_underscores = find(base_filename, '_')
        tile_value = base_filename[index_of_underscores[1]+1: index_of_underscores[3]]
        if tile_value not in tile_values:
            tile_values.append(tile_value)
    return tile_values

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

        dataset_download_location = os.path.join('/home', dataset_name)

        download = pyclowder.datasets.download(connector, host, secret_key, dataset_id)
        with zipfile.ZipFile(download, 'r') as zip:
            zip.extractall(dataset_download_location)

        path_to_landsattrend_data = os.path.join(dataset_download_location, 'data', 'data', 'Z056-Kolyma')
        path_to_data_folder_in_dataset = os.path.join(dataset_download_location, 'data', 'data')
        logger.info("path to landsattrend data")
        logger.info(str(path_to_landsattrend_data))
        if os.path.isdir(path_to_landsattrend_data):
            logger.info("We have landsattrend data")
            if os.path.isdir('/home/data'):
                logger.info("we have old data, removing")
                shutil.rmtree('/home/data')
            else:
                shutil.move(path_to_data_folder_in_dataset, '/home/data')
        logger.info("run_lake_analysis now")
        contents_of_tiles = os.listdir('/home/data/Z056-Kolyma/1999-2019/tiles')
        logger.info("contents of tiles")
        logger.info(str(contents_of_tiles))
        tiles_in_folder = get_tiles_from_files('/home/data/Z056-Kolyma/1999-2019/tiles')
        logger.info('tiles_in_folder')
        logger.info(str(tiles_in_folder))
        if len(tiles_in_folder) > 1:
            location_of_final_dataset = run_lake_analysis.process_tiles(tiles_in_folder)
            logger.info("final dataset location")
            logger.info(str(location_of_final_dataset))

            upload_ids = []

            contents_of_final_dataset = os.listdir(str(location_of_final_dataset))
            for each_file in contents_of_final_dataset:
                try:
                    current_file_path = os.path.join(location_of_final_dataset, each_file)
                    current_id = pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, filepath=current_file_path)
                    upload_ids.append(current_id)
                except Exception as e:
                    logger.info("Could not upload : " + str(current_file_path))

            logger.info("We have uploaded : " + str(len(upload_ids)))
            logger.info("These files have ids : " + str(upload_ids))
             # TODO upload to dataset as files, then move to new folder

            params = dict()
            headers = {'content-type': 'application/json'}
            params['key'] = secret_key
            params['parentId'] = dataset_id
            params['parentType'] = 'dataset'
            params['name'] = 'process'

            # creating folder

            url = host+'api/datasets/'+dataset_id+'/newFolder'

            response = requests.post(url, data=json.dumps(params), headers=headers, params=params,
                                     auth=None, timeout=1000, verify=False)
            response.raise_for_status()



            as_json = response.json()
            folder_id = as_json["id"]
            for each_id in upload_ids:
                try:
                    logger.info("trying to move " + str(each_id))
                    current_id = each_id
                    url = host+'api/datasets/'+dataset_id+'/moveFile/'+folder_id+'/'+current_id
                    requests.post(url, data=json.dumps(params), headers=headers, params=params,
                                  auth=None, timeout=1000, verify=False)
                except Exception as e:
                    logger.info("Could not move file to folder : " + str(current_id))
            logger.info("Finished moving files to folder")

            result = {"ran landsattrend extractor": "true", "tiles": str(tiles_in_folder)}
            metadata = self.get_metadata(result, 'dataset', dataset_id, host)
            pyclowder.datasets.upload_metadata(connector, host, secret_key, dataset_id, metadata)

        else:
            logger.info("Not enough tiles to run extractor")

        try:
            if os.path.isdir('/home/data'):
                shutil.rmtree('/home/data')
        except Exception as e:
            logger.info("Could not delete /home/data")




if __name__ == "__main__":


    extractor = LandsattrendExtractor()
    extractor.start()
