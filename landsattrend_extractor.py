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

             # TODO upload to dataset as files, then move to new folder
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
