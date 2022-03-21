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
import lake_analysis
import distutils

PROCESS_ROOT = os.getcwd()

HOME_DIR = os.path.join(os.getcwd(), 'home')

DATA_DIR = os.path.join(HOME_DIR, 'data')

def clean_out_data_dir(path_to_data):
    contents = os.listdir(path_to_data)
    for entry in contents:
        path_to_entry = os.path.join(path_to_data, entry)
        if os.path.isfile(path_to_entry):
            os.remove(path_to_entry)
        elif os.path.isdir(path_to_entry):
            shutil.rmtree(path_to_entry)
    print('data dir cleaned out')
    print(os.listdir(path_to_data))

# 500PM

def add_file_to_tiles(path_to_file):
    print('adding file to tiles')
    print(path_to_file)
    filename = path_to_file.split('/')[-1]
    print("file name is", filename)
    filename_parts = filename.split('_')
    print('filename_parts')
    print(filename_parts)
    print('how many filename parts?', len(filename_parts))
    current_site_name = filename_parts[2]
    print('site name')
    print(current_site_name)
    current_class_period = filename_parts[1]
    print('class period')
    print(current_class_period)
    current_lat = filename_parts[3].lstrip('-')
    print('lat lon')
    current_lon = filename_parts[4].rstrip('.tif')
    print(current_lat, current_lon)

    current_tiles_directory = os.path.join(PROCESS_ROOT, 'home','data', current_site_name, current_class_period, 'tiles')
    print('current tiles directory', current_tiles_directory)
    if not os.path.isdir(current_tiles_directory):
        try:
            tiles_path = Path(current_tiles_directory)
            tiles_path.mkdir(parents=True)
        except Exception as e:
            print(e)
    if os.path.isdir(current_tiles_directory):
        new_filename = 'trendimage_'+current_site_name+'_'+current_lat+'_'+current_lon+'.tif'
        print('new filename is', new_filename)
        destination = os.path.join(current_tiles_directory, new_filename)
        print('destination is', destination)
        shutil.move(path_to_file, destination)
    return current_tiles_directory

def get_files_to_move(path_to_unzipped_dataset):
    files_to_move = []
    contents = os.listdir(path_to_unzipped_dataset)
    for item in contents:
        path_to_item = os.path.join(path_to_unzipped_dataset, item)
        if item.endswith('.tif'):
            files_to_move.append(path_to_item)
    return files_to_move

def move_file_to_tiles(path_to_file):
    try:
        path_parts = path_to_file.split('/')
        filename = path_parts[-1]
        print('filename is', filename)
        file_parts = filename.split('_')
        print('file parts are', file_parts)
        for i in range(0, len(file_parts)):
            print(i, file_parts[i])
        file_class_period = file_parts[1]
        print('class period', file_class_period)
        file_site_name = file_parts[2]
        file_lat = file_parts[3].lstrip('-')
        file_lon = file_parts[4].rstrip('.tif')
        tile_dir = os.path.join(DATA_DIR, file_site_name, file_class_period, 'tiles' )
        print('tile dir is', tile_dir)
        print('trying to make the tile dir')
        try:
            pathlib.Path(tile_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print('failed to make dir')
            print(e)
        print('CHECK SOMETHING HERE')
        print(os.path.exists(tile_dir))
        print('does the tile dir exist? see above')
        new_filename = 'trendimage_' + file_site_name + '_' + file_lat + '_' + file_lon + '.tif'
        destination = os.path.join(tile_dir, new_filename)
        print('original', path_to_file)
        print(os.path.isfile(path_to_file))
        print('destination', destination)
        shutil.move(path_to_file, destination)
        print('moved file')
        return tile_dir
    except Exception as e:
        print('error moving file', path_to_file)
        print(e)
    return None




def move_files_in_dataset(path_to_unzipped_dataset):

    contents = os.listdir(path_to_unzipped_dataset)
    path_to_tiles = ""
    for item in contents:
        path_to_item = os.path.join(path_to_unzipped_dataset, item)
        if item.endswith('.tif'):
            try:
                path_to_tiles = add_file_to_tiles(os.path.join(path_to_unzipped_dataset, item))
            except Exception as e:
                print(e)
        else:
            try:
                os.remove(os.path.join(path_to_unzipped_dataset, item))
            except Exception as e:
                print(e)
    return path_to_tiles


def delete_unnecessary_files(path_to_unzipped_dataset):
    contents = os.listdir(path_to_unzipped_dataset)
    for item in contents:
        if os.path.isdir(os.path.join(path_to_unzipped_dataset, item)):
            if item == 'metadata':
                try:
                    shutil.rmtree(os.path.join(path_to_unzipped_dataset, item))
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

        DATA_DIR = os.path.join(HOME_DIR, 'data')
        dataset_download_location = os.path.join(DATA_DIR, dataset_name)
        print('dataset download location', dataset_download_location)
        #
        download = pyclowder.datasets.download(connector, host, secret_key, dataset_id)
        with zipfile.ZipFile(download, 'r') as zip:
            zip.extractall(dataset_download_location)

        files_to_move = get_files_to_move(os.path.join(dataset_download_location, 'data'))
        print('got files to move')
        tile_dirs = []
        for f in files_to_move:
            tile_dir = move_file_to_tiles(f)
            if tile_dir:
                if tile_dir not in tile_dirs:
                    tile_dirs.append(tile_dir)


        # delete everything except the files
        # delete_unnecessary_files(dataset_download_location)
        # time.sleep(60*2)

        # path_to_tiles = move_files_in_dataset(os.path.join(dataset_download_location, 'data'))
        #
        # path_to_tiles_components = path_to_tiles.split('/')
        # data_dir_index = path_to_tiles_components.index('data')
        # current_site_name = path_to_tiles_components[data_dir_index + 1]
        # current_class_period = path_to_tiles_components[data_dir_index + 2]





        # move files to somewhere else
        print('stopping to check that things copies')

        print('right before running late analysis')
        path_to_tiles = tile_dirs[0]
        print(path_to_tiles)
        path_to_tiles_components = path_to_tiles.split('/')
        print('we got path to tiles')
        time.sleep(60*3)
        current_site_name = path_to_tiles_components[2]
        print('site name', current_site_name)
        current_class_period = path_to_tiles_components[3]
        print('class period', current_class_period)
        print('taking a rest')
        time.sleep(60*3)
        lake_analysis.run_lake_analysis(path_to_tiles=path_to_tiles, current_class_period=current_class_period, current_site_name=current_site_name)

        RESULT_DIR = os.path.join(os.getcwd(),'process',current_site_name,'05_Lake_Dataset_Raster_02_final')
        results = os.listdir(RESULT_DIR)

        client = pyclowder.datasets.ClowderClient(host=host, key=secret_key)


        # create folder in dataset
        data = dict()
        data["name"] = 'process'
        data["parentId"] = dataset_id
        data["parentType"] = "dataset"
        new_folder = client.post('/datasets/' + dataset_id + '/newFolder', content=data, params=data)

        # upload files
        for result in results:
            path_to_result = os.path.join(RESULT_DIR, result)
            file_result = client.post_file("/uploadToDataset/%s" % dataset_id, path_to_result)
            file_id = file_result['id']
            data = dict()
            move_result = client.post('/datasets/' + dataset_id + '/moveFile/' + new_folder['id'] + '/' + file_id, content=data)
        path_to_home = os.path.join(os.getcwd(),'home')
        home_contents = os.listdir(path_to_home)
        for each in home_contents:
            current_path = os.path.join(path_to_home, each)
            if os.path.isdir(current_path):
                try:
                    shutil.rmtree(current_path)
                except Exception as e:
                    logger.info('could not delete,' + str(current_path))
                    logger.info(e)
            else:
                try:
                    os.remove(current_path)
                except Exception as e:
                    logger.info('could not delete,' + str(current_path))
                    logger.info(e)




if __name__ == "__main__":
    DATA_DIR = os.path.join(HOME_DIR, 'data')
    # delete all data
    clean_out_data_dir(path_to_data=DATA_DIR)
    extractor = LandsattrendExtractor()
    extractor.start()


    print('nothing yet')