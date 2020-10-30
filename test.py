import time
import os
import numpy as np
from pathlib import Path


path_to_them = '/Users/helium/ncsa/pdg/landsattrend/home/data/Z056-Kolyma/1999-2019/tiles'


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_tiles_from_files(path_to_files):
    tile_values = []

    tile_file_map = {}

    all_files = os.listdir(path_to_files)
    for each in all_files:
        base_filename = Path(each).stem
        print("base_filename")
        print(str(base_filename))
        index_of_underscores = find(base_filename, '_')
        tile_value = base_filename[index_of_underscores[1] + 1: index_of_underscores[3]]
        file_ending = each[index_of_underscores[1] + 1:]

        if tile_value not in tile_values:
            tile_values.append(tile_value)

        if tile_value in tile_file_map:
            current_map_entry = tile_file_map[tile_value]
            current_map_entry.append(file_ending)
            tile_file_map[tile_value] = current_map_entry
        else:
            tile_file_map[tile_value] = [file_ending]

    # TODO check for each index list
    indexlist = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']
    tiles = list(tile_file_map.keys())

    for tile in tiles:
        for each in indexlist:
            entry = tile+'_'+each+'.tif'
            if entry not in tile_file_map[tile]:
                print("not in")
                tile_values.remove(tile)

    return tile_values


def check_for_files(tile_dict):
    indexlist = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']

def main():
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    tiles = get_tiles_from_files(path_to_them)
    time.sleep(60*60)


if __name__ == '__main__':
    main()
