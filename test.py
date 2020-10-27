import time
import os
import numpy as np
from pathlib import Path


path_to_them = '/Users/helium/ncsa/pdg/landsattrend/home/data/Z056-Kolyma/1999-2019/tiles'

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_tiles_from_files(path_to_files):
    tile_values = []

    all_files = os.listdir(path_to_files)
    for each in all_files:
        base_filename = Path(each).stem
        index_of_underscores = find(base_filename, '_')
        tile_value = base_filename[index_of_underscores[1]+1: index_of_underscores[3]]
        if tile_value not in tile_values:
            tile_values.append(tile_value)
    return tile_values

def main():
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    tiles = get_tiles_from_files(path_to_them)
    time.sleep(60*60)


if __name__ == '__main__':
    main()
