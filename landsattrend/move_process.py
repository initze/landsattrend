import os
import sys
import shutil
class_period = sys.argv[1]
path_to_process = sys.argv[2]
dir_in_process = os.listdir(path_to_process)

new_dir = os.path.join(path_to_process, class_period)

for dir in dir_in_process:
    if dir != class_period:
        path_to_dir = os.path.join(path_to_process, dir)
        if os.path.isdir(path_to_dir):
            print('moving', path_to_dir, new_dir)
            shutil.move(path_to_dir, new_dir)
