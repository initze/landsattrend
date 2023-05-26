import os
import shutil

path_to_landsattrend = '/scratch/bbou/toddn/landsat-delta/landsattrend'
print(path_to_landsattrend)
path_to_projects = '/projects/bbou/toddn'

dir_contents = os.listdir(path_to_landsattrend)
for content in dir_contents:
    content_path = os.path.join(path_to_landsattrend, content)
    if os.path.isfile(content_path):
        print('file', content_path)
        new_path = os.path.join(path_to_projects, 'landsattrend', content)
        if content is 'setup.py':
            shutil.copy(content_path, new_path)
    elif os.path.isdir(content_path):
        print('dir', content_path)
        if content is not 'data' and content is not 'process':
            new_path = os.path.join(path_to_projects, 'landsattrend', content)
            if content is 'aux_data':
                shutil.copytree(content_path, new_path)