import os
import shutil

data_path = os.path.join(os.getcwd(), 'data')

data_contents = os.listdir(data_path)

for content in data_contents:
    content_path = os.path.join(data_path, content)
    if os.path.isdir(content_path):
        shutil.rmtree(content_path)