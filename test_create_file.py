import os

print('we are going to test a file')

current_dir = os.getcwd()
print(current_dir)

test = 'this is a test file'

path_for_file = os.path.join(current_dir,'sample.txt')

with open(path_for_file, 'w') as f:
    f.write(test)

print('wrote file')