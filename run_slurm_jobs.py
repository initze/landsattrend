import os
import sys

current_files = os.listdir(os.getcwd())

commands = []

for current_file in current_files:
    if current_file.endswith('sbatch'):
        if 'test' not in current_file:
            command = 'sbatch ' + current_file
            commands.append(command)

print('doing the commands')

for command in commands:
    print(command)
    # os.system(command)