import os


def read_pip_freeze_file(path_to_pip_freeze):
    package_version = dict()

    with open(path_to_pip_freeze, 'r') as f:
        lines = f.readlines()
    return lines


def read_conda_file(path_to_conda_file):
    package_version = dict()

    with open(path_to_conda_file, 'r') as f:
        lines = f.readlines()
    return lines


def main():
    piplines = read_pip_freeze_file('pipfreeze.txt')
    condalines = read_conda_file('conda.txt')
    print('done')


if __name__ == '__main__':
    main()
