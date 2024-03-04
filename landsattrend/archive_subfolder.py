import argparse
import os
import shutil
import tarfile

# Argument parsing #
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--delete', action='store_true')
args = parser.parse_args()

def make_tgz(folder, delete=args.delete):
    # list all subfolders
    subfld = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    # loop over each folder
    for f in subfld:
        print(os.path.join(f, '.'))
        files = os.listdir(os.path.join(f, '.'))
        # zip
        outfile = '{0}.tar.gz'.format(f)
        print("Creating output archive: ", outfile)
        tar = tarfile.open(outfile, 'w:gz')
        for fx in files:
            tar.add(os.path.join(f, fx))
            print("adding {0}".format(os.path.join(f, fx)))
        tar.close()
        print("Archiving complete")
        # check if delete necessary
        if delete:
            print("Deleting: ", f)
            shutil.rmtree(f)

    return 0

def main():
    make_tgz('.', args.delete)

if __name__ == '__main__':
    main()
