__author__ = 'initze'
"""

"""
import fiona
import glob
import os

def write_depth(infile, outfile):
    with fiona.drivers():
        with fiona.open(infile) as source:
            meta = source.meta
            meta['schema']['properties']['depth'] = 'float'
            with fiona.open(outfile, 'w', **meta) as sink:
                # Process only the records intersecting a box.
                for f in source:
                    prp = f['properties']['Descriptio']
                    depth = float(prp.split()[1][:-1])
                    f['properties']['depth'] = depth
                    sink.write(f)
    return 0


def main():
    flist = glob.glob('*\\\TrackPoints.shp')
    for f in flist:
        p, file = os.path.split(f)
        infile = os.path.join(p, file)
        outfile = os.path.join(p, 'depth.shp')
        write_depth(infile, outfile)


if __name__ == "__main__":
    main()

