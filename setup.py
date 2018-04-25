from distutils.core import setup
exec(open('landsattrend/version.py').read())
from landsattrend import __version__

setup(
    name = 'landsattrend',
    version=__version__,
    description='Package to preprocess and create trends on Landsat time-series',
    author='Ingmar Nitze',
    author_email='ingmar.nitze@awi.de',
    package_dir={'': '.'},
    packages=['landsattrend'],
    install_requires=['numpy',
    'pandas', 'geopandas',
    'netCDF4', 'gdal',
    'rasterio',
    'fiona',
    'joblib',
    'bottleneck',
    'scipy',
    'pyproj',
    'shapely',
    'scikit-learn',
    'scikit-image'])