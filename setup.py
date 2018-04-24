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
    install_requires=['numpy>=1.13',
    'pandas>=0.20', 'geopandas>=0.3',
    'netCDF4>=1.3', 'gdal>=2.0',
    'rasterio>=0.36',
    'fiona>=1.7',
    'joblib>=0.11',
    'bottleneck>=1.2',
    'scipy>=0.19',
    'pyproj',
    'shapely',
    'scikit-learn',
    'scikit-image'])