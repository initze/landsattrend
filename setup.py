from distutils.core import setup
exec(open('landsattrend/version.py').read())
from landsattrend import __version__

setup(
    name = 'landsattrend',
    version=__version__,
    description='Package to preprocess and create trends on Landsat time-series',
    author='Ingmar Nitze',
    author_email='ingmar.nitze@awi.de',
    package_dir={'': '.',
                 'landsattrend.utils': 'landsattrend/utils'},
    packages=['landsattrend', 'landsattrend.utils'],
