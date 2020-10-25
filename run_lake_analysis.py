from landsattrend.lake_analysis import LakeMaker
import os
import logging

os.environ['GDAL_PATH'] = r'C:\Users\initze\AppData\Local\Continuum\anaconda3\envs\landsattrend2\Scripts'
os.environ['GDAL_BIN'] = r'C:\Users\initze\AppData\Local\Continuum\anaconda3\envs\landsattrend2\Library\bin'
PROCESSING_DIR = os.getcwd()
DEM_LOCATION = os.path.join(PROCESSING_DIR, r'aux_data', 'dem', 'DEM.vrt')
FOREST_LOCATION = os.path.join(PROCESSING_DIR, r'aux_data', 'forestfire', 'forestfire.vrt')
tiles = ['32_8', '32_9', '32_10', '32_11']
zone = 'Z056-Kolyma'
directory_location = os.path.join(PROCESSING_DIR, 'process')
site_name = 'Z056-Kolyma'

CLASS_MODEL = os.path.join(PROCESSING_DIR, 'models', '20180831_RF_4class_noDEM_973samples_py3_v0.z')
LAKE_FILTER_MODEL = os.path.join(PROCESSING_DIR, 'models', '20180820_lakefilter_12039samples_py3.z')
CURRENT_WORKING_DIR = '/Users/helium/ncsa/pdg/landsattrend/'

def process_tiles():
    logging.getLogger('pyclowder').setLevel(logging.DEBUG)
    logging.getLogger('__main__').setLevel(logging.DEBUG)

    logging.debug("in main method")
    print('in main method')

    l = LakeMaker(zone, os.path.join(directory_location, site_name), classperiod='1999-2019')
    logging.info("\nStart Classification")
    print("\nStart Classification")
    l.classify(CLASS_MODEL, tiles)
    # TODO: continue here
    logging.info("\nPreparing additional Data")
    print("\nPreparing additional Data")
    l.prepare_aux_data(DEM_LOCATION, FOREST_LOCATION)

    logging.info("\nCreating Masks")
    print("\nCreating Masks")
    l.make_masks()

    logging.info("\nCalculating Stats")
    print("\nCalculating Stats")
    l.make_stats()

    logging.info("\nSaving DataFrame to Disk")
    print("\nSaving DataFrame to Disk")
    l.save_df()

    logging.info("\nFiltering non-lake objects")
    print("\nFiltering non-lake objects")
    l.filter_data(LAKE_FILTER_MODEL)

    logging.info("\nSaving DataFrame to Disk")
    print("\nSaving DataFrame to Disk")
    l.save_filtered_data()

    logging.info("\nTransforming data to metric values")
    print("\nTransforming data to metric values")
    l.finalize_calculations()

    logging.info("\nSaving DataFrame to Disk")
    print("\nSaving DataFrame to Disk")
    l.save_results()

    logging.info("\nSaving ResultGrid at 3km resolution")
    print("\nSaving ResultGrid at 3km resolution")
    l.export_gridded_results([100, 250, 500])


if __name__ == "__main__":
    process_tiles()
