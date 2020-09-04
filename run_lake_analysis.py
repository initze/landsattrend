from landsattrend.lake_analysis import LakeMaker
import os


PROCESSING_DIR = os.getcwd()
DEM_LOCATION = os.path.join(PROCESSING_DIR, r'aux_data', 'dem', 'DEM.vrt')
FOREST_LOCATION = os.path.join(PROCESSING_DIR, r'aux_data', 'forestfire', 'forestfire.vrt')
tiles = ['33_9']
zone = 'Z056-Kolyma'
directory_location = os.path.join(PROCESSING_DIR, 'process')
site_name = 'Z056-Kolyma'

CLASS_MODEL = os.path.join(PROCESSING_DIR, 'models', '20180831_RF_4class_noDEM_973samples_py3_v0.z')
LAKE_FILTER_MODEL = os.path.join(PROCESSING_DIR, 'models', '20180820_lakefilter_12039samples_py3.z')


def main():
    l = LakeMaker(zone, os.path.join(directory_location, site_name), classperiod='1999-2019')
    print("\nStart Classification")
    l.classify(CLASS_MODEL, tiles)
    # TODO: continue here
    print("\nPreparing additional Data")
    l.prepare_aux_data(DEM_LOCATION, FOREST_LOCATION)

    print("\nCreating Masks")
    l.make_masks()
    print("\nCalculating Stats")
    l.make_stats()
    print("\nSaving DataFrame to Disk")
    l.save_df()
    print("\nFiltering non-lake objects")
    l.filter_data(LAKE_FILTER_MODEL)
    print("\nSaving DataFrame to Disk")
    l.save_filtered_data()
    print("\nTransforming data to metric values")
    l.finalize_calculations()
    print("\nSaving DataFrame to Disk")
    l.save_results()
    print("\nSaving ResultGrid at 3km resolution")
    l.export_gridded_results([100, 250, 500])


if __name__ == "__main__":
    main()
