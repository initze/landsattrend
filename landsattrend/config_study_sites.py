__author__ = 'initze'
import os
import pandas as pd


DATA_DIR = r'N:\petacarbnob1\petacarb\initze\01_Data_Archive\02_Processed_Masked'
DEM_DIR = r'D:\01_RasterData\01_Landsat\04_DEM'
PROCESSING_DIR_01 = r'D:\01_RasterData\01_Landsat\02_Processed'
PROCESSING_DIR_02 = r'K:\01_RasterData\01_Landsat\02_Processed'
VECTOR_DIR = r'D:\05_Vector'
RESULT_DIR = r'E:\06_Trendimages'

# load study sites from configuration file
csvdir = os.path.join(os.environ['HOMEDRIVE'], os.environ['HOMEPATH'], 'landsattrend')
csvpath = os.path.join(csvdir, 'config_study_sites.csv')
study_sites = pd.DataFrame.from_csv(csvpath).T.to_dict()
"""
study_sites = {

    'GFZ-Brasil-1':{
        'name' : 'GFZ-Brasil-1',
        'fishnet_file' : os.path.join(VECTOR_DIR,'07_GFZ-Brazil1.shp'),
        'data_dir' : os.path.join(DATA_DIR, '99_Brasil'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '09_GFZ-Brasil-1_tiles'),
        'dem_dir' : DEM_DIR,
        'result_dir' : r'',
        'epsg' : 32724,
        'bbox' : [355000, 565000, 9270000, 9600000]},

    'Kolyma':{
        'name' : 'Kolyma',
        'fishnet_file' : os.path.join(VECTOR_DIR,'08_Kolyma.shp'),
        'data_dir' : r'E:\01_RasterData\01_Landsat\01_Data\08_Kolyma',
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '09_Kolyma_tiles'),
        'dem_dir' : DEM_DIR,
        'result_dir' : r'',
        'epsg' : 32657,
        'bbox' : [300000, 570000, 7600000, 7900000]},

    'KolymaWest':{
        'name' : 'KolymaWest',
        'fishnet_file' : os.path.join(VECTOR_DIR,'09_KolymaWest.shp'),
        'data_dir' : r'E:\01_RasterData\01_Landsat\01_Data\08_Kolyma',
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '09_KolymaWest_tiles'),
        'dem_dir' : DEM_DIR,
        'result_dir' : r'',
        'epsg' : 32657,
        'bbox' : [210000, 300000, 7600000, 7900000]},

    '11-PokhKeper':{
        'name' : '11-PokhKeper',
        'fishnet_file' : os.path.join(VECTOR_DIR,'11_PokhKeper.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '11_PokhKeper_tiles'),
        'dem_dir' : DEM_DIR,
        'result_dir' : r'',
        'epsg' : 32658,
        'bbox' : [300000, 740000, 7400000, 7800000]},

    'Tibet':{
        'name' : 'Tibet',
        'fishnet_file' : os.path.join(VECTOR_DIR,'Tibet.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, 'Tibet-Z047_tiles'),
        'dem_dir' : DEM_DIR,
        'result_dir' : os.path.join(RESULT_DIR,'Tibet'),
        'epsg' : 32646,
        'bbox' : [300000, 740000, 7400000, 7800000]},

    'Tibet-Z047':{
        'name' : 'Tibet-Z047',
        'fishnet_file' : os.path.join(VECTOR_DIR,'Tibet-Z047.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, 'Tibet-Z047_tiles'),
        'dem_dir' : DEM_DIR,
        'result_dir' : os.path.join(RESULT_DIR,'Tibet-Z047'),
        'epsg' : 32647,
        'bbox' : [300000, 740000, 7400000, 7800000]},

    'Z003':{
        'name' : 'Z003',
        'fishnet_file' : os.path.join(VECTOR_DIR,'003_Z003.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '003_Z003_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z003'),
        'result_dir' : os.path.join(RESULT_DIR,'Z003_2016_2mths'),
        'epsg' : 32603,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z004':{
        'name' : 'Z004',
        'fishnet_file' : os.path.join(VECTOR_DIR,'004_Z004.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '004_Z004_tiles'),
        'dem_dir' : os.path.join(DEM_DIR,'Z004'),
        'result_dir' : os.path.join(RESULT_DIR,'Z004_2016_2mths'),
        'epsg' : 32604,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z005':{
        'name' : 'Z005',
        'fishnet_file' : os.path.join(VECTOR_DIR,'005_Z005.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '005_Z005_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z005'),
        'result_dir' : os.path.join(RESULT_DIR,'Z005_2016_2mths'),
        'epsg' : 32605,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z005_TOA':{
        'name' : 'Z005_TOA',
        'fishnet_file' : os.path.join(VECTOR_DIR,'005_Z005.shp'),
        'data_dir' : os.path.join(DATA_DIR, '88_TOA'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '005_Z005_TOA_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z005_TOA'),
        'result_dir' : os.path.join(RESULT_DIR,'Z005_TOA_2016_2mths_late'),
        'epsg' : 32605,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z006':{
        'name' : 'Z006',
        'fishnet_file' : os.path.join(VECTOR_DIR,'006_Z006.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '006_Z006_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z006'),
        'result_dir' : os.path.join(RESULT_DIR,'Z006_2016_2mths'),
        'epsg' : 32606,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z007':{
        'name' : 'Z007',
        'fishnet_file' : os.path.join(VECTOR_DIR,'007_Z007.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '007_Z007_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z007'),
        'result_dir' : os.path.join(RESULT_DIR,'Z007'),
        'epsg' : 32607,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z008':{
        'name' : 'Z008',
        'fishnet_file' : os.path.join(VECTOR_DIR,'008_Z008.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '008_Z008_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z008'),
        'result_dir' : os.path.join(RESULT_DIR,'Z008'),
        'epsg' : 32608,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z010':{
        'name' : 'Z010',
        'fishnet_file' : os.path.join(VECTOR_DIR,'010_Z010.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '010_Z010_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z010'),
        'result_dir' : os.path.join(RESULT_DIR,'Z010'),
        'epsg' : 32610,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z010_TOA':{
        'name' : 'Z010_TOA',
        'fishnet_file' : os.path.join(VECTOR_DIR,'010_Z010.shp'),
        'data_dir' : os.path.join(DATA_DIR, '88_TOA'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '010_Z010_TOA_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z010_TOA'),
        'result_dir' : os.path.join(RESULT_DIR,'Z010_TOA'),
        'epsg' : 32610,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z011':{
        'name' : 'Z011',
        'fishnet_file' : os.path.join(VECTOR_DIR,'011_Z011.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '011_Z011_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z011'),
        'result_dir' : os.path.join(RESULT_DIR,'Z011'),
        'epsg' : 32611,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z013':{
        'name' : 'Z013',
        'fishnet_file' : os.path.join(VECTOR_DIR,'013_Z013.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '013_Z013_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z013'),
        'result_dir' : os.path.join(RESULT_DIR,'Z013_2016_2mths'),
        'epsg' : 32613,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z014':{
        'name' : 'Z014',
        'fishnet_file' : os.path.join(VECTOR_DIR,'014_Z014.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '014_Z014_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z014'),
        'result_dir' : os.path.join(RESULT_DIR,'Z014_2016_2mths'),
        'epsg' : 32614,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z017':{
        'name' : 'Z017',
        'fishnet_file' : os.path.join(VECTOR_DIR,'017_Z017.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '017_Z017_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z017'),
        'result_dir' : os.path.join(RESULT_DIR,'Z017_2016_2mths'),
        'epsg' : 32617,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z018':{
        'name' : 'Z018',
        'fishnet_file' : os.path.join(VECTOR_DIR,'018_Z018.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '018_Z018_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z018'),
        'result_dir' : os.path.join(RESULT_DIR,'Z018_2016_2mths'),
        'epsg' : 32618,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z018_X': {
        'name': 'Z018',
        'fishnet_file': os.path.join(VECTOR_DIR, '018_Z018.shp'),
        'data_dir': os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir': os.path.join(PROCESSING_DIR_01, '018_Z018_tiles'),
        'dem_dir': os.path.join(DEM_DIR, 'Z018'),
        'result_dir': os.path.join(RESULT_DIR, 'Z018X_2016'),
        'epsg': 32618,
        'bbox': [319995, 709995, 7060000, 7900000]},

    'Z027':{
        'name' : 'Z027',
        'fishnet_file' : os.path.join(VECTOR_DIR,'027_Z027.shp'),
        'data_dir' : os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '027_Z027_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z027'),
        'result_dir' : os.path.join(RESULT_DIR,'Z027'),
        'epsg' : 32627,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z033':{
        'name' : 'Z033',
        'fishnet_file' : os.path.join(VECTOR_DIR,'033_Z033.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '033_Z033_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z033'),
        'result_dir' : os.path.join(RESULT_DIR,'Z033'),
        'epsg' : 32633,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z041':{
        'name' : 'Z041',
        'fishnet_file' : os.path.join(VECTOR_DIR,'041_Z041.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '041_Z041_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z41'),
        'result_dir' : os.path.join(RESULT_DIR,'Z041_2016_2mths'),
        'epsg' : 32641,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z042':{
        'name' : 'Z042',
        'fishnet_file' : os.path.join(VECTOR_DIR,'042_Z042.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '042_Z042_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z042'),
        'result_dir' : os.path.join(RESULT_DIR,'Z042_2016_2mths'),
        'epsg' : 32642,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z043':{
        'name' : 'Z043',
        'fishnet_file' : os.path.join(VECTOR_DIR,'043_Z043.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '043_Z043_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z043'),
        'result_dir' : os.path.join(RESULT_DIR,'Z043_2016_2mths'),
        'epsg' : 32643,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z046':{
        'name' : 'Z046',
        'fishnet_file' : os.path.join(VECTOR_DIR,'046_Z046.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '046_Z046_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z046'),
        'result_dir' : os.path.join(RESULT_DIR,'Z046_2016_2mths'),
        'epsg' : 32646,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z047':{
        'name' : 'Z047',
        'fishnet_file' : os.path.join(VECTOR_DIR,'047_Z047.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '047_Z047_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z047'),
        'result_dir' : os.path.join(RESULT_DIR,'Z047_2016_2mths'),
        'epsg' : 32647,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z048':{
        'name' : 'Z048',
        'fishnet_file' : os.path.join(VECTOR_DIR,'048_Z048.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '048_Z048_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z048'),
        'result_dir' : os.path.join(RESULT_DIR,'Z048_2016_2mths'),
        'epsg' : 32648,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z049':{
        'name' : 'Z049',
        'fishnet_file' : os.path.join(VECTOR_DIR,'049_Z049.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '049_Z049_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z049'),
        'result_dir' : os.path.join(RESULT_DIR,'Z049_2016_2mths'),
        'epsg' : 32649,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z050':{
        'name' : 'Z050',
        'fishnet_file' : os.path.join(VECTOR_DIR,'050_Z050.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '050_Z050_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z050'),
        'result_dir' : os.path.join(RESULT_DIR,'Z050_2016_2mths'),
        'epsg' : 32650,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z051':{
        'name' : 'Z051',
        'fishnet_file' : os.path.join(VECTOR_DIR,'051_Z051.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '051_Z051_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z051'),
        'result_dir' : os.path.join(RESULT_DIR,'Z051_2016_2mths'),
        'epsg' : 32651,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z052':{
        'name' : 'Z052',
        'fishnet_file' : os.path.join(VECTOR_DIR,'052_Z052.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '052_Z052_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z052'),
        'result_dir' : os.path.join(RESULT_DIR,'Z052_2016_2mths'),
        'epsg' : 32652,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z053':{
        'name' : 'Z053',
        'fishnet_file' : os.path.join(VECTOR_DIR,'053_Z053.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '053_Z053_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z053'),
        'result_dir' : os.path.join(RESULT_DIR,'Z053_2016_2mths'),
        'epsg' : 32653,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z054': {
        'name': 'Z054',
        'fishnet_file': os.path.join(VECTOR_DIR, '054_Z054.shp'),
        'data_dir': os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir': os.path.join(PROCESSING_DIR_02, '054_Z054_tiles'),
        'dem_dir': os.path.join(DEM_DIR, 'Z054'),
        'result_dir': os.path.join(RESULT_DIR, 'Z054'),
        'epsg': 32654,
        'bbox': [319995, 709995, 7060000, 7900000]},

    'Z055': {
        'name': 'Z055',
        'fishnet_file': os.path.join(VECTOR_DIR, '055_Z055.shp'),
        'data_dir': os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir': os.path.join(PROCESSING_DIR_02, '055_Z055_tiles'),
        'dem_dir': os.path.join(DEM_DIR, 'Z055'),
        'result_dir': os.path.join(RESULT_DIR, 'Z055'),
        'epsg': 32655,
        'bbox': [319995, 709995, 7060000, 7900000]},

    'Z056': {
        'name': 'Z056',
        'fishnet_file': os.path.join(VECTOR_DIR, '056_Z056.shp'),
        'data_dir': os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir': os.path.join(PROCESSING_DIR_02, '056_Z056_tiles'),
        'dem_dir': os.path.join(DEM_DIR, 'Z056'),
        'result_dir': os.path.join(RESULT_DIR, 'Z056'),
        'epsg': 32656,
        'bbox': [319995, 709995, 7060000, 7900000]},

    'Z057':{
        'name' : 'Z057',
        'fishnet_file' : os.path.join(VECTOR_DIR,'057_Z057.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '057_Z057_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z057'),
        'result_dir' : os.path.join(RESULT_DIR,'Z057_2016_2mths'),
        'epsg' : 32657,
        'bbox' : [319995, 709995, 7060000, 7900000]},


    'Z058':{
        'name' : 'Z058',
        'fishnet_file' : os.path.join(VECTOR_DIR,'058_Z058.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '058_Z058_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z058'),
        'result_dir' : os.path.join(RESULT_DIR,'Z058_2016_2mths'),
        'epsg' : 32658,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Z059':{
        'name' : 'Z059',
        'fishnet_file' : os.path.join(VECTOR_DIR,'059_Z059.shp'),
        'data_dir' : os.path.join(DATA_DIR, '11_PokhKeper'),
        'processing_dir' : os.path.join(PROCESSING_DIR_02, '059_Z059_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Z059'),
        'result_dir' : r'',
        'epsg' : 32659,
        'bbox' : [319995, 709995, 7060000, 7900000]},

    'Dummy':{
        'name' : 'Dummy',
        'fishnet_file' : os.path.join(VECTOR_DIR,',999_Dummy.shp'),
        'data_dir' : r'E:\01_RasterData\01_Landsat\01_Data\999_Dummy',
        'processing_dir' : os.path.join(PROCESSING_DIR_01, '999_Dummy_tiles'),
        'dem_dir' : os.path.join(DEM_DIR, 'Dummy'),
        'result_dir' : r'',
        'epsg' : 32633,
        'bbox' : [350000, 563000, 5600000, 6000000]},


    'Z005_test': {
        'name': 'Z005',
        'fishnet_file': os.path.join(VECTOR_DIR, '005_Z005.shp'),
        'data_dir': os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir': os.path.join(PROCESSING_DIR_01, '005_Z005_tiles'),
        'dem_dir': os.path.join(DEM_DIR, 'Z005'),
        'result_dir': os.path.join(RESULT_DIR, 'Z005_test'),
        'epsg': 32605,
        'bbox': [319995, 709995, 7060000, 7900000]},

    'Z005_local': {
        'name': 'Z005',
        'fishnet_file': os.path.join(VECTOR_DIR, '005_Z005_local.shp'),
        'data_dir': os.path.join(DATA_DIR, r'99_Alaska_general'),
        'processing_dir': os.path.join(PROCESSING_DIR_01, '005_Z005_local'),
        'dem_dir': os.path.join(DEM_DIR, 'Z005_local'),
        'result_dir': os.path.join(RESULT_DIR, 'Z005_local'),
        'epsg': 32605,
        'bbox': [319995, 709995, 7060000, 7900000]},

    'testcase': {
        'name': 'testcase',
        'fishnet_file': os.path.join(VECTOR_DIR, '053_Z053.shp'),
        'data_dir': None,
        'processing_dir': None,
        'dem_dir': None,
        'result_dir': r'P:\initze\landsattrend\landsattrend\tests\data\raster\result',
        'epsg': 32653,
        'bbox': None},
}

"""
wrs2_path = os.path.join(VECTOR_DIR, 'wrs2_descending.shp')