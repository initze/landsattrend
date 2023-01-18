import glob
import os

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal_array as ga, gdal
from skimage import morphology, segmentation, feature, measure, filters
from sklearn import cluster

from landsattrend.classify import Classify, ClassifyDEM
from landsattrend.config_study_sites import study_sites
from landsattrend.utils import array_to_file



# remove from landsattrend !!!


SCRATCH_DATA_DIR = '/scratch/bbou/toddn/landsat'
PROCESS_ROOT = SCRATCH_DATA_DIR
process_dir = os.path.join(PROCESS_ROOT, 'process')

def get_stats(labelid, L_1, L_all, pr_array, factor=1., selem=np.ones((3,3))):
    """calculate statistics of objects
    :param selem:
    :param factor:
    :param pr_array:
    :param L_all:
    :param L_1:
    :param labelid:
    """
    # erode lake polygon by one
    L_1_erode = morphology.erosion(L_1==labelid, footprint=selem)
    # dilate full polygon by one
    L_all_dilate = morphology.dilation(L_all==labelid, footprint=selem)
    # get transition area between outer boundary of dilated full polygon and eroded lake polygon
    L_margin = np.logical_and(L_all_dilate, ~L_1_erode)

    # get basic statistics
    obj = L_margin
    obj1 = L_1_erode
    stats = [pr_array[0, obj ].sum(), pr_array[1, obj ].sum(), pr_array[2, obj ].sum(), pr_array[3, obj ].sum()]
    lake_area = np.sum(obj1) + pr_array[0, obj ].sum() * factor

    # get change statistics
    grow_area = stats[2] * factor
    shrink_area = stats[3] * factor
    grow_percentage = grow_area / (lake_area + shrink_area) * 100
    shrink_percentage = shrink_area / (lake_area + shrink_area) * 100
    return lake_area, grow_area, shrink_area, grow_percentage, shrink_percentage

def remove_edge_polygons(label_array):
    A = np.ones_like(label_array, np.bool)
    A[1:-1, 1:-1] = False
    edgelist = np.unique(label_array[A])
    label_array_reduced = label_array * ~np.in1d(label_array, edgelist).reshape(label_array.shape)
    return label_array_reduced

def remove_edge_polygons2(label_array, class_array):

    # set mask for edge polygons
    A1 = np.ones_like(label_array, np.bool)
    A1[1:-1, 1:-1] = False
    # set mask for
    M = np.ma.masked_equal(class_array, 0).mask
    if M.sum() > 0:
        A2 = morphology.dilation(M)
        # merge masks to one
        A = np.logical_or(A1, A2)
    else:
        A = A1
    # get intersecting labels
    edgelist = np.unique(label_array[A])
    label_array_reduced = label_array * ~np.in1d(label_array, edgelist).reshape(label_array.shape)
    return label_array_reduced

def df_to_raster(df, coordinates, feature, dtype=np.float, shp=(1000,1000), nodata=-9999):
    outraster= np.ones(shp, dtype=dtype)*nodata
    for index, row in df.iterrows():
        c = tuple(row[coordinates].T)
        outraster[c] = row[feature]
    return outraster

def get_total_perc(df, axis=0):
    if axis == 0:
        return (df.area_water + df.area_wgain).sum(axis=axis) / (df.area_water + df.area_wloss).sum(axis=axis) - 1
    else:
        return (df[['area_water', 'area_wgain']].sum(axis=axis) / (df[['area_water', 'area_wloss']]).sum(axis=axis)) - 1

def load_data(zone, query='proba <= 0.5', model_path = r'F:\18_Paper02_LakeAnalysis\01_Classification\03_Remove_Rivers\classmodel_RF_river_removal5.z', dem_fire=False, subdir=True):
    working_dir = r'F:\18_Paper02_LakeAnalysis\01_Classification\02_Results\01_Algorithmtest'

    if subdir:
        outdir = r'F:\18_Paper02_LakeAnalysis\01_Classification\02_Results\01_Algorithmtest\01_Results\{z}'.format(z=zone)
    else:
        r'F:\18_Paper02_LakeAnalysis\01_Classification\02_Results\01_Algorithmtest\01_Results'
    results_csv = os.path.join(outdir, '{z}_lake_dataset.csv'.format(z=zone))
    label_raster = os.path.join(outdir, '{z}_label_raster.tif'.format(z=zone))
    class_vector = os.path.join(outdir, '{z}_label_vector.shp'.format(z=zone))
    # Automatic path definition
    classfile = os.path.join(working_dir, zone, r'class.vrt')
    probafile = os.path.join(working_dir, zone, r'proba.vrt')

    cl = rasterio.open(classfile)
    cl_array = cl.read(1)
    cl.close()

    array_shp = cl_array.shape

    ds = rasterio.open(probafile)
    pr_array = ds.read()
    ds.close()

    ds = rasterio.open(label_raster)
    L_all = ds.read()
    ds.close()
    L_all = np.array(L_all, dtype=np.int)

    # get basic region properties + with intensity
    # load raster
    props = measure.regionprops(L_all)
    xx = []
    for p in props:
        xx.append([p.label, p.coords])
    df_coords = pd.DataFrame(columns=['id', 'coords'], data=xx)
    df_coords.index=df_coords.id

    df = pd.DataFrame.from_csv(results_csv)
    df = pd.merge(df_coords, df.drop(['coords'], axis=1), on='id')
    df.index = df.id

    # Load fire, DEM props
    if dem_fire:
        fire_array = ga.LoadFile(r'F:\18_Paper02_LakeAnalysis\02_AuxData\01_ForestLoss\forestfire_{zone}.tif'.format(zone=zone))
        f2 = morphology.remove_small_objects(fire_array, min_size=49)
        df_fire = improps_to_df(L_all, f2, properties=['label', 'min', 'mean', 'max'], prefix='fire_')
        dem_array = ga.LoadFile(r'F:\18_Paper02_LakeAnalysis\02_AuxData\04_DEM\DEM_{zone}.tif'.format(zone=zone))
        df_el = improps_to_df(L_all, dem_array, properties=['label', 'min', 'mean', 'max'], prefix='el_')
        slp_array = ga.LoadFile(r'F:\18_Paper02_LakeAnalysis\02_AuxData\04_DEM\DEM_{zone}_slope.tif'.format(zone=zone))
        df_slp = improps_to_df(L_all, slp_array,properties=['label', 'min', 'mean', 'max'], prefix='slp_')
        df_aux = pd.concat([df_fire, df_el, df_slp], axis=1)
        df = pd.concat([df, df_aux], axis=1)
    #####################################


    if model_path:
        model = joblib.load(model_path)

        # check data for specific columns and proceed accordingly
        if 'class' in df.columns:
            X = df.drop(['id', 'coords', 'class', 'proba'], axis=1).dropna()
        else:
            X = df.drop(['id', 'coords'], axis=1).dropna()

        # apply classifcation model
        pr = model.predict(X)
        proba = model.predict_proba(X)

        # insert columns
        df['class'] = pr
        df['proba'] = proba[:, 1]

        df_query = df.query(query)
    else:
        df_query = df

    return df_query, cl_array, outdir, classfile

# TODO: integrate to LakeMaker object?
def improps_to_df(label_image, intensity_image,
                  properties=None, prefix=None):
    """
    Function to read imagestatistics from labelled image raster objects and intensity image.
    Function based on skimage.measure.regionprops
    :param label_image:
    :param intensity_image:
    :param properties:
    :param prefix:
    :return:
    """
    if properties is None:
        properties = ['label', 'min', 'mean', 'max']
    zz = []
    props = measure.regionprops(label_image, intensity_image=intensity_image)
    for p in props:
        zz.append([p.label, p.min_intensity, p.mean_intensity, p.max_intensity])
    zz = np.array(zz)
    df = pd.DataFrame(data=zz, columns=properties, index=np.array(zz[:,0], np.int))
    if 'label' in properties:
        df = df.drop(['label'], axis=1)
    if prefix:
        df.columns = [prefix + c for c in df.columns]
    return df


class LakeMaker(object):
    """This is a Class to run the lake extraction and characterization workflow
    """
    def __init__(self, zone, directory, tiles_directory, suffix='', classperiod='1999-2014'):
        self.zone = zone
        self.directory = directory
        self.tiles_directory = tiles_directory
        self.classperiod = classperiod
        #self._setup_image_paths()
        self._setup_class_vrt_paths()
        self._setup_aux_paths()
        self._setup_mask_paths()
        self._setup_df_path()
        self._setup_filtered_export_paths()
        self._setup_final_dataset_path()
        self._startup_check()
        self._check_dir_structure()
        pass

    def _startup_check(self):
        """
        Function to check for the existence of processed intermediate steps, like masks or lake datasets
        :return:
        """
        # check for existance of inputfolder
        print("Site definition:", os.path.exists(self.directory))
        print("Auxilliary Data calculated:", self._check_auxdata_exist())
        print("Masks calculated:", self._check_masks_exist())
        print("Dataset CSV calculated:", os.path.exists(self.lake_dataset_path_))
        print("Filtered Mask and Vectors calculated:", os.path.exists(self.label_CfilterVector_path_))
        print("Final Output Dataset calculated:", os.path.exists(self.final_dataset_path_json_))

    @staticmethod
    def _make_classification_vrt(directory, ctype, nodata=0):
        #TODO: Docstring
        txtfile = os.path.join(directory, '{ctype}.txt'.format(ctype=ctype))
        vrtfile = os.path.join(directory, '{ctype}.vrt'.format(ctype=ctype))
        files = glob.glob(os.path.join(directory, '{ctype}*.tif'.format(ctype=ctype)))
        f = open(txtfile, 'w')
        for fi in files:
            f.write(fi + '\n')
        f.close()
        # TODO the text file is empty!!!
        command = r'gdalbuildvrt -input_file_list {txtfile} {vrtfile}'.format(gdal_path=os.environ['GDAL_BIN'],
                                                                              txtfile=txtfile, vrtfile=vrtfile)
        os.system(r'gdalbuildvrt -input_file_list {txtfile} {vrtfile}'.format(gdal_path=os.environ['GDAL_BIN'],
                                                                              txtfile=txtfile, vrtfile=vrtfile))

    def _setup_class_vrt_paths(self):
        self.class_vrt_path_ = os.path.join(self.directory, r'01_Classification_Raster', 'class.vrt')
        self.proba_vrt_path_ = os.path.join(self.directory, r'01_Classification_Raster', 'proba.vrt')
        self.confidence_vrt_path_ = os.path.join(self.directory, r'01_Classification_Raster', 'confidence.vrt')

    def _check_dir_structure(self):
        """
        Function that checks if the assigned data directory exists. Creates defined subdirectory structure if it
        does not exist yet.
        :return:
        """
        if not os.path.exists(self.directory):
            subdirs = ['01_Classification_Raster',
                       '02_Aux_Data',
                       '03_Lake_Masks',
                       '04_Lake_Dataset_Table',
                       '05_Lake_Dataset_Raster_01_raw',
                       '05_Lake_Dataset_Raster_02_final']
            [os.makedirs(os.path.join(self.directory, s)) for s in subdirs]

    def classify(self, class_model):
        """
        Function to classify the data with the defined scikit-learn classification model. tile structure needs to be
        indicated as a list
        :param tiles:
        :param class_model:
        :return:
        """
        print('in classify method')
        model = joblib.load(class_model)
        model.n_jobs=-1
        outdir = os.path.join(process_dir, '01_Classification_Raster')
        # TODO quick fix - make more sophisticated solution
        #imagefolder = self.tiles_directory
        #imagefolder = os.path.join(study_sites[0]['result_dir'], self.classperiod, 'tiles')

        image_list = glob.glob(os.path.join(self.tiles_directory, '*.tif'))
        print('the image list in classify', image_list)
        # run Classification
        for image in image_list:
            print(image)
            cl = Classify(model, image=image,
                          outputfolder=outdir)

            # Skip if there are no data available
            try:
                cl.load_raster_for_classify()
            except ValueError:
                continue
            cl.classify()
            cl.write_output()
        # create vrt-file (virtual raster tile) that merges all tiles to one mosaic for each of
        # class, proba and confidence
        for ctype in ['class', 'proba', 'confidence']:
            nodata=0
            if ctype == 'proba':
                nodata=None
            self._make_classification_vrt(outdir, ctype, nodata=nodata)

    def _setup_aux_paths(self):
        """
        Function to define paths to fire, dem and slope datasets
        :return:
        """
        self.firemask_path_ = os.path.join(self.directory, r'02_Aux_Data','forestfire.tif')
        self.dem_path_ = os.path.join(self.directory, r'02_Aux_Data','dem.tif')
        self.slope_path_ = os.path.join(self.directory, r'02_Aux_Data','slope.tif')

    def _check_auxdata_exist(self):
        """
        Function to check if Auxilliary data are already preprocessed
        :return: Boolean
        """
        paths = [self.firemask_path_, self.dem_path_, self.slope_path_]
        return np.all([os.path.exists(p) for p in paths])

    def prepare_aux_data(self, demfile, firefile):
        """
        Function to create auxiliary data - DEM data and firemask
        :param demfile: path to master DEM image (Panarctic DEM)
        :param firefile: path to master Firemask image (Hansen et al., 2013)
        :return:
        """
        # read classification raster/vrt to get data extent
        class_vrt = os.path.join(process_dir, '01_Classification_Raster', 'class.vrt')
        with rasterio.open(class_vrt) as ds:
            bnd = ds.bounds
            epsg = (list(ds.crs.values())[0]).split(':')[-1]

        # create firemap/mask
        s = r'gdalwarp -t_srs EPSG:{epsg} -tr 30 30 -te {xmin} {ymin} {xmax} {ymax} {infile} {outfile}'.format(gdal_path=os.environ['GDAL_BIN'],
                                                                                                                      epsg=epsg,
                                                                                                               xmin=bnd.left,
                                                                                                               ymin=bnd.bottom,
                                                                                                               xmax=bnd.right,
                                                                                                               ymax=bnd.top,
                                                                                                               infile=firefile,
                                                                                                               outfile=self.firemask_path_)
        os.system(s)
        # create elevation model
        s = r'gdalwarp -t_srs EPSG:{epsg} -tr 30 30 -r cubic -te {xmin} {ymin} {xmax} {ymax} {infile} {outfile}'.format(gdal_path=os.environ['GDAL_BIN'],
                                                                                                                               epsg=epsg,
                                                                                                                        xmin=bnd.left,
                                                                                                                        ymin=bnd.bottom,
                                                                                                                        xmax=bnd.right,
                                                                                                                        ymax=bnd.top,
                                                                                                                        infile=demfile,
                                                                                                                        outfile=self.dem_path_)
        os.system(s)
        # create slope from created DEM
        s_slope = r'gdaldem slope -alg ZevenbergenThorne {infile} {slopefile}'.format(gdal_path=os.environ['GDAL_BIN'],
                                                                                             infile=self.dem_path_, slopefile=self.slope_path_)
        os.system(s_slope)

    def _load_classdata(self):
        """
        private Function to load classified data to numpy array
        :return:
        """
        # define path to mosaicked VRT of classified data
        self.classfile = os.path.join(process_dir, '01_Classification_Raster', 'class.vrt')
        self.probafile = os.path.join(process_dir, '01_Classification_Raster', 'proba.vrt')

        # read data content
        cl = rasterio.open(self.classfile)
        cl_array = cl.read(1)
        cl.close()
        ds = rasterio.open(self.probafile)
        pr_array = ds.read()
        ds.close()

        return cl_array, pr_array

    def make_masks(self, selem=np.ones((3,3))):
        """
        Function to create masks and labelled masks of Zone A (stable water area)
        and Zone B (dynamic margin and change areas)
        :param selem:
        :return:
        """
        cl_array, pr_array = self._load_classdata()

        # get shape of array/image
        array_shp = cl_array.shape

        # Lakes only
        M_1 = cl_array == 1
        # combined objects: Lake + drain + erode
        M_all = np.in1d(cl_array, [1,12,21]).reshape(array_shp)
        # remove small objects
        M_all = morphology.remove_small_objects(M_all, min_size=11, connectivity=1)

        # make label for all full objects
        L_all = measure.label(M_all, background=0, connectivity=1)
        # remove objects that touch the edge of the image
        #TODO: can be improved (incorporate to object/class?)
        L_all = remove_edge_polygons2(L_all, cl_array)

        M_1_erode = morphology.erosion(M_1, footprint=selem)

        # dilate full polygon by one
        M_all_dilate = morphology.dilation(M_all, footprint=selem)
        L_all_dilate = morphology.dilation(L_all, footprint=selem)
        # get transition area between outer boundary of dilated full polygon and eroded lake polygon
        M_margin = np.logical_and(M_all_dilate, ~M_1_erode)
        L_margin = M_margin * L_all_dilate
        L_1_erode = M_1_erode * L_all

        self.mask_A = M_1_erode
        self.mask_B = M_margin

        self.label_A = L_1_erode
        self.label_B = L_margin
        self.label_C = L_all

        #save masks to raster file
        self._save_masks()

    def _setup_mask_paths(self):
        """
        Function to define paths to masks
        :return:
        """
        self.mask_A_path_ = os.path.join(self.directory, '03_Lake_Masks', 'mask_A.tif')
        self.mask_B_path_ = os.path.join(self.directory, '03_Lake_Masks', 'mask_B.tif')
        self.label_A_path_ = os.path.join(self.directory, '03_Lake_Masks', 'label_A.tif')
        self.label_B_path_ = os.path.join(self.directory, '03_Lake_Masks', 'label_B.tif')
        self.label_C_path_ = os.path.join(self.directory, '03_Lake_Masks', 'label_C.tif')

    def _check_masks_exist(self):
        """
        Function to check if masks are already preprocessed
        :return: Boolean
        """
        mlist = [self.mask_A_path_, self.mask_B_path_, self.label_A_path_, self.label_B_path_, self.label_C_path_]
        return np.all([os.path.exists(m) for m in mlist])

    def _save_masks(self):
        """
        Function to save masks
        :return:
        """
        # save mask to files
        array_to_file(self.mask_A, self.mask_A_path_, self.classfile, dtype=gdal.GDT_Byte)
        array_to_file(self.mask_B, self.mask_B_path_, self.classfile, dtype=gdal.GDT_Byte)
        array_to_file(self.label_A, self.label_A_path_, self.classfile, dtype=gdal.GDT_UInt32)
        array_to_file(self.label_B, self.label_B_path_, self.classfile, dtype=gdal.GDT_UInt32)
        array_to_file(self.label_C, self.label_C_path_, self.classfile, dtype=gdal.GDT_UInt32)

    def load_masks(self):
        """
        Function to load preprocessed and saved masks
        :return:
        """
        self.mask_A = ga.LoadFile(self.mask_A_path_)
        self.mask_B = ga.LoadFile(self.mask_B_path_)
        self.label_A = ga.LoadFile(self.label_A_path_)
        self.label_B = ga.LoadFile(self.label_B_path_)
        self.label_C = ga.LoadFile(self.label_C_path_)

    def _get_unique_labels(self):
        """
        Function to retrieve individual labels from labelled mask
        :return:
        """
        un_labels = pd.unique(self.label_C.ravel())
        self.un_labels = un_labels[un_labels!=0]

    def _make_stats_shape(self, probability):
        """
        calculate shape parameters of detected lake objects
        :param probability:
        :return: pandas.DataFrame
        """
        # get basic region properties + with intensity
        props = measure.regionprops(self.label_C, probability[[0,2,3]].sum(axis=0))
        xx = []
        for p in props:
            xx.append([p.label, p.area, p.convex_area, p.coords, p.equivalent_diameter, p.eccentricity,
                       p.major_axis_length, p.minor_axis_length, p.orientation, p.perimeter, p.solidity,
                       p.max_intensity, p.mean_intensity, p.min_intensity])
        df = pd.DataFrame(index=self.un_labels, data=xx,
                          columns=['id', 'area', 'convex_area', 'coords',
                                                                      'equivalent_diameter', 'eccentricity',
                                                                      'majax_length', 'minax_length',
                                                                      'orientation', 'perimeter', 'solidity',
                                                                      'max', 'mean', 'min'])
        return df

    def _make_stats_ZoneA(self):
        """
        calculate stats of Zone A (stable water area)
        :return: pandas.DataFrame
        """
        props = measure.regionprops(self.label_A)
        zz = []
        for p in props:
            zz.append([p.label,p.area])
        df = pd.DataFrame(data=zz, columns=['id', 'area_water_inside'])
        df.index = df.id
        return df

    def _make_stats_ZoneB(self, probability):
        """
        calculate stats of Zone B (dynamic and margin area)
        :param probability:
        :return: pandas.DataFrame
        """
        props = measure.regionprops(self.label_B)
        yy = []
        for p in props:
            yy.append([p.label,
                       probability[0][p.coords[:,0], p.coords[:,1]].sum(),
                       probability[2][p.coords[:,0], p.coords[:,1]].sum(),
                       probability[3][p.coords[:,0], p.coords[:,1]].sum()])
        df = pd.DataFrame(index=self.un_labels, data=yy, columns=['id', 'area_water_steady', 'area_wgain', 'area_wloss'])
        return df

    def _make_stats_auxData(self):
        fire_array = ga.LoadFile(self.firemask_path_)
        f2 = morphology.remove_small_objects(fire_array, min_size=49)
        df_fire = improps_to_df(self.label_C, f2,
                                properties=['label', 'min', 'mean', 'max'],
                                prefix='fire_')
        dem_array = ga.LoadFile(self.dem_path_)
        df_el = improps_to_df(self.label_C, dem_array,
                              properties=['label', 'min', 'mean', 'max'],
                              prefix='el_')
        slp_array = ga.LoadFile(self.slope_path_)
        df_slp = improps_to_df(self.label_C, slp_array,
                               properties=['label', 'min', 'mean', 'max'],
                               prefix='slp_')
        df_aux = pd.concat([df_fire, df_el, df_slp], axis=1)
        return df_aux

    @staticmethod
    def _calc_stats(df):
        """
        fill dataframe with calculated statistics
        :param df:
        :return: pandas.DataFrame
        """
        df['area_water'] = df.area_water_inside + df.area_water_steady
        df['area_total'] = df[['area_water', 'area_wloss', 'area_wgain']].sum(axis=1)
        df['area_budget'] = df.area_wgain - df.area_wloss
        df['perc_water'] = df.area_water / df.area_total
        df['perc_wloss'] = df.area_wloss / df.area_total
        df['perc_wgain'] = df.area_wgain / df.area_total
        # calculate rate_change

        return df

    def make_stats(self):
        """
        create dataframe with lake specific change and shape statistics
        :return:
        """
        # load probability layer
        _, pr_array = self._load_classdata()
        # get unique labels
        self._get_unique_labels()
        # get specific statistics for each Zone
        df_shape = self._make_stats_shape(pr_array)
        df_A = self._make_stats_ZoneA()
        df_B = self._make_stats_ZoneB(pr_array)
        # Join three calculated zones
        df_j1 = df_shape.join(df_B.drop(['id'], axis=1))
        df = df_j1.join(df_A.drop(['id'], axis=1))
        df.area_water_inside = df.area_water_inside.fillna(value=0.)
        # merge all data to one DataFrame
        self.df_start = pd.concat([self._calc_stats(df), self._make_stats_auxData()], axis=1)

    def save_df(self):
        """
        function to save dataframe to csv
        :return:
        """
        if not os.path.exists(self.lake_dataset_path_):
            self.df_start.to_csv(self.lake_dataset_path_)
        else:
            print("Dataset csv-file already exists")

    def _setup_df_path(self):
        """
        Create path to data output path (csv-file)
        :return:
        """
        self.lake_dataset_path_ = os.path.join(process_dir, r'04_Lake_Dataset_Table', 'lake_dataset.csv')

    def load_df(self):
        """
        function to load dataframe from csv
        :return:
        """
        self.df_start =  pd.DataFrame.from_csv(self.lake_dataset_path_)

    #TODO: tmp
    def filter_data(self, model_path, query='proba <= 0.5 and max > 0.95'):
        """
        Function to filter rivers and fire data
        :param model_path: path to saved scikit-learn model for filtering river or other low-quality info
        :param query: SQL/pandas query for filtering
        :return:
        """
        #"""
        # load model
        model = joblib.load(model_path)
        # setup data
        X = self.df_start.drop(['id', 'coords'], axis=1).dropna()
        # apply classifcation model
        pr = model.predict(X)
        proba = model.predict_proba(X)
        # insert result to DataFrame
        df = self.df_start.copy()
        df['class'] = pr
        df['proba'] = proba[:, 1]
        self.df_filter = df[np.all([df['proba'] <= 0.5, df['max'] > 0.95], axis=0)]
        #"""
        #self.df_filter = self.df_start


    def save_filtered_data(self):
        """
        Function to save filtered dataset to raster (labelled ids) and polygon Shapefile
        :return:
        """
        self.load_masks()
        self.label_Cfilter = self._filter_labelled_mask(self.label_C, self.df_filter.index)
        array_to_file(self.label_Cfilter, self.label_Cfilter_path_, self.class_vrt_path_, dtype=gdal.GDT_UInt32)
        gdal_call = os.path.join(os.environ['GDAL_PATH'],'gdal_polygonize.py')
        s = r'python {gdal_call} -q -8 -f "ESRI Shapefile" -mask {raster} {raster} {vector} label_id label_id'.format(
                gdal_call=gdal_call,
                raster=self.label_Cfilter_path_,
                vector=self.label_CfilterVector_path_)
        os.system(s)

    # TODO:Implementation to read filtered data
    def load_filtered_data(self):
        pass

    @staticmethod
    def _filter_labelled_mask(mask, labels):
        """
        private Function to filter labelled raster mask to predefined labels - remove all values that are not in labels
        :param mask: labelled 2-D numpy array
        :param labels: list/array of allowed values, all other values will be discarded from the mask
        :return: filtered labelled 2-D numpy array
        """
        m = np.zeros_like(mask)
        idx = np.in1d(mask.ravel(), labels).reshape(mask.shape)
        m[idx] = mask[idx]
        return m

    def _setup_filtered_export_paths(self):
        """
        private function to setup file names for saving the output data
        :return:
        """
        self.label_Cfilter_path_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_01_raw', 'label_C.tif')
        self.label_CfilterVector_path_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_01_raw', 'label_C.shp')


    # TODO: make more efficient - use temporary dataframes
    def finalize_calculations(self):
        """
        Function that include the transformation from pixel values to metric values (ha for area and m for length)
        :return:
        """
        df_geom = gpd.read_file(self.label_CfilterVector_path_).set_index("label_id").join(self.df_filter)
        df_geom['id'] = df_geom.index
        self.df_metric = df_geom.copy()
        self._transform_area_to_ha(df_geom)
        self._transform_len_to_meter(df_geom)
        self._transform_radians_to_degree(df_geom)
        # TODO calculate change rates
        self._calculate_statistics(self.df_metric)

    def _transform_area_to_ha(self, px_df, factor_px_to_ha=0.09):
        """
        private function to transform area from pixels to ha
        :param px_df:
        :param factor_px_to_ha:
        :return:
        """
        col_area = ['area', 'convex_area', 'area_water_steady', 'area_wgain',
                    'area_wloss', 'area_water_inside', 'area_water', 'area_total', 'area_budget']
        self.df_metric.loc[:, col_area] = px_df.loc[:, col_area] * factor_px_to_ha

    def _transform_len_to_meter(self, px_df, factor_px_to_m=30.):
        """
        private function to transform length from pixels to meters
        :param px_df:
        :param factor_px_to_m:
        :return:
        """
        col_len = ['equivalent_diameter', 'perimeter']
        self.df_metric.loc[:, col_len] = px_df.loc[:, col_len] * factor_px_to_m

    #TODO: needs to get fixed
    def _transform_radians_to_degree(self, px_df):
        """
        private function to transform angles from radians to degrees
        :param px_df:
        :return:
        """
        col_rad = ['orientation']
        self.df_metric.loc[:, col_rad] = -np.rad2deg(px_df.loc[:, col_rad])+90

    def _calculate_statistics(self, df):
        """
        calculate major lake specific statistics into human readable and understandable format and save into new
        DataFrame for output
        :param df:
        :return:
        """
        self.df_final = df[['id', 'geometry']].copy()
        area_start = (df.area_water + df.area_wloss)
        area_end = (df.area_water + df.area_wgain)
        self.df_final['Area_start_ha'] = area_start
        self.df_final['Area_end_ha'] = area_end
        self.df_final['NetChange_ha'] = area_end - area_start
        self.df_final['NetChange_perc'] = (area_end / area_start - 1) * 100
        self.df_final['GrossIncrease_ha'] = df.area_wgain
        self.df_final['GrossIncrease_perc'] = df.area_wgain / area_start * 100
        self.df_final['GrossDecrease_ha'] = df.area_wloss
        self.df_final['GrossDecrease_perc'] = df.area_wloss / area_end * 100
        self.df_final['StableWater_ha'] = df.area_water
        # shape parameters
        self.df_final['Perimeter_meter'] = df['perimeter']
        self.df_final['Orientation_degree'] = df['orientation']
        self.df_final['Solidity_ratio'] = df['solidity']
        self.df_final['Eccentricity_ratio'] = df['eccentricity']

        # Calculate change rates replace with years
        start = int(self.classperiod.split('-')[0])
        end = int(self.classperiod.split('-')[1])
        n_years = end-start

        self.df_final['ChangeRateNet_myr-1'] = (self.df_final['NetChange_ha'] * 1e4) / self.df_final['Perimeter_meter'] / n_years
        self.df_final['ChangeRateGrowth_myr-1'] = (self.df_final['GrossIncrease_ha'] * 1e4) / self.df_final['Perimeter_meter'] / n_years

    def save_results(self):
        """
        Function to save final dataset to GeoJSON
        :return:
        """
        # save polygons
        if not os.path.exists(self.final_dataset_path_json_):
            self.df_final.to_file(self.final_dataset_path_json_, driver='GeoJSON')
        if not os.path.exists(self.final_dataset_path_gpkg_):
            self.df_final.to_file(self.final_dataset_path_gpkg_, driver='GPKG')
        # save centroids
        if not os.path.exists(self.final_dataset_ctr_path_json_):
            df_centroid = self.df_final.copy()
            df_centroid['geometry'] = self.df_final.convex_hull.centroid
            df_centroid.to_file(self.final_dataset_ctr_path_json_, driver='GeoJSON')
            df_centroid.to_file(self.final_dataset_ctr_path_gpkg_, driver='GPKG')

    def load_results(self):
        """
        Function to load final dataset
        :return:
        """
        self.df_final = gpd.read_file(self.final_dataset_path_json_)

    def _setup_final_dataset_path(self):
        """
        Function to setup path for dataset output
        :return:
        """
        self.final_dataset_path_json_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_02_final',
                                                'lake_change.geojson')
        self.final_dataset_path_gpkg_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_02_final',
                                                'lake_change.gpkg')
        self.final_dataset_ctr_path_json_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_02_final',
                                                    'lake_change_centroid.geojson')
        self.final_dataset_ctr_path_gpkg_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_02_final',
                                                    'lake_change_centroid.gpkg')

    def print_regional_statistics(self):
        pass

    def plot_regional_statistics(self):
        pass

    # TODO: Add option for feature and different types
    def export_gridded_results(self, blocksize_list=None, type_list=None):
        """
        function to export chosen result/feature to gridded information of defined size
        :param blocksize: int - number of pixels for aggregating info
        :return:
        """

        if type_list is None:
            type_list = ['netchange', 'grossgain', 'grossloss',
                         'waterstable', 'area_start_ha',
                         'area_end_ha', 'limnicity_start',
                         'limnicity_end']
        if blocksize_list is None:
            blocksize_list = [100, 250]
        _, pr_array = self._load_classdata()

        # check if df_filter is loaded already, otherwise load explicitly
        try:
            self.df_final
        except AttributeError:
            self.load_results()
        # check if mask is loaded already, otherwise load explicitly
        try:
            self.label_B
            self.label_A
        except AttributeError:
            print("Loading Masks")
            self.load_masks()

        # create mask of filtered edge zones or stable water zone
        label_Bfilter = self._filter_labelled_mask(self.label_B, self.df_final.id)
        label_Afilter = self._filter_labelled_mask(self.label_A, self.df_final.id)

        for blocksize in blocksize_list:
            # create row and column vectors
            rows, cols = pr_array[0].shape # maybe improve
            rr = np.arange(0, rows, blocksize, np.int)
            cc = np.arange(0, cols, blocksize, np.int)
            outresolution = blocksize * 30

            grid = self._calculate_grid(label_Afilter, label_Bfilter, pr_array, blocksize)
            for type in type_list:
                A = np.zeros((len(rr), len(cc)))
                for r in rr:
                    for c in cc:
                        A[int(r/blocksize):int(r/blocksize+1), int(c/blocksize):int(c/blocksize+1)] = grid[type][r:int(r+blocksize), c:int(c+blocksize)].sum()

                self._setup_gridded_result_paths(blocksize, type=type)
                array_to_file(A*0.09, self.gridded_result_netchange_path_,
                              self.class_vrt_path_, outresolution=(outresolution, outresolution))

    def _setup_gridded_result_paths(self, blocksize, type=type):
        """
        :param blocksize:
        :param type:
        :return:
        """
        size = blocksize * 30
        self.gridded_result_netchange_path_ = os.path.join(process_dir, r'05_Lake_Dataset_Raster_02_final',
                                                           'lake_change_grid_{size}_{type}.tif'.format(size=size,
                                                                                                       type=type))
    @staticmethod
    def _calculate_grid(label_Afilter, label_Bfilter, pr_array, blocksize):
        """
        :param label_Afilter: filtered mask of stable water area
        :param label_Bfilter: filtered mask of changing lake margins
        :param pr_array: array of classification probabilities
        :param blocksize:
        :return: dictionary with area statistics on original pixel-level
        """
        grid = {}
        # calculate
        grossgain = ((label_Bfilter != 0) * pr_array[2])
        grossloss = ((label_Bfilter != 0) * pr_array[3])
        area_transform_formula = ((blocksize*0.3)**2) / 100
        waterstable = ((label_Bfilter != 0) * pr_array[0] + (label_Afilter != 0))

        grid['netchange'] = grossgain - grossloss
        grid['grossgain'] = grossgain
        grid['grossloss'] = grossloss
        grid['waterstable'] = waterstable
        grid['area_start_ha'] = waterstable + grossloss
        grid['area_end_ha'] = waterstable + grossgain
        grid['limnicity_start'] = (waterstable + grossloss) / area_transform_formula
        grid['limnicity_end'] = (waterstable + grossgain) / area_transform_formula

        return grid

class SlumpMaker(LakeMaker):

    def _startup_check(self):
        """
        Function to check for the existence of processed intermediate steps, like masks or lake datasets
        :return:
        """
        # check for existance of inputfolder
        print("Site definition:", os.path.exists(self.directory))
        print("Data Classfied:", os.path.exists(self.probafile_path_))
        #print "Masks calculated:", self._check_masks_exist()
        print("Dataset CSV calculated:", os.path.exists(self.lake_dataset_path_))
        print("Segments calculated:", os.path.exists(self.segment_raster_path_))
        print("Segment Statistics calculated:", os.path.exists(self.segment_vectorstats_path_))

    def _setup_mask_paths(self):
        """
        Function to define paths to masks
        :return:
        """
        self.segment_raster_path_ = os.path.join(self.directory, '03_Lake_Masks', 'segment_label.tif')
        self.segment_vector_path_ = os.path.join(self.directory, '03_Lake_Masks', 'segment_label.geojson')
        self.segment_vectorstats_path_ = os.path.join(self.directory, '05_Lake_Dataset_Raster_01_raw', 'segment_stats.geojson')
        self.classfile_path_ = os.path.join(self.directory, '01_Classification_Raster', 'class.vrt')
        self.probafile_path_ = os.path.join(self.directory, '01_Classification_Raster', 'proba.vrt')
        self.demfile_path_ = os.path.join(self.directory, '02_Aux_Data', 'dem.vrt')
        self.slopefile_path_ = os.path.join(self.directory, '02_Aux_Data', 'slope.vrt')
        self.final_dataset_path_ = os.path.join(self.directory, r'05_Lake_Dataset_Raster_02_final',
                                                'segment_stats_classified.geojson')

    def classify(self, class_model, tiles):
        """
        Function to classify the data with the defined scikit-learn classification model. tile structure needs to be
        indicated as a list
        :param tiles:
        :param class_model:
        :return:
        """
        #TODO: make quiet option/verbosity
        # Loop classification
        model = joblib.load(class_model)
        outdir = os.path.join(self.directory, '01_Classification_Raster')
        imagefolder = os.path.join(study_sites[self.zone]['result_dir'], self.classperiod, 'tiles')

        # run Classification
        for t in tiles:
            print(t)
            cl = ClassifyDEM(model, zone=self.zone, tile=t,
                          imagefolder=imagefolder,
                          outputfolder=outdir)

            # Skip if there are no data available
            try:
                cl.load_raster_for_classify()
            except ValueError:
                print("skip")
                continue
            cl.classify()
            cl.write_output()
        # create vrt-file (virtual raster tile) that merges all tiles to one mosaic for each of
        # class, proba and confidence
        for ctype in ['class', 'proba', 'confidence']:
            nodata=0
            if ctype == 'proba':
                nodata=None
            self._make_classification_vrt(outdir, ctype, nodata=nodata)
        self._make_dem_vrt(tiles)


    def segmentation(self, mode='simple', thresh_peaks=0.5, thresh_mask = 0.3, sigma=2,
                     sigma_edge=0.5, min_distance=1, pr_index=4, edge_filter='sobel', rescale=True):
        """
        Function to create and export segmentation layer
        :param edge_filter:
        :param sigma_edge:
        :param thresh_mask:
        :param thresh_peaks:
        :param mode:
        :param sigma: float - size of gaussian kernel for smoothing
        :param min_distance: int - minimum distance of local maxima in pixel size
        :param pr_index: int - index of probability layer (e.g. 4 for slumps, 3 for fire)
        :return:
        """
        try:
            self._load_segment_vector()
            self._load_segment_raster()
        except:
            print("Starting Segmentation!")
            pr_array, dem_array, slope_array = self._load_classdata(pr_index=pr_index)

            if mode == 'simple':
                print("Segmentation in simple mode")
                ds = self._load_classdata2()
                mask = (ds == 14)
                self.segment_raster = measure.label(mask, connectivity=1)
                self.segment_raster[~mask] = -1

            elif mode == 'local_kmeans':
                print("Segmentation in local kmeans mode")
                pr_array = filters.gaussian(pr_array, sigma=sigma, preserve_range=True)
                mask = self._local_cluster_seg(pr_array, thresh=thresh_mask)
                self.segment_raster = measure.label(mask, connectivity=1)
            elif mode == 'local_threshold':
                mask = (filters.threshold_local(pr_array, 3, offset=0.3) > 0)
                self.segment_raster = measure.label(mask, connectivity=1)
            elif mode == 'local_thresh_edge':
                edge = filters.sobel(pr_array)
                ts = filters.threshold_local(edge, 3, offset=0)
                local_thresh = filters.threshold_yen(ts.ravel())
                edge_bin = ts> local_thresh
                edge_bin_closed = morphology.binary_closing(edge_bin)
                edge_bin_open = morphology.binary_opening(edge_bin_closed)
                edge_bin_open = morphology.remove_small_holes(edge_bin_open)
                self.segment_raster = measure.label(edge_bin_open, connectivity=1)

            elif mode == 'watershed2':
                # image filter
                filtered = filters.gaussian(pr_array, sigma=sigma, preserve_range=True)
                # find threshold of foreground vs. background
                thresh = filters.threshold_otsu(filtered.ravel())
                # find peaks
                peaks = feature.peak_local_max(filtered, min_distance=3, threshold_abs=thresh_peaks, indices=False)
                markers = morphology.label(peaks)
                ds_bin = pr_array > thresh_mask
                markers[~ds_bin] = -1
                edge = filters.sobel(filtered)
                self.segment_raster = segmentation.watershed(edge, markers, mask=ds_bin)
            elif mode == 'watershed3':
                #
                filtered = filters.gaussian(pr_array, sigma=sigma, preserve_range=True)
                if rescale:
                    filtered = filtered * (pr_array.max() / filtered.max())
                ds_bin = pr_array >= thresh_mask
                markers = np.array(feature.peak_local_max(filtered, min_distance=min_distance,
                                                          threshold_abs=thresh_peaks, indices=False), dtype=np.int)
                markers = morphology.label(markers)
                markers[~ds_bin] = -1

                filter_ds = filters.gaussian(pr_array, sigma=sigma_edge, preserve_range=True)
                edge = filters.sobel(filter_ds)

                self.segment_raster = segmentation.watershed(edge, markers, mask=ds_bin, compactness=1)

            else:
                print("Segmentation in default mode")
                filtered = filters.gaussian(pr_array, sigma=sigma, preserve_range=True)
                ds_bin = pr_array >= thresh_mask
                markers = np.array(feature.peak_local_max(filtered, min_distance=min_distance,
                                                          threshold_abs=thresh_peaks, indices=False), dtype=np.int)
                markers = morphology.label(markers)
                markers[~ds_bin] = -1

                filter_ds = filters.gaussian(pr_array, sigma=sigma_edge, preserve_range=True)
                edge = filters.sobel(filter_ds)

                self.segment_raster = segmentation.watershed(-filtered, markers, mask=ds_bin, compactness=0.5)

            self._export_segment_files()
            self._load_segment_vector()

    def delete_segmented_files(self):
        os.remove(self.segment_raster_path_)
        os.remove(self.segment_vector_path_)

    def _export_segment_files(self):
        array_to_file(self.segment_raster, self.segment_raster_path_, self.probafile_path_, noData=True, noDataVal=-1)
        gdal_call = os.path.join(os.environ['GDAL_PATH'],'gdal_polygonize.py')
        #s = r'python {gdal_call}
        s = r'python {gdal_call} -q -8 -f "GeoJSON" -mask {raster} {raster} {vector} label label'.format(
                        gdal_call=gdal_call,
                        raster=self.segment_raster_path_,
                        vector=self.segment_vector_path_)
        os.system(s)

    @staticmethod
    def _local_cluster_seg(ds, thresh=0.3):
        shp = ds.shape
        ds_test = ds
        ds_mask = ds_test > thresh
        label = measure.label(ds_mask)
        cl_mask = np.zeros_like(ds_test)
        rp = measure.regionprops(label, ds_test)
        for rr in rp:
            if rr.area > 3:
                # get bbox coords
                rmin, cmin, rmax, cmax = rr.bbox
                ds_loc = ds_test[rmin:rmax, cmin:cmax]
                # cluster
                mn, cl, inertia = cluster.k_means(np.expand_dims(ds_loc.ravel(), 1), n_clusters=2)
                slump = cl==np.argmax(np.squeeze(mn))
                rr, cc = np.where(slump.reshape(ds_loc.shape))
                cl_mask[rmin+rr, cmin+cc] = True
                #cl_mask[rmin:rmax, cmin:cmax] = slump.reshape(ds_loc.shape)
        return cl_mask

    def _load_segment_raster(self):
        self.segment_raster = np.array(ga.LoadFile(self.segment_raster_path_), dtype=np.int)

    def _load_segment_vector(self):
        self.segment_vector = gpd.read_file(self.segment_vector_path_).set_index('label')

    def _make_dem_vrt(self, tiles):
        for ctype in ['dem', 'slope']:

            #TODO: Docstring
            directory = os.path.join(self.directory, '02_Aux_Data')
            txtfile = os.path.join(directory, '{ctype}.txt'.format(ctype=ctype))
            vrtfile = os.path.join(directory, '{ctype}.vrt'.format(ctype=ctype))
            with open(txtfile, 'w') as f:
                for pr in tiles:
                    fi = (os.path.join(study_sites[self.zone]['dem_dir'], '{zone}_{pr}_{idx}.tif'.format(zone=self.zone, pr=pr, idx=ctype)))
                    f.write(fi + '\n')
            gdal_call = os.path.join(os.environ['GDAL_PATH'], 'gdal_polygonize.py')
            os.system(r'{gdal_call} -input_file_list {txtfile} {vrtfile}'.format(gdal_call=gdal_call,
                                                                                                    txtfile=txtfile,
                                                                                                    vrtfile=vrtfile))

    def _load_classdata(self, pr_index=4):
        """
        private Function to load classified data to numpy array
        :return:
        """
        # define path to mosaicked VRT of classified data
        self.probafile_path_ = os.path.join(self.directory, '01_Classification_Raster', 'proba.vrt')
        self.demfile_path_ = os.path.join(self.directory, '02_Aux_Data', 'dem.vrt')
        self.slopefile_path_ = os.path.join(self.directory, '02_Aux_Data', 'slope.vrt')

        # read data content
        pr_array = ga.LoadFile(self.probafile_path_)[pr_index]
        dem_array = ga.LoadFile(self.demfile_path_)
        slope_array = ga.LoadFile(self.slopefile_path_)

        return pr_array, dem_array, slope_array

    def _load_classdata2(self):
        """
        private Function to load classified data to numpy array
        :return:
        """
        # read data content
        array = ga.LoadFile(self.classfile_path_)
        return array

    def make_stats(self, pr_index=4):
        """
        Function to trigger calculation of segment statistics
        :param pr_index: int - index of probability layer (e.g. 4 for slumps, 3 for fire)
        :return:
        """
        pr_array, dem_array, slope_array = self._load_classdata(pr_index=pr_index)
        df_shape = self._make_stats_shape()
        df_proba = self._make_stats_layer(pr_array, prefix='p_')
        df_dem = self._make_stats_layer(dem_array, prefix='dem_')
        df_slope = self._make_stats_layer(slope_array, prefix='slope_')
        df_dem_offset2 = self._make_stats_layer(dem_array - filters.gaussian(dem_array, sigma=2, preserve_range=True),
                                               prefix='demoffset2_')
        df_dem_offset3 = self._make_stats_layer(dem_array - filters.gaussian(dem_array, sigma=3, preserve_range=True),
                                               prefix='demoffset3_')
        df_stats = pd.concat([df_shape, df_proba, df_dem, df_slope, df_dem_offset2, df_dem_offset3], axis=1)

        self.df_stats = self.segment_vector.join(df_stats)

    def delete_stats_files(self):
        os.remove(self.segment_vectorstats_path_)

    def _make_stats_shape(self):
        p_shape = measure.regionprops(self.segment_raster)
        xx = []
        for p in p_shape:
            xx.append([p.label, p.area, p.convex_area, p.equivalent_diameter, p.eccentricity,
                       p.major_axis_length, p.minor_axis_length, p.orientation, p.perimeter, p.solidity])
        # transfer data to pd.DataFrame
        df = pd.DataFrame(data=xx,
                      columns=['label', 'area', 'convex_area', 'equivalent_diameter', 'eccentricity',
                               'majax_length', 'minax_length', 'orientation', 'perimeter', 'solidity'])
        df = df.set_index('label')
        return df

    def _make_stats_layer(self, array, prefix=''):
        p_shape = measure.regionprops(self.segment_raster, array)
        xx = []
        for p in p_shape:
            xx.append([p.label, p.min_intensity, p.mean_intensity, p.max_intensity])

        df = pd.DataFrame(data=xx,
                          columns=['label', prefix+'min', prefix+'mean', prefix+'max'])
        df = df.set_index('label')
        return df


    def export_stats(self):
        """
        Function to trigger export of vector/segment statistics to geojson file
        :return:
        """
        if not os.path.exists(self.segment_vectorstats_path_):
            self.df_stats.to_file(self.segment_vectorstats_path_, driver='GeoJSON')
        else:
            "File already exists"
