import glob
import os

import geopandas
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from osgeo import gdal_array as ga

from .config_study_sites import study_sites
from landsattrend.utils import array_to_file, merge_pr, coord_raster


def combine_idxlist(idxlist, extlist):
    outlist = []
    for idx in idxlist:
        for ext in extlist:
            outlist.append(idx + ext)
    return outlist

class Classify(object):
    """
    This is a class to classify trend data and write output to raster files
    :param model: scikit-learn Classifier Object
    :param zone: project specific location identifier
    :param zone: location specific tile identifier
    :param imagefolder: path of trendimages to load for classification
    :param cirange: Boolean to set if confidence ranges should be included (upper CI minus lower CI)
    :param indexlist: list of used indices
    :param outputfolder: path to save output files (files will be saved into subfolder named after zone identifier)
    """
    def __init__(self,
                 model,
                 zone='Z004',
                 tile='27_11',
                 imagefolder=r'F:\06_Trendimages\Z004_2016_2mths\1999-2014\tiles',
                 cirange=True,
                 indexlist=None,
                 outputfolder=r'F:\08_Classification\test',
                 coords=True,
                 overwrite=False):

        # TODO: remove this part
        if indexlist is None:
            indexlist = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']
        self.model = model
        self.zone = zone
        self.tile = tile
        self.imagefolder = imagefolder
        self.cirange = cirange
        self.indexlist = indexlist
        self.outputfolder = outputfolder
        self.coords = coords
        self.overwrite = overwrite
        self._make_extlist()
        self._make_outpath()
        self._check_outpath()

    def _make_extlist(self):
        """
        function to create list of dataframe columns
        """
        if self.cirange:
            extlist = ['_slp', '_ict', '_cil', '_ciu', '_cirange']
        else:
            extlist = ['_slp', '_ict', '_cil', '_ciu']
        self.columns_ = combine_idxlist(self.indexlist, extlist)

    def _make_outpath(self):
        """
        create output file names
        """
        #self.outdir_ = r'{p}\{z}'.format(p=self.outputfolder, z=self.zone)
        self.outdir_ = r'{p}'.format(p=self.outputfolder)

        self.outfile_class_ = r'{p}\{z}_class_{t}_class.tif'.format(p=self.outdir_, t=self.tile, z=self.zone)
        self.outfile_proba_ = r'{p}\{z}_class_{t}_proba.tif'.format(p=self.outputfolder, t=self.tile, z=self.zone)
        self.outfile_confidence_ = r'{p}\{z}_class_{t}_confidence.tif'.format(p=self.outputfolder, t=self.tile, z=self.zone)

    def _check_outpath(self):
        self.outfile_class_exists_ = os.path.exists(self.outfile_class_)
        self.outfile_proba_exists_ = os.path.exists(self.outfile_proba_)
        self.outfile_confidence_exists_ = os.path.exists(self.outfile_confidence_)
        self.all_exists_ = all([self.outfile_class_exists_, self.outfile_proba_exists_, self.outfile_confidence_exists_])

    def load_raster_for_classify(self):
        """
        function to load raster data (trends) into dataframe
        """
        if not all([self.all_exists_, ~self.overwrite]):
            #data_full = []
            # TODO: solve index and load from one file
            #for e in self.indexlist:
            f = os.path.join(self.imagefolder, f'trendimage_{self.zone}_{self.tile}.tif')
            self.prototype_ = f
            # FIX
            with rasterio.open(f) as src:
                self.xsize = src.height
                self.ysize = src.width
                data = src.read()
            #data = ga.LoadFile(f, xsize=1000, ysize=1000)
            """
            if self.cirange:
                dt = data[-1] - data[-2]
                data = np.vstack((data, np.expand_dims(dt, 0)))
                #data_full.append(data)
            """
            #data_full = np.array(data_full)
            data_full = data
            shp = data_full.shape
            #data_full = data_full.reshape(shp[0]*shp[1],-1).T
            data_full = data_full.reshape(shp[0],-1).T
            #self.data = pd.DataFrame(columns=self.columns_, data=data_full)
            self.data = pd.DataFrame(data=data_full).fillna(0)


    def classify(self):
        """
        function to classify loaded raster based on pre-trained classification model
        """

        # FIX
        if not all([self.all_exists_, ~self.overwrite]):
            shp = [0, 0, self.xsize, self.ysize]
            self.prediction_class_ = self.model.predict(self.data).reshape(shp[2],shp[3])
            self.prediction_proba_ = self.model.predict_proba(self.data).T.reshape(len(self.model.classes_),shp[2],shp[3])
            self.prediction_confidence_ = self.prediction_proba_.max(axis=0)

    def write_output(self):
        """
        function to write output to GeoTiff raster files
        """
        if not all([self.all_exists_, ~self.overwrite]):
            # TODO: FIX - does not check proper path
            if not os.path.exists(self.outdir_):
                os.mkdir(self.outdir_)
            array_to_file(self.prediction_class_, self.outfile_class_,
                 self.prototype_, dtype=gdal.GDT_Int16, compress=False)
            array_to_file(self.prediction_proba_, self.outfile_proba_,
                 self.prototype_, dtype=gdal.GDT_Float32, compress=False)
            array_to_file(self.prediction_confidence_, self.outfile_confidence_,
                 self.prototype_, dtype=gdal.GDT_Float32, compress=False)

class ClassifyDEM(Classify):
    """
    This is a class to classify trend data and write output to raster files
    :param model: scikit-learn Classifier Object
    :param zone: project specific location identifier
    :param zone: location specific tile identifier
    :param imagefolder: path of trendimages to load for classification
    :param cirange: Boolean to set if confidence ranges should be included (upper CI minus lower CI)
    :param indexlist: list of used indices
    :param outputfolder: path to save output files (files will be saved into subfolder named after zone identifier)
    """

    def load_raster_for_classify(self):
        """
        function to load raster data (trends) into dataframe
        """
        data_full = []
        for e in self.indexlist:
            f = self.imagefolder + r'\trendimage_{zone}_{0}_{1}.tif'.format(self.tile, e, zone=self.zone)
            self.prototype_=f
            # FIX
            with rasterio.open(f) as src:
                self.xsize = src.height
                self.ysize = src.width
                data = src.read()
            #data = ga.LoadFile(f, xsize=1000, ysize=1000)
            if self.cirange:
                dt = data[-1] - data[-2]
                data = np.vstack((data, np.expand_dims(dt, 0)))
            data_full.append(data)
        data_full = np.array(data_full)
        shp_trends = data_full.shape
        data_full = data_full.reshape(shp_trends[0]*shp_trends[1],-1).T
        self.data = pd.DataFrame(columns=self.columns_, data=data_full)

        dem_dir = study_sites[self.zone]['dem_dir']
        for index in ['dem', 'slope']:
            filepath = os.path.join(dem_dir, '{zone}_{pr}_{idx}.tif'.format(zone=self.zone, pr=self.tile, idx=index))
            data = ga.LoadFile(filepath, xsize=1000, ysize=1000)
            self.data[index] = data.ravel()

        if self.coords:
            lon, lat = coord_raster(f, epsg_in=study_sites[self.zone]['epsg'])
            self.data['lat'] = lat.ravel()
            self.data['lon'] = lat.ravel()




class GroundTruth(object):
    """

    """
    def __init__(self,
                 zone,
                 gt_file,
                 study_sites,
                 image_dir,
                 cirange=True,
                 indexlist=None,
                 image_dir_style=r'1999-2014\tiles'):

        if indexlist is None:
            indexlist = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']
        self.zone = zone
        self.gt_file = gt_file
        self.study_sites = study_sites
        self.image_dir = image_dir
        self.cirange = cirange
        self.indexlist = indexlist
        self.ids = image_dir_style
        self._make_extlist()
        self._make_indir()
        self._make_filelist()
        self._load_vector_file()
        self._join_and_filter_dataframes()
        self._find_valid_pathrow()
        self._add_empty_value_columns()
        self._add_empty_value_columns()
        self._add_zone_id()


    def _make_extlist(self):
        """
        function to create list of dataframe columns
        """
        if self.cirange:
            extlist = ['_slp', '_ict', '_cil', '_ciu']
        else:
            extlist = ['_slp', '_ict', '_cil', '_ciu']
        self.columns_ = combine_idxlist(self.indexlist, extlist)

    def _make_indir(self):
        """
        function to setup path to image directory
        :return:
        """
        self.in_dir_ = os.path.join(self.study_sites[self.zone]['result_dir'], self.ids)
        print(self.in_dir_)
        if not os.path.exists(self.in_dir_):
            raise ValueError("Input data directory does not exist")

    def _make_filelist(self):
        """
        function to create input file paths
        """
        self.flist_ = glob.glob(os.path.join(self.in_dir_, '*.tif'))
        self.fn_file_ = self.study_sites[self.zone]['fishnet_file']

    def _load_vector_file(self):
        """
        function to load vector/shapefiles (Ground Truth + Fishnet) to GeoPandas data frame
        """
        # load shapefiles to dataframes, reproject to 4326 if necessary
        self.gp_df_groundtruth = geopandas.GeoDataFrame.from_file(self.gt_file)
        self.gp_df_fishnet = geopandas.GeoDataFrame.from_file(self.fn_file_).to_crs({'init':'epsg:4326'})

    def _join_and_filter_dataframes(self):
        """
        function to join vector files and filter ground truth to only spatially overlapping entries
        """
        #TODO: crash if no overlap
        df_joined = geopandas.tools.sjoin(self.gp_df_groundtruth, self.gp_df_fishnet)
        self.df_joined_filt_ = df_joined[list(self.gp_df_groundtruth.columns) + ['row', 'path']]
        self.df_joined_filt_.loc[:, 'pr_string'] = self.df_joined_filt_.apply(merge_pr, axis=1)
        # reproject dataframe to current zone (UTM) - maybe change to study_sites based
        self.df_temp_reprojected_ = self.df_joined_filt_.to_crs({'init':'epsg:{epsg}'.format(epsg=self.study_sites[self.zone]['epsg'])})

    def _find_valid_pathrow(self):
        """
        function to extract only path_row loacations where GT exist
        """
        self.valid_pr_ = self.df_joined_filt_['pr_string'].unique()

    def _add_empty_value_columns(self):
        """
        function to add empty value columns(trend values)
        """
        self.df_joined_filt_ = pd.concat([self.df_joined_filt_, pd.DataFrame(columns=self.columns_)])

    def _add_zone_id(self):
        self.df_joined_filt_['zone'] = self.zone

    def load_data(self):
        """
        function to load ground truth data into dataframe.
        """
        for pr in self.valid_pr_:
            # create filtered dataframe of GT points only within currently selected tile
            local_df = self.df_joined_filt_[self.df_joined_filt_['pr_string']==pr].to_crs({'init':'epsg:{epsg}'.format(epsg=self.study_sites[self.zone]['epsg'])})
            # get UTM coordinates
            crds = [(crd.x, crd.y) for crd in local_df.geometry.values]

            # iterate over index (Tasselled Cap etc.)
            for idx in self.indexlist:
                # create list of feature/column names
                feat_name = [idx + ext for ext in ('_slp', '_ict', '_cil', '_ciu')]
                # find tile and index specific trend image
                f = glob.glob(r'{idir}\trendimage_*{pr}*{index}*'.format(pr=pr,
                                                                         idir=self.in_dir_,
                                                                         index=idx))

                # check if image is found
                if len(f) != 0:
                    # open file and get raster values from Ground Truth points
                    with rasterio.open(f[0]) as src:
                        v = [smp for smp in src.sample(crds)]
                        # create temporary data frame and update record in parent file for the zone
                        xx = pd.DataFrame(data=v, columns=feat_name, index=local_df.index)
                        self.df_joined_filt_.update(xx, join='left')
                if self.cirange:
                    self.df_joined_filt_[idx+'_cirange'] = self.df_joined_filt_[idx+'_ciu'] - self.df_joined_filt_[idx+'_cil']

class GroundTruthDEM(GroundTruth):
    def __init__(self,
                 zone,
                 gt_file,
                 study_sites):
        self.zone = zone
        self.gt_file = gt_file
        self.study_sites = study_sites
        self._make_extlist()
        self._make_indir()
        self._make_filelist()
        self._load_vector_file()
        self._join_and_filter_dataframes()
        self._find_valid_pathrow()
        self._add_empty_value_columns()

    def _make_extlist(self):
        """
        function to create list of dataframe columns
        """
        self.columns_ = ['dem', 'slope']

    def _make_indir(self):
        """
        function to setup path to image directory
        :return:
        """
        self.in_dir_ = self.study_sites[self.zone]['dem_dir']
        print(self.in_dir_)
        if not os.path.exists(self.in_dir_):
            raise ValueError("Input data directory does not exist")

    def _make_filelist(self):
        """
        function to create input file paths
        """
        self.flist_ = glob.glob(os.path.join(self.in_dir_, '*.tif'))
        self.fn_file_ = self.study_sites[self.zone]['fishnet_file']


    def load_data(self):
        """
        function to load ground truth data into dataframe.
        """
        for pr in self.valid_pr_:
            # create filtered dataframe of GT points only within currently selected tile
            local_df = self.df_joined_filt_[self.df_joined_filt_['pr_string']==pr].to_crs({'init':'epsg:{epsg}'.format(epsg=self.study_sites[self.zone]['epsg'])})
            # get UTM coordinates
            crds = [(crd.x, crd.y) for crd in local_df.geometry.values]
            #
            for index in ['dem', 'slope']:

                #xx = self._load_aux_data(pr, crds, index, local_df)
                #self.df_joined_filt_.update(xx, join='left')

                filepath_ = os.path.join(self.in_dir_, '{zone}_{pr}_{idx}.tif'.format(zone=self.zone, pr=pr, idx=index))
                try:
                    with rasterio.open(filepath_) as src:
                        v = [smp for smp in src.sample(crds)]
                        # create temporary data frame and update record in parent file for the zone
                        xx = pd.DataFrame(data=v, columns=[index], index=local_df.index)
                        self.df_joined_filt_.update(xx, join='left')
                except:
                    pass


    def _load_aux_data(self, pr, crds, index, local_df):
        filepath_ = os.path.join(self.in_dir_, '{zone}_{pr}_{idx}.tif'.format(zone=self.zone, pr=pr, idx=index))
        try:
            with rasterio.open(filepath_) as src:
                v = [smp for smp in src.sample(crds)]
                # create temporary data frame and update record in parent file for the zone
                xx = pd.DataFrame(data=v, columns=[index], index=local_df.index)
                return xx
        except:
            pass
