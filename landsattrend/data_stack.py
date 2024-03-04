import datetime
import glob
import os

import numpy as np
import pandas as pd
from osgeo import gdal_array as ga, gdal

import landsattrend.utils
from landsattrend.config_study_sites import study_sites
from landsattrend.utils import get_datafolder, global_to_local_coords, reproject_coords

gdal.UseExceptions()

from .utils import sensorlist


class DataStack(object):
    def __init__(self, infolder, filetype='tif', indices=None,
                 xoff=0, yoff=0, xsize=None, ysize=None, factor=10000.,
                 startmonth=7, endmonth=8,
                 startyear=1985, endyear=2017, tc_sensor='auto'):
        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']
        self.infolder = infolder
        self.filetype = filetype
        self.indices = indices
        self.xoff = xoff
        self.yoff = yoff
        self.xsize = xsize
        self.ysize = ysize
        self.factor = factor
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.startyear = startyear
        self.endyear = endyear
        self.tc_sensor = tc_sensor
        self.indices_calculated = False
        self.infiles_exist = False
        self.data_stack = None
        self.index_data_grouped = None
        self.func_wrapper_1()

    def func_wrapper_1(self):
        self._create_dataframe()
        self._make_filelist()
        if self.infiles_exist:
            self._file_validation_check_raster()
            self._file_validation_check_name()
            self._filter_invalid_files()
            self._fill_dataframe()
            self._sort_filelist()
            self._file_filter_check()
            self._apply_file_filter()

    def load_data(self):
        self._load_stack()
        self._calc_indices()

    # TODO add grouping for spectral
    def group_data(self, attribute='year', type='median'):
        """
        function to group data/indices by common temporal attributes (e.g. year)
        :param attribute: str - attribute name to group on {'year', 'month', 'day'}
        :param type: str - type of grouping {'median', 'mean', 'max, 'min'}
        :return:
        """
        self.index_data_grouped = {}
        for index in self.indices:
            df_indexdata = pd.DataFrame(data=self.index_data[index].reshape(len(self.df_indata), -1), index=self.df_indata.index)
            shp = self.index_data[index].shape
            joined = self.df_indata[[attribute]].join(df_indexdata)
            if type == 'median':
                grouped = joined.groupby(by=attribute, axis=0).median().sort_index()
            elif type == 'mean':
                grouped = joined.groupby(by=attribute, axis=0).mean().sort_index()
            elif type == 'max':
                grouped = joined.groupby(by=attribute, axis=0).max().sort_index()
            elif type == 'min':
                grouped = joined.groupby(by=attribute, axis=0).min().sort_index()
            m = grouped.as_matrix().T
            self.grouped_feature = grouped.index.values
            self.index_data_grouped[index] = np.ma.MaskedArray(data=m, mask=np.isnan(m)).T.reshape(len(self.grouped_feature), shp[1], shp[2])

    def _create_dataframe(self):
        """
        Create empty pandas DataFrame
        :return:
        """
        columns = ['basename', 'filepath', 'datetime', 'year', 'month', 'day', 'doy', 'ordinal_day', 'sensor', 'timestamp',
                   'infiles_valid_raster', 'infiles_valid_name', 'infiles_valid_filter', 'infiles_naming_version', 'process', 'file_loaded']
        self.df_indata_raw = pd.DataFrame(columns=columns)
        self.df_indata = pd.DataFrame(columns=columns)


    def _make_filelist(self):
        """
        make list of input folder within indicated folder
        :return:
        """
        self.df_indata_raw.filepath = glob.glob('{0}/*.{1}'.format(self.infolder, self.filetype))
        self.df_indata_raw.basename = np.array([os.path.basename(f) for f in self.df_indata_raw.filepath])
        self.infiles_exist = len(self.df_indata_raw.filepath) > 0

    # TODO: Create consistency with self._get_datetime()
    def _fill_dataframe(self):
        """
        Fill empty dataframe with file properties
        :return:
        """
        self.df_indata.timestamp = np.array([datetime.datetime.fromtimestamp(os.path.getmtime(f)) for f in self.df_indata.filepath])
        self._get_datetime()
        self.df_indata.loc[:, ['datetime']] = pd.to_datetime(self.df_indata['datetime'])
        self.df_indata.loc[:, ['year']] = self.df_indata['datetime'].dt.year
        self.df_indata.loc[:, ['month']] = self.df_indata['datetime'].dt.month
        self.df_indata.loc[:, ['day']] = self.df_indata['datetime'].dt.day
        self.df_indata.loc[:, ['doy']] = self.df_indata['datetime'].dt.dayofyear
        self.df_indata.loc[:, ['ordinal_day']] = np.array([f.toordinal() for f in self.df_indata['datetime']])
        self.df_indata.loc[:, ['sensor']] = np.array(sensorlist(self.df_indata.basename))
        self.df_indata.loc[:, ['process']] = False

    def _sort_filelist(self):
        """
        sort all input files by date (ascending)
        :return:
        """
        self.df_indata = self.df_indata.sort_values(by=['datetime'])

    def _check_ls_naming_version(self):
        pass

    @staticmethod
    def _get_datetime_v1(basename):
        basename = basename.split('_')[0]
        return datetime.datetime.strptime("{0}-{1}".format(basename[9:13], basename[13:16]), "%Y-%j")

    @staticmethod
    def _get_datetime_v2(basename):
        return datetime.datetime.strptime("{0}-{1}-{2}".format(basename[10:14], basename[14:16],
                                                               basename[16:18]), "%Y-%m-%d")

    def _get_datetime(self):
        """
        private function to get datetme from filename
        Checks for old and new naming convention
        :return:
        """
        v1 = self.df_indata[self.df_indata.infiles_naming_version == 'v1']
        self.df_indata.loc[v1.index, 'datetime'] = pd.to_datetime(v1.basename.apply(self._get_datetime_v1))
        v2 = self.df_indata[self.df_indata.infiles_naming_version == 'v2']
        self.df_indata.loc[v2.index, 'datetime'] = pd.to_datetime(v2.basename.apply(self._get_datetime_v2))

    def _file_validation_check_raster(self):
        """
        Check integrity of input file to avoid error on loading
        :return:
        """
        self.df_indata_raw.infiles_valid_raster = [self._is_gdal_dataset(f) for f in self.df_indata_raw['filepath']]
        self.df_indata_raw = self.df_indata_raw[self.df_indata_raw.infiles_valid_raster]

    def _file_validation_check_name(self):
        """
        Check integrity of input file name to avoid error on loading
        Check landsat ESPA naming version
        :return:
        """
        valid_name = []
        naming_version = []

        for f in self.df_indata_raw['basename']:
            if len(f.split('_')) == 5:
                valid_name.append(True)
                if len(f.split('_')[0]) == 16:
                    naming_version.append('v1')
                elif len(f.split('_')[0]) == 22:
                    naming_version.append('v2')
                else:
                    naming_version.append(None)
                    valid_name.append(False)
            else:
                naming_version.append(None)
                valid_name.append(False)

        self.df_indata_raw.infiles_valid_name = valid_name
        self.df_indata_raw.infiles_naming_version = naming_version

    @staticmethod
    def _is_gdal_dataset(filepath):
        """
        Function to check if file is valid gdal-dataset
        :param filepath: str
        :return: bool
        """
        try:
            src = gdal.Open(filepath)
            is_dataset = src is not None
            src = None
        except:
            is_dataset = False
        return is_dataset

    def _filter_invalid_files(self):
        valid = self.df_indata_raw.loc[:,['infiles_valid_name', 'infiles_valid_raster']].all(axis=1)
        self.df_indata = self.df_indata_raw.loc[valid]

    def _file_filter_check(self):
        """
        Check if files comply with filter settings (e.g. dates)
        :return:
        """
        # TODO: close dataset properly
        self.df_indata.loc[:, 'infiles_valid_filter'] = np.all([self.df_indata.month.isin(list(range(self.startmonth, self.endmonth+1))),
                                                self.df_indata.year.isin(list(range(self.startyear, self.endyear+1)))],
                                                axis=0)

    def _apply_file_filter(self):
        """
        reduce dataframe to valid files
        :return:
        """
        self.df_indata.loc[:, 'process'] = self.df_indata[['infiles_valid_filter', 'infiles_valid_raster', 'infiles_valid_name']].all(axis=1)
        self.df_indata = self.df_indata[self.df_indata['process']]

    def _load_stack(self):
        """
        load data to 4-D data stack. dim1: date, dim2: spectral band, dim3: row, dim4: path
        :return:
        """
        # TODO check for errors
        #self.data_stack = np.array([ga.LoadFile(f, xoff=self.xoff, xsize=self.xsize, yoff=self.yoff, ysize=self.ysize) for f in self.df_indata.filepath])
        data_stack = []
        for index, row in self.df_indata.iterrows():
            d = ga.LoadFile(row['filepath'], xoff=self.xoff, xsize=self.xsize, yoff=self.yoff, ysize=self.ysize)
            if isinstance(d, np.ndarray):
                self.df_indata.loc[index, 'file_loaded'] = True
                data_stack.append(d)
            else:
                continue
        self.data_stack = np.ma.masked_equal(np.asarray(data_stack), 0)
        self.data_stack = self.data_stack / self.factor
        self.df_indata = self.df_indata[self.df_indata['file_loaded'] == True]

    def _calc_indices(self):
        """
        calculate indices
        :return:
        """
        self.index_data = {}
        if 'tcb' or 'tcg' or 'tcw' in self.indices:
            if self.tc_sensor == 'auto':
                tc = np.array([landsattrend.utils.tasseled_cap(i, s) for i, s in zip(self.data_stack, self.df_indata.sensor)])
            elif self.tc_sensor in ['TM', 'ETM', 'OLI', 'OLI_TIRS']:
                tc = np.array([landsattrend.utils.tasseled_cap(i, self.tc_sensor) for i in self.data_stack])
            else:
                raise ValueError("Please use one of following values {'auto', 'TM', 'ETM', 'OLI', 'OLI_TIRS'")

            tc = np.ma.masked_equal(tc, 0)
            self.index_data['tcb'] = tc[:, 0]
            self.index_data['tcg'] = tc[:, 1]
            self.index_data['tcw'] = tc[:, 2]

        if 'ndvi' in self.indices:
            ndvi = [landsattrend.utils.ndvi(i, 3, 2) for i in self.data_stack]
            self.index_data['ndvi'] = np.ma.masked_equal(ndvi, 0)

        if 'ndwi' in self.indices:
            ndwi = [landsattrend.utils.ndvi(i, 1, 3) for i in self.data_stack]
            self.index_data['ndwi'] = np.ma.masked_equal(ndwi, 0)

        if 'ndmi' in self.indices:
            ndmi = [landsattrend.utils.ndvi(i, 3, 4) for i in self.data_stack]
            self.index_data['ndmi'] = np.ma.masked_equal(ndmi, 0)

        if 'ndbr' in self.indices:
            ndbr = [landsattrend.utils.ndvi(i, 3, 5) for i in self.data_stack]
            self.index_data['ndbr'] = np.ma.masked_equal(ndbr, 0)

        self.indices_calculated = True


class DataStackList(DataStack):
    def __init__(self, inlist, infolder, filetype='tif', indices=None, xoff=0,
                 yoff=0, xsize=None, ysize=None, nodata=0, factor=10000., startmonth=7, endmonth=8, startyear=1985,
                 endyear=2014):
        super(DataStackList, self).__init__(infolder, filetype, indices, xoff, yoff, xsize, ysize, nodata, factor,
                                            startmonth, endmonth, startyear, endyear)
        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']
        self.inlist = inlist
        self.filetype = filetype
        self.indices = indices
        self.xoff = xoff
        self.yoff = yoff
        self.xsize = xsize
        self.ysize = ysize
        self.factor = factor
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.startyear = startyear
        self.endyear = endyear
        self.indices_calculated = False
        self.infiles_exist = False
        self.func_wrapper_1()

    def _make_filelist(self):
        """
        make list of input folder within indicated folder
        :return:
        """
        self.df_indata.filepath = np.array([os.path.abspath(f) for f in self.inlist])

# TODO: Improve
def load_point_ts(study_site, coordinates, startmonth=7, endmonth=8, startyear=1999, endyear=2014, infolder=None, **kwargs):
    """
    wrapper function to load Stack of one specific point
    :param study_site: string
    :param coordinates: tuple
    :param startmonth: int
    :param endmonth: int
    :return:
    """
    if infolder is None:
        try:
            infolder = get_datafolder(study_site, coordinates, epsg='auto')
            xout, yout = global_to_local_coords(infolder, coordinates)
        except:
            # transform coords from latlon to local coordinates (e.g. UTM)
            coordinates_tr = reproject_coords(4326, study_sites[study_site]['epsg'], coordinates)
            infolder = get_datafolder(study_site, coordinates_tr)
            # make error handler if file does not exist
            xout, yout = global_to_local_coords(infolder, coordinates_tr)
    else:
        xout, yout = global_to_local_coords(infolder, coordinates)

    ds = DataStack(infolder=infolder, xoff=xout, yoff=yout, xsize=1, ysize=1,
                   startmonth=startmonth, endmonth=endmonth,
                   startyear=startyear, endyear=endyear, **kwargs)
    ds.load_data()
    # TODO reorganize to single DF with data
    # Check function --> def _group_by_year(self, index)
    return ds
    """
    df = ds.df_indata
    for k in list(ds.index_data.keys()):
        df.loc[:, k] = np.squeeze(ds.index_data[k])
        df.loc[:,'mask'] = df[k] != 0
        #df.mask(df[k] == 0, inplace=True)
    return df[df['mask']]
    """