import datetime
import glob
import os

import numpy as np
import pandas as pd
from osgeo import gdal_array as ga

from . import lstools
from .helper_funcs import sensorlist


class DataStack(object):
    def __init__(self, infolder, filetype='tif', indices=['tcb', 'tcg','tcw', 'ndvi', 'ndwi', 'ndmi'],
                 xoff=0, yoff=0, xsize=None, ysize=None, nodata=0, factor=10000.,
                 startmonth=7, endmonth=8,
                 startyear=1985, endyear=2017, tc_sensor='auto'):
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
        self.func_wrapper_1()

    def func_wrapper_1(self):
        self.create_dataframe()
        self.make_filelist()
        if self.infiles_exist:
            self.fill_dataframe()
            self.sort_filelist()
            self.filter_filelist()

    def load_data(self):
        self.load_stack()
        self.calc_indices()

    def create_dataframe(self):
        self.df_indata = pd.DataFrame(columns=['basename', 'filepath', 'dt', 'year', 'month', 'day', 'doy', 'ordinal_day',
                                               'sensor', 'timestamp', 'process'])

    def make_filelist(self):
        """
        make list of input folder within indicated folder
        :return:
        """
        self.df_indata.filepath = glob.glob('{0}/*.{1}'.format(self.infolder, self.filetype))
        self.infiles_exist = len(self.df_indata.filepath) > 0

    def fill_dataframe(self):
        self.df_indata.basename = np.array([os.path.basename(f) for f in self.df_indata.filepath])
        self.df_indata.timestamp = np.array([datetime.datetime.fromtimestamp(os.path.getmtime(f)) for f in self.df_indata.filepath])
        self.df_indata.dt = self._get_datetime()
        self.df_indata.year = self.df_indata.dt.dt.year
        self.df_indata.month = self.df_indata.dt.dt.month
        self.df_indata.day = self.df_indata.dt.dt.day
        self.df_indata.doy = self.df_indata.dt.dt.dayofyear
        self.df_indata.ordinal_day = np.array([f.toordinal() for f in self.df_indata.dt])
        self.df_indata.sensor = np.array(sensorlist(self.df_indata.basename))
        self.df_indata.process = False

    def _get_datetime(self):
        """
        private function to get datetme from filename
        Checks for old and new naming convention
        :return:
        """
        dt = []
        for f in self.df_indata.basename:
            if len(f.split('_')[0]) == 16:
                dt.append(datetime.datetime.strptime("{0}-{1}".format(f[9:13], f[13:16]), "%Y-%j"))
            elif len(f.split('_')[0]) == 22:
                dt.append(datetime.datetime.strptime("{0}-{1}-{2}".format(f[10:14], f[14:16], f[16:18]), "%Y-%m-%d"))
            else:
                pass
        return np.array(dt)

    def filter_filelist(self):
        self.df_indata.process = np.all([self.df_indata.month.isin(list(range(self.startmonth, self.endmonth+1))),
                                         self.df_indata.year.isin(list(range(self.startyear, self.endyear+1)))],
                                        axis=0)
        self.df_indata = self.df_indata[self.df_indata.process]

    def sort_filelist(self):
        """
        sort all input files by date (ascending)
        :return:
        """
        self.df_indata.sort_values(by=['dt'], inplace=True)

    def load_stack(self):
        """
        load data to 4-D data stack. dim1: date, dim2: spectral band, dim3: row, dim4: path
        :return:
        """
        # TODO check for errors
        self.data_stack = np.array([ga.LoadFile(f, xoff=self.xoff, xsize=self.xsize, yoff=self.yoff, ysize=self.ysize) for f in self.df_indata.filepath])
        self.data_stack = np.ma.masked_equal(self.data_stack, 0)
        self.data_stack = self.data_stack / self.factor

    def calc_indices(self):
        """
        calculate indices
        :return:
        """
        self.index_data = {}
        if 'tcb' or 'tcg' or 'tcw' in self.indices:
            if self.tc_sensor == 'auto':
                tc = np.array([lstools.tasseled_cap(i, s) for i, s in zip(self.data_stack, self.df_indata.sensor)])
            elif self.tc_sensor in ['TM', 'ETM', 'OLI', 'OLI_TIRS']:
                tc = np.array([lstools.tasseled_cap(i, self.tc_sensor) for i in self.data_stack])
            else:
                raise ValueError("Please use one of following values {'auto', 'TM', 'ETM', 'OLI', 'OLI_TIRS'")

            tc = np.ma.masked_equal(tc, 0)
            self.index_data['tcb']= tc[:, 0]
            self.index_data['tcg']= tc[:, 1]
            self.index_data['tcw']= tc[:, 2]

        if 'ndvi' in self.indices:
            ndvi = [lstools.ndvi(i, 3, 2) for i in self.data_stack]
            self.index_data['ndvi']= np.ma.masked_equal(ndvi, 0)

        if 'ndwi' in self.indices:
            ndwi = [lstools.ndvi(i, 1, 3) for i in self.data_stack]
            self.index_data['ndwi']= np.ma.masked_equal(ndwi, 0)

        if 'ndmi' in self.indices:
            ndmi = [lstools.ndvi(i, 3, 4) for i in self.data_stack]
            self.index_data['ndmi']= np.ma.masked_equal(ndmi, 0)

        if 'ndbr' in self.indices:
            ndbr = [lstools.ndvi(i, 3, 5) for i in self.data_stack]
            self.index_data['ndbr']= np.ma.masked_equal(ndbr, 0)

        self.indices_calculated = True


class DataStackList(DataStack):
    def __init__(self, inlist, infolder, filetype='tif', indices=['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi'], xoff=0,
                 yoff=0, xsize=None, ysize=None, nodata=0, factor=10000., startmonth=7, endmonth=8, startyear=1985,
                 endyear=2014):
        super(DataStackList, self).__init__(infolder, filetype, indices, xoff, yoff, xsize, ysize, nodata, factor,
                                            startmonth, endmonth, startyear, endyear)
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

    def make_filelist(self):
        """
        make list of input folder within indicated folder
        :return:
        """
        self.df_indata.filepath = np.array([os.path.abspath(f) for f in self.inlist])

