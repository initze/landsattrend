import datetime
import os
import time

import fiona
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from osgeo import osr, gdal

from landsattrend.breakpoint import Breakpoint
from landsattrend.config_study_sites import wrs2_path, study_sites
from landsattrend.data_stack import DataStack
from landsattrend.utils import get_foldernames, array_to_file, tiling, trend_image2
from landsattrend.version import __version__

__author__ = 'initze'


class Process(object):
    def __init__(self, study_site, outfolder, infolder=None, indices=None,
                 prefix='trendimage',
                 startyear=1985, endyear=2014, startmonth=7, endmonth=8,
                 path=None, row=None, pr_string=None,
                 parallel=True, tile_size=250,
                 tc_sensor='auto',
                 n_jobs=-1):

        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi', 'nobs']
        self.study_site = study_site
        self.outfolder = outfolder
        self.infolder = infolder
        self.indices = np.array(indices)
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.prefix = prefix
        self.row = row
        self.path = path
        self.pr_string = pr_string
        self.parallel = parallel
        self.tile_size = tile_size
        self.n_jobs = n_jobs
        self.tc_sensor = tc_sensor
        self.data = None
        self.report_file = None
        self.df_outdata = None
        self.ss_name = study_sites[study_site]['name']
        self.ss_indir = study_sites[study_site]['processing_dir']
        self.infiles_check = False
        self.outfiles_check = False
        self.start_time = time.time()
        self._startup_funcs1()
        if self.infiles_check:
            self._startup_funcs2()
        if self.outfiles_check:
            self._startup_funcs3()

    def load_data(self, i=0):
        """
        Load input data and indices using DataStack
        :param i:
        :return:
        """
        # print 'loading Data'
        self.data = DataStack(self.infolder, indices=self.indices_process,
                              xoff=int(self.coff[i]), xsize=int(self.csize[i]),
                              yoff=int(self.roff[i]), ysize=int(self.rsize[i]),
                              startmonth=self.startmonth, endmonth=self.endmonth,
                              startyear=self.startyear, endyear=self.endyear, tc_sensor=self.tc_sensor)
        self.data.load_data()

    def _startup_funcs1(self):
        self._check_pr()
        self._setup_infolder()
        self._make_infile_list()
        self._print_info()

    def _startup_funcs2(self):
        self._make_outfile_list()
        self._setup_outfolder()
        self._set_processing_bool()
        self._final_precheck()

    def _startup_funcs3(self):
        self._get_raster_size(self.infiles.filepath.iloc[0])
        self._subsample()
        self._setup_result_layers()

    def _check_pr(self):
        """
        check and transform Path/Row string
        :return:
        """
        if (self.row is None) and (self.path is None) and (type(self.pr_string) == str):
            self._pr_string_to_pr()
        elif (self.row is None) and (self.path is None) and (type(self.pr_string) != str):
            raise ValueError("Please indicate either row and path or pr_string")

    def _setup_infolder(self):
        """
        Create path for outfolder
        :return:
        """
        if not self.infolder:
            self.infolder = os.path.join(self.ss_indir, '{0}_{1}_{2}'.format(self.ss_name, self.row, self.path))

    def _make_infile_list(self):
        """
        Get infile properties using DataStack without loading data explicitly
        :return:
        """
        print('loading Data')  # fix for skip
        self.infiles = DataStack(self.infolder,
                                 startmonth=self.startmonth, endmonth=self.endmonth,
                                 startyear=self.startyear, endyear=self.endyear).df_indata
        self.infiles_check = len(self.infiles) > 0

    def _print_info(self):
        """
        Small printing wrapper
        :return:
        """
        print("\n\n")
        print("Study Site: {0}".format(self.study_site))
        print("Processing Dataset: {0}".format(self.pr_string))
        if not self.infiles_check:
            print("Skip processing, No Data available for this subset!\n")

    def _final_precheck(self):
        if not self.outfiles_check:
            print("Skip processing, data already exist!")

    def _get_raster_size(self, path):
        """
        Get Rastersize of Raster
        :param path:
        :return:
        """
        ds = gdal.Open(path)
        self.nrows = ds.RasterYSize
        self.ncols = ds.RasterXSize

    def _group_data_by_year(self):
        self.index_data_filt = {}
        [self._group_by_year(idx) for idx in self.indices_process]

    def _group_by_year(self, index):
        """
        :param index:
        :return:
        """
        df_indexdata = pd.DataFrame(data=self.data.index_data[index].reshape(len(self.data.df_indata), -1), index=self.data.df_indata.index)
        shp = self.data.index_data[index].shape
        joined = self.data.df_indata[['year']].join(df_indexdata)
        grouped = joined.groupby(by='year', axis=0).median().sort_index()
        m = grouped.as_matrix().T
        self.years = grouped.index.values
        self.index_data_filt[index] = np.ma.MaskedArray(data=m, mask=np.isnan(m)).T.reshape(len(self.years), shp[1], shp[2])

    def _make_outfile_list(self):
        """
        Create Dataframe with outfile properties, e.g. if already existing, timestamp and if it will be processed
        :return:
        """
        def make_timestamp(x):
            return datetime.datetime.fromtimestamp(os.path.getmtime(x))
        self.df_outdata = pd.DataFrame(index=self.indices, columns=['filepath', 'exists', 'timestamp', 'process'])
        self.df_outdata.filepath = np.array(['{0}_{1}_{2}_{3}_{4}.tif'.format(self.prefix, self.ss_name, self.row,
                                                                              self.path, idx) for idx in self.indices])
        self.df_outdata.filepath = [os.path.join(self.outfolder, f) for f in self.df_outdata.filepath]
        self.df_outdata['exists'] = self.df_outdata['filepath'].apply(os.path.exists)
        self.df_outdata['timestamp'] = datetime.datetime(1800, 1, 2)
        self.df_outdata['timestamp'] = self.df_outdata['filepath'][self.df_outdata['exists']].apply(make_timestamp)
        self.df_outdata['process'] = \
            self.df_outdata['timestamp'][self.df_outdata['exists']] < self.infiles.timestamp.max()
        # TODO: fix this line: throws warning at runtime
        self.df_outdata['process'][~self.df_outdata['exists']] = True

    def _pr_string_to_pr(self):
        """
        make row/path integers from string
        :return:
        """
        self.row, self.path = np.array(self.pr_string.split('_'), dtype=np.int)

    def _set_processing_bool(self):
        """
        Create boolean variables to check for later processing
        :return:
        """
        self.outfiles_check = any(self.df_outdata.process)
        x = self.indices[self.indices != 'nobs']
        self.indices_process = np.array(self.df_outdata.loc[x][self.df_outdata.loc[x]['process']].index)
        self.nobs_process = False
        if 'nobs' in self.df_outdata.index:
            self.nobs_process = self.df_outdata.loc['nobs', 'process']

    def _setup_outfolder(self):
        """
        Check if outfolder exists, create otherwise
        :return:
        """
        if not os.path.isdir(self.outfolder):
            os.makedirs(self.outfolder)

    def _setup_result_layers(self):
        pass

    def _subsample(self):
        """
        setup subsampling coordinates for tiled-processing
        :return:
        """
        self.roff, self.coff, self.rsize, self.csize = tiling(self.nrows, self.ncols, self.tile_size, self.tile_size)
        self.ntiles = len(self.roff)


# TODO: reporting if completely new data arrived
class Processor(Process):
    def __init__(self, study_site, outfolder, infolder=None, indices=None,
                 startyear=1985, endyear=2014, startmonth=7, endmonth=8,
                 naming_convention='old', prefix='trendimage',
                 path=None, row=None, pr_string=None,
                 rescale_intercept=datetime.datetime(2014, 7, 1),
                 nobs=False, tc_sensor='auto', ts_mode='full',
                 parallel=True, tile_size=250,
                 n_jobs=12):

        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi', 'nobs']
        self.study_site = study_site
        self.outfolder = outfolder
        self.infolder = infolder
        self.indices = np.array(indices)
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.naming_convention = naming_convention
        self.prefix = prefix
        self.row = row
        self.path = path
        self.pr_string = pr_string
        self.rescale_intercept = rescale_intercept
        self.nobs = nobs
        self.tc_sensor = tc_sensor
        self.ts_mode = ts_mode
        self.parallel = parallel
        self.tile_size = tile_size
        self.n_jobs = n_jobs
        self.data = None
        self.report_file = None
        self.df_outdata = None
        self.ss_name = study_sites[study_site]['name']
        self.ss_indir = study_sites[study_site]['processing_dir']
        self.infiles_check = False
        self.outfiles_check = False
        self.start_time = time.time()
        self._startup_funcs1()

        if self.infiles_check:
            self._startup_funcs2()

        if self.outfiles_check:
            self._startup_funcs3()



    def calculate_trend(self):
        print("Index: {0}".format(' '.join(self.df_outdata[self.df_outdata['process']].index)))
        # TODO: insert here if new files arrived after last processing
        if self.outfiles_check:
            print("Parallel Processing of trends with {0} CPUs".format(self.n_jobs))
            for i in range(self.ntiles):
                if self.ts_mode == 'full':
                    self._run_calculation_mode_full(i)
                elif self.ts_mode == 'median_year':
                    self._run_calculation_mode_median(i)
                else:
                    raise ValueError("Please choose correct ts_mode. 'full' or 'median_year'")

    def export_result(self):
        if self.outfiles_check:
            self._create_metadata()
            self._export_files()
            print("Full Processing took {0} seconds".format(round(time.time()-self.start_time)))

    def _run_calculation_mode_full(self, i=0):
        print("\nProcessing tile {0}/{1}".format(i+1, self.ntiles))
        self.load_data(i)
        self._rescaling_intercept()
        self._calc_nobs(i)
        if self.parallel:
            self.calc_trend_parallel(i)
        else:
            self.calc_trend(i)

    def _run_calculation_mode_median(self, i=0):
        print("\nProcessing tile {0}/{1}".format(i+1, self.ntiles))
        self.load_data(i)
        self._rescaling_intercept()
        self._group_data_by_year()
        self._calc_nobs_median(i)
        if self.parallel:
            self._calc_trend_parallel_median(i)
        else:
            self._calc_trend_median()

    def _setup_result_layers(self):
        """
        Create result layers as np.arrays wrapped in a dict
        :return:
        """
        self.results = {}
        for i in self.indices_process:
            self.results[i] = np.zeros((4, self.nrows, self.ncols), dtype=np.float)
        if 'nobs' in self.indices:
            self.results['nobs'] = np.zeros((self.nrows, self.ncols), dtype=np.uint16)

    def _rescaling_intercept(self):
        """
        Rescale intercept value to defined date
        :return:
        """
        if self.rescale_intercept:
            self.data.df_indata.ordinal_day -= self.rescale_intercept.toordinal()
            self.data.df_indata.year -= 2014

    def calc_trend(self, i=0):
        """
        Calculate Trend
        :param i:
        :return:
        """
        out = [trend_image2(self.data.index_data[idx], self.data.df_indata.ordinal_day) for idx in self.indices_process]
        ctr = 0
        for idx in self.indices_process:
            self.results[idx][:, self.roff[i]:self.roff[i]+self.rsize[i],
            self.coff[i]:self.coff[i]+self.csize[i]] = out[ctr]
            ctr += 1

    def calc_trend_parallel(self, i=0):
        """
        Calculate Trend with parallel processing
        :param i:
        :return:
        """
        processing_mask = \
            self.results['nobs'][self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]]
        processing_mask = processing_mask >= 6
        out = Parallel(n_jobs=self.n_jobs)\
            (delayed(trend_image2)(self.data.index_data[idx], self.data.df_indata.ordinal_day,
                                   processing_mask=processing_mask) for idx in self.indices_process)
        ctr = 0
        for idx in self.indices_process:
            self.results[idx][:, self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]] = out[ctr]
            ctr += 1



    def _calc_trend_median(self, i=0):
        """
        Calculate Trend with parallel processing
        :param i:
        :return:
        """
        print("Processing of trends")
        # arange data
        out = [trend_image2(self.index_data_filt[idx], self.years, factor=10.) for idx in self.indices_process]
        ctr = 0
        for idx in self.indices_process:
            self.results[idx][:, self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]] = out[ctr]
            ctr += 1

    def _calc_trend_parallel_median(self, i=0):
        """
        Calculate Trend with parallel processing
        :param i:
        :return:
        """
        # print("Parallel Processing of trends with {0} CPUs".format(self.n_jobs))
        # arange data
        processing_mask = self.results['nobs'][self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]]
        processing_mask = processing_mask >= 6
        try:
            out = Parallel(n_jobs=self.n_jobs)(delayed(trend_image2)(self.index_data_filt[idx],
                                                                     self.years, factor=10.,
                                                                     processing_mask=processing_mask)
                                               for idx in self.indices_process)
            ctr = 0
            for idx in self.indices_process:
                self.results[idx][:, self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]] = out[ctr]
                ctr += 1
        except Exception as e:
            # TODO logging
            print(e)
            pass

    def _calc_nobs(self, i=0):
        """
        create layer with number of obervations
        :param i:
        :return:
        """
        if self.nobs_process:
            nobs_out = (~self.data.data_stack.mask[:, 0]).sum(axis=0)
            self.results['nobs'][self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]] = nobs_out

    def _calc_nobs_median(self, i=0):
        """
        create layer with number of obervations
        :param i:
        :return:
        """
        if self.nobs_process:
            nobs_out = (~self.index_data_filt[self.indices_process[0]].mask).sum(axis=0)
            self.results['nobs'][self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]] = nobs_out

    def _create_metadata(self):
        """
        Function to setup Metadata
        :return:
        """
        self.metadata = {}
        for key in list(self.results.keys()):
            self.metadata[key] = {'DESCRIPTION': 'Trend Map of Index: {0}'.format(key),
                                  'CREATION TIMESTAMP': datetime.datetime.fromtimestamp(time.time()).isoformat(),
                                  'PROCESSING_VERSION': __version__,
                                  'REGRESSION_ALGORITHM': 'Theil-Sen',
                                  'DATA_YEARS': '{s}-{e}'.format(s=self.startyear, e=self.endyear),
                                  'DATA_MONTHS': '{s}-{e}'.format(s=self.startmonth, e=self.endmonth),
                                  'DATA_FILTER_MODE': self.ts_mode,
                                  'TASSELED_CAP_MODE': self.tc_sensor}

    def _export_files(self):
        """
        Write calculated output data to Raster files
        :return:
        """
        for idx in self.indices_process:
            print(self.df_outdata.loc[idx]['filepath'])
            array_to_file(self.results[idx],
                          self.df_outdata.loc[idx]['filepath'],
                          self.infiles.filepath.iloc[0], dtype=gdal.GDT_Float32,
                          compress=False, metadata=self.metadata[idx])

        if 'nobs' in list(self.results.keys()):
            array_to_file(self.results['nobs'],
                          self.df_outdata.loc['nobs', 'filepath'],
                          self.infiles.filepath.iloc[0], dtype=gdal.GDT_UInt16,
                          compress=False, metadata=self.metadata['nobs'])

    def make_report(self):
        """
        Make processing report
        :return:
        """
        tstamp = datetime.datetime.fromtimestamp(self.start_time).strftime('%Y%m%d-%H%M%S')
        self.report_file = os.path.join(os.path.abspath(os.path.curdir), 'report_{0}.txt'.format(tstamp))
        text = ''.join(['Processing Report \n',
                       "Index: {0}\n".format(' '.join(self.df_outdata[self.df_outdata['process']].index))])
        with open(self.report_file, 'a') as src:
            src.write(text)
        pass

class ProcessorBreakpoint(Process):
    def __init__(self, study_site, outfolder, infolder=None, indices=None,
                 startyear=1985, endyear=2014, startmonth=7, endmonth=8,
                 naming_convention='old', prefix='trendimage',
                 path=None, row=None, pr_string=None,
                 rescale_intercept=datetime.datetime(2014, 7, 1),
                 nobs=False, tc_sensor='auto', ts_mode='full',
                 parallel=True, tile_size=250,
                 n_jobs=-1):

        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi', 'nobs']
        self.study_site = study_site
        self.outfolder = outfolder
        self.infolder = infolder
        self.indices = np.array(indices)
        self.startyear = startyear
        self.endyear = endyear
        self.startmonth = startmonth
        self.endmonth = endmonth
        self.naming_convention = naming_convention
        self.prefix = prefix
        self.row = row
        self.path = path
        self.pr_string = pr_string
        self.rescale_intercept = rescale_intercept
        self.nobs = nobs
        self.tc_sensor = tc_sensor
        self.ts_mode = ts_mode
        self.parallel = parallel
        self.tile_size = tile_size
        self.n_jobs = n_jobs
        self.data = None
        self.report_file = None
        self.df_outdata = None
        self.ss_name = study_sites[study_site]['name']
        self.ss_indir = study_sites[study_site]['processing_dir']
        self.infiles_check = False
        self.outfiles_check = False
        self.start_time = time.time()
        self._startup_funcs1()
        if self.infiles_check:
            self._startup_funcs2()
        if self.outfiles_check:
            self._startup_funcs3()

    def calculate_breaks(self):
        print("Index: {0}".format(' '.join(self.df_outdata[self.df_outdata['process']].index)))
        # TODO: insert here if new files arrived after last processing
        for i in range(self.ntiles):
            self._run_calculation_mode_median(i)

    def export_result(self):
        if self.outfiles_check:
            self._export_files()
            print("Full Processing took {0} seconds".format(round(time.time()-self.start_time)))

    def _run_calculation_mode_median(self, i=0):
        print("\nProcessing tile {0}/{1}".format(i+1, self.ntiles))
        self.load_data(i)
        self._group_data_by_year()
        self._calc_break_parallel_median(i)

    @staticmethod
    def breakpoint(data, years):
        res = []
        for d in data:
            msk = d.mask
            x, y = years[~msk], d[~msk]
            bp = Breakpoint(x, y, predictor='mae')
            bp.fit()
            res.append(bp.results_best_)
        result = pd.concat(res, ignore_index=True, axis=1).transpose()
        return result

    def _calc_break_parallel_median(self, i=0):
        """
        Calculate Trend with parallel processing
        :param i:
        :return:
        """
        # print("Parallel Processing of trends with {0} CPUs".format(self.n_jobs))
        for idx in self.indices_process:
            data = self.index_data_filt[idx].reshape(len(self.years), -1).T
            try:
                out = Parallel(n_jobs=self.n_jobs)(delayed(self.breakpoint)(d, self.years) for d in np.array_split(data, 50))
                #out = [self.breakpoint(d, self.years) for d in np.array_split(data, 50)]
                out = pd.concat(out)
                tmp = np.asarray(out['break_year1'].values).reshape((int(self.rsize[i]), int(self.csize[i])))
                self.results[idx][self.roff[i]:self.roff[i]+self.rsize[i], self.coff[i]:self.coff[i]+self.csize[i]] = tmp
            except Exception as e:
                # TODO logging
                print(e)
                pass

    def _setup_result_layers(self):
        """
        Create result layers as np.arrays wrapped in a dict
        :return:
        """
        self.results = {}
        for i in self.indices_process:
            self.results[i] = np.zeros((self.nrows, self.ncols), dtype=np.float)

    def _export_files(self):
        """
        Write calculated output data to Raster files
        :return:
        """
        for idx in self.indices_process:
            print(self.df_outdata.loc[idx]['filepath'])
            array_to_file(self.results[idx],
                          self.df_outdata.loc[idx]['filepath'],
                          self.infiles.filepath.iloc[0], dtype=gdal.GDT_UInt16, compress=False, noData=True)

class LocPreProcessor(object):
    """
    Class to process final data to trend images
    """

    def __init__(self, study_site, row=None, path=None, pr_string=None, parallel=False, bufsize=4000, *args, **kwargs):
        """
        Class to clip pre-processed Landsat image to local subsets
        :param study_site: str
        :param row: int
        :param path: int
        :param pr_string: str
        :param parallel: bool
        :param args:
        :param kwargs:
        :return:
        """
        self.study_site = study_site
        self.row = row
        self.path = path
        self.pr_string = pr_string
        self.parallel = parallel
        self.bufsize = bufsize
        self.pr_exists = False
        self.continue_process = True
        self.n_files_in = 0
        self.n_files_out = 0
        self.infolders = []
        self.infiles = np.array([])
        self.outfiles = np.array([])
        self.empty_fld = np.array([])
        self.process_ok = np.array([], dtype=np.bool)
        self.output_string = np.array([], dtype=np.str)
        self._initial_check()
        try:
            self.startup_funcs()
            self.startup_funcs2()
            if self.pr_exists and self.continue_process:
                self.startup_funcs3()
        except:
            pass

    def _initial_check(self):
        """
        check for input values, must either be int for row and path or string for pr_string
        :return:
        """
        if (self.row is None) and (self.path is None) and (type(self.pr_string) == str):
            self.pr_string_to_pr()
        elif (self.row is None) and (self.path is None) and (type(self.pr_string) != str):
            raise ValueError("Please indicate either row and path or pr_string")
        # TODO: Check for GDAL_DATA environment variable
        if not os.path.exists(wrs2_path):
            raise IOError("Path to WRS-2 vector file is not correctly defined")

    def startup_funcs(self):
        """
        startup wrapper 1 - running each time
        :return:
        """
        self.get_study_site_features()
        self.check_pr_existing()

    def startup_funcs2(self):
        """
        startup wrapper - running if local row/path exists
        :return:
        """
        self.get_wrs2tiles()
        self.get_existing_wrs2()

    def startup_funcs3(self):
        """
        startup wrapper - running if local row/path exists
        :return:
        """
        self.get_infolders()
        self.get_infiles()
        self.make_outnames()
        self.check_outnames()
        self.check_outfolder_structure()
        self.make_output_string()

    def get_study_site_features(self):
        """
        fetch study site properties
        :return:
        """
        self.fishnet_file_ = study_sites[self.study_site]['fishnet_file']
        self.data_dir_ = study_sites[self.study_site]['data_dir']
        self.out_dir_ = study_sites[self.study_site]['processing_dir']
        self.ss_name_ = study_sites[self.study_site]['name']
        self.epsg_ = study_sites[self.study_site]['epsg']

    def pr_string_to_pr(self):
        """
        make row/path integers from string
        :return:
        """
        self.row, self.path = np.array(self.pr_string.split('_'), dtype=np.int)

    def check_pr_existing(self):
        """
        Check if indicated fishnet row/path combination exists
        :return:
        """
        ds2 = fiona.open(self.fishnet_file_)
        filtered = len([f for f in ds2 if (f['properties']['path'] == self.path) and (f['properties']['row'] == self.row)])
        if filtered == 1:
            self.pr_exists = True
        else:
            raise ValueError("This path/row combination does not exists")

    def get_wrs2tiles(self):
        """
        Function to get intersecting WRS-tiles
        :return:
        """
        ds1 = fiona.open(wrs2_path)
        ds2 = fiona.open(self.fishnet_file_)
        filtered = [f for f in ds2 if (f['properties']['path'] == self.path) and (f['properties']['row'] == self.row)]
        sr2 = osr.SpatialReference()
        sr2.ImportFromWkt(ds2.crs_wkt)
        sr1 = osr.SpatialReference()
        sr1.ImportFromWkt(ds1.crs_wkt)

        ds2.close()
        ds2 = filtered

        # check for upper and lower case
        try:
            self.coords = np.array([ds2[0]['properties'][feat] for feat in ['XMIN', 'YMIN', 'XMAX', 'YMAX']])
        except:
            self.coords = np.array([ds2[0]['properties'][feat] for feat in ['xmin', 'ymin', 'xmax', 'ymax']])
        bbox = self.coords + np.array([-self.bufsize, -self.bufsize, self.bufsize, self.bufsize])
        bnds = np.array(bbox).reshape((2, 2))

        tr = osr.CoordinateTransformation(sr2, sr1)
        bbx = np.array(tr.TransformPoints(bnds))

        ds1_f = ds1.filter(bbox=tuple(bbx.ravel()[[0, 1, 3, 4]]))
        pr = np.array([(d['properties']['PATH'], d['properties']['ROW']) for d in ds1_f])
        self.wrs2path, self.wrs2row = pr.T

    def get_existing_wrs2(self):
        """
        Find existing data
        :return:
        """
        self.wrs_folderlist = []
        for p, r in zip(self.wrs2path, self.wrs2row):
            fld = os.path.join(self.data_dir_, 'p{0:03d}_r{1:02d}'.format(p, r))
            if os.path.exists(fld):
                self.wrs_folderlist.append(fld)
        if len(self.wrs_folderlist) == 0:
            print("No Data available for selected region")
            self.continue_process = False

    def get_infolders(self):
        """
        Find all subfolders within the defined folder structure
        :return:
        """
        for f in self.wrs_folderlist:
            self.infolders.extend(get_foldernames(f, global_path=True))

    def get_infiles(self):
        """
        Find all preprocessed tif_files within defined folder structure
        :return:
        """
        for fld in self.infolders:
            basename = os.path.basename(fld).split('-')[0]
            infile = os.path.join(fld, 'tmp', basename + '_masked.tif')
            if os.path.exists(infile):
                self.infiles = np.append(self.infiles, infile)  # all valid files
            else:
                self.empty_fld = np.append(self.empty_fld, infile)  # non existant files (empty folder)
        self.n_files_in = len(self.infiles)

    def make_outnames(self):
        """
        Make list of output filenames
        :return:
        """
        self.out_dir_tile = os.path.join(self.out_dir_, '{0}_{1}_{2}'.format(self.ss_name_, self.row, self.path))
        for f in self.infiles:
            outname = os.path.basename(f).split('.tif')[0] + '_{0}_{1}_{2}.tif'.format(self.ss_name_, self.row, self.path)
            self.outfiles = np.append(self.outfiles, os.path.join(self.out_dir_tile, outname))

    def check_outnames(self):
        """
        Check which outfiles already exist and filter accordingly
        :return:
        """
        self.outfile_exists = np.array([os.path.exists(f) for f in self.outfiles])

    def check_outfolder_structure(self):
        """
        make output dir if not existant
        :return:
        """
        if not os.path.exists(self.out_dir_tile):
            os.makedirs(self.out_dir_tile)

    def make_output_string(self):
        """
        Create processing command for gdalwarp and system processing
        :return:
        """
        self.n_files_out = len(self.outfiles[~self.outfile_exists])
        xmin, ymin, xmax, ymax = self.coords
        coords = '{0} {1} {2} {3}'.format(xmin, ymin, xmax, ymax)
        outstr = []
        for infile, outfile in zip(self.infiles[~self.outfile_exists], self.outfiles[~self.outfile_exists]):
            outstr.append(r'gdalwarp -tr 30 30 -te {0} -srcnodata 0 -dstnodata 0 -t_srs EPSG:{3} -r cubic -co COMPRESS=LZW {1} {2}'.format(coords, infile, outfile, self.epsg_))
        self.output_string = np.array(outstr)

    def report_pre_processing(self):
        """
        printed report of data availability
        :return:
        """
        if not self.pr_exists:
            return
        elif not self.continue_process:
            return
        print("\n\nProcessing Dataset: {0} : {1}_{2}".format(self.study_site, self.row, self.path))
        print("Total Datasets Available: {0}".format(len(self.infolders)))
        print("Missing Images: {0}".format(len(self.empty_fld)))
        print("Already Existing: {0}".format(np.sum(self.outfile_exists)))
        if self.n_files_in > 0:
            print("Processing {0}/{1} Images!".format(np.sum(~self.outfile_exists), len(self.infolders)))

    def process(self):
        """
        run processing
        :return:
        """
        if not self.pr_exists:
            print("Tile {0}_{1} does not exist".format(self.row, self.path))
            return
        elif (self.n_files_in == 0) or (self.n_files_out == 0):
            self.process_ok = np.array([False])
            return

        if self.parallel:
            Parallel(n_jobs=10)(delayed(os.system)(o) for o in self.output_string)
        else:
            [os.system(o) for o in self.output_string]
        self.process_ok = np.array([os.path.exists(o) for o in self.outfiles[~self.outfile_exists]])

    def report_post_processing(self):
        """
        printed report of processing success
        :return:
        """
        # TODO: make reporting for no coverage
        if (self.n_files_in > 0) and self.pr_exists and (self.n_files_out > 0):
            print("Finished files: {0}/{1}".format(self.process_ok.sum(), np.sum(~self.outfile_exists)))
        elif self.n_files_out == 0 and self.pr_exists:
            print("All output files already exist!")
        else:
            print("No Data available, skip processing!")



# TODO: cleanup structure, e.g. make generic class and each processing as subclass
class LocPreProcessorDEM(LocPreProcessor):
    def __init__(self, study_site, master_dem=r'F:\18_Paper02_LakeAnalysis\02_AuxData\04_DEM\DEM.vrt', row=None,
                 path=None, pr_string=None, parallel=False, bufsize=4000, *args, **kwargs):
        """
        Class to clip pre-processed Landsat image to local subsets
        :param study_site: str
        :param row: int
        :param path: int
        :param pr_string: str
        :param parallel: bool
        :param args:
        :param kwargs:
        :return:
        """
        super().__init__(study_site, row, path, pr_string, parallel, bufsize, *args, **kwargs)
        self.study_site = study_site
        self.master_dem = master_dem
        self.row = row
        self.path = path
        self.pr_string = pr_string
        self.parallel = parallel
        self.bufsize = bufsize
        self.pr_exists = False
        self.infolders = []
        self.infiles = np.array([])
        self.outfiles = np.array([])
        self.empty_fld = np.array([])
        self.process_ok = np.array([], dtype=np.bool)
        self.output_string = np.array([], dtype=np.str)
        self._initial_check()
        try:
            self.startup_funcs()
            self._get_coords()
            self._setup_outpaths()
        except:
            pass

    def _check_master_dem(self):
        """
        Function to check if DEM master file exists
        :return:
        """
        if not os.path.exists(self.master_dem):
            raise ValueError("DEM Master File does not exist!")

    def _get_coords(self):
        """
        Function to get intersecting WRS-tiles
        :return:
        """
        ds2 = fiona.open(self.fishnet_file_)
        filtered = [f for f in ds2 if (f['properties']['path'] == self.path) and (f['properties']['row'] == self.row)]
        sr2 = osr.SpatialReference()
        sr2.ImportFromWkt(ds2.crs_wkt)
        ds2.close()
        ds2 = filtered

        self.coords = np.array([ds2[0]['properties'][feat] for feat in ['XMIN', 'YMIN', 'XMAX', 'YMAX']])

    def _setup_outpaths(self):
        """
        setup file paths for output files
        :return:
        """
        path = study_sites[self.study_site]['dem_dir']
        self.dem_path_ = os.path.join(path, '{ss}_{pr}_dem.tif'.format(ss=self.study_site, pr=self.pr_string))
        self.slope_path_ = os.path.join(path, '{ss}_{pr}_slope.tif'.format(ss=self.study_site, pr=self.pr_string))
        if not os.path.exists(path):
            os.makedirs(path)

    def make_subset(self):
        """
        run process using gdalwarp
        :return:
        """
        s_dem = r'gdalwarp -t_srs EPSG:{epsg} -tr 30 30 -r cubic -te {xmin} {ymin} ' \
                r'{xmax} {ymax} {infile} {outfile}'.format(epsg=self.epsg_,
                                                           xmin=self.coords[0],
                                                           ymin=self.coords[1],
                                                           xmax=self.coords[2],
                                                           ymax=self.coords[3],
                                                           infile=self.master_dem,
                                                           outfile=self.dem_path_)

        s_slope = r'gdaldem slope -compute_edges -alg ZevenbergenThorne {infile} {slopefile}'.format(infile=self.dem_path_, slopefile=self.slope_path_)

        if not os.path.exists(self.dem_path_):
            os.system(s_dem)

        if not os.path.exists(self.slope_path_):
            os.system(s_slope)


def auto_prlist(prlist, study_site):
    if prlist[0] in ['full', 'auto']:
        fld = get_foldernames(study_sites[study_site]['processing_dir'])
        fld = [f.split(study_sites[study_site]['name'])[-1][1:] for f in fld]
    else:
        fld = prlist
    return fld
