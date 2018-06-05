from unittest import TestCase
from landsattrend.breakpoint import Breakpoint
from landsattrend.data_stack import DataStack, load_point_ts
import numpy as np

class DataStackTest(TestCase):

    def setUp(self):
        raster_dir = 'landsattrend/tests/data/raster'
        self.singlepoint = load_point_ts('', (470307.882181, 7899742.63389), startmonth=1, endmonth=12, startyear=1980, endyear=2017, infolder=raster_dir)
        self.singlepoint.load_data()
        self.singlepoint.group_data(type='median')
        index = 'tcb'
        df_ndvi_max = np.squeeze(self.singlepoint.index_data_grouped[index])
        msk = df_ndvi_max.mask
        self.x, self.y = self.singlepoint.grouped_feature[~msk], df_ndvi_max[~msk]
        self.breakpoint = Breakpoint(self.x, self.y, n_breaks=1, predictor='r2')

    def test_break_loc(self):
        self.breakpoint.fit()
        assert self.breakpoint.results_best_['break_year1'] == 2012

    def tearDown(self):
        pass