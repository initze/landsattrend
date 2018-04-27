from unittest import TestCase
from landsattrend import DataStack


class DataStackTest(TestCase):

    def setUp(self):
        raster_dir = 'landsattrend/tests/data/raster'
        self.data19992014 = DataStack(raster_dir, indices=['ndvi', 'tcb'], tc_sensor='auto', startyear=1999, endyear=2014,
                              startmonth=7, endmonth=8)
        self.datafull = DataStack(raster_dir, indices=['ndvi', 'tcb'], tc_sensor='auto', startyear=1900, endyear=2100,
                              startmonth=1, endmonth=12)

    def test_filelist_length(self):
        assert len(self.datafull.df_indata) == 164
        assert len(self.data19992014.df_indata) == 142

    def test_filelist_sensorcount(self):
        assert len(self.datafull.df_indata.query('sensor == "ETM"')) == 114
        assert len(self.datafull.df_indata.query('sensor == "TM"')) == 20
        assert len(self.datafull.df_indata.query('sensor == "OLI"')) == 30
        assert len(self.data19992014.df_indata.query('sensor == "ETM"')) == 108
        assert len(self.data19992014.df_indata.query('sensor == "TM"')) == 20
        assert len(self.data19992014.df_indata.query('sensor == "OLI"')) == 14

    def test_filelist_dates(self):
        assert len(self.datafull.df_indata.query('year == 1989')) == 0
        assert len(self.datafull.df_indata.query('year == 2001')) == 5
        assert len(self.datafull.df_indata.query('month == 7')) == 95
        assert len(self.datafull.df_indata.query('day == 23')) == 5
        assert len(self.data19992014.df_indata.query('year == 2016')) == 0
        assert len(self.data19992014.df_indata.query('year == 2012')) == 11
        assert len(self.data19992014.df_indata.query('month == 8')) == 58
        assert len(self.data19992014.df_indata.query('day == 7')) == 4

    def tearDown(self):
        pass