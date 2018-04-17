from unittest import TestCase
from landsattrend import DataStack


class DataStackTest(TestCase):

    def setUp(self):
        self.data19992014 = DataStack('data/raster', indices=['ndvi', 'tcb'], tc_sensor='auto', startyear=1999, endyear=2014,
                              startmonth=7, endmonth=8)
        self.datafull = DataStack('data/raster', indices=['ndvi', 'tcb'], tc_sensor='auto', startyear=1900, endyear=2100,
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

    def tearDown(self):
        pass