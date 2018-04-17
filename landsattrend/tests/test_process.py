"""Tests for landsattrend processing Objects"""

from unittest import TestCase
from landsattrend import Processor


class ProcessorTest(TestCase):

    def setUp(self):
        self.proc = Processor('testcase', 'data/raster/result', infolder='data/raster', indices=['ndvi'],
                              startyear=1999, endyear=2014,
                              startmonth=7, endmonth=8, pr_string='32_7',
                              n_jobs=10, tc_sensor='auto', ts_mode='full')
    """
    def test_filelist_length(self):
        assert len(self.proc.infiles) == 142

    def test_filelist_sensorcount(self):
        assert len(self.proc.infiles.query('sensor == "ETM"')) == 108
        assert len(self.proc.infiles.query('sensor == "TM"')) == 20
        assert len(self.proc.infiles.query('sensor == "OLI"')) == 14
    """
"""
    def test_indices_reader(self):
        assert True == False
        pass

    def test_filename_version(self):
        assert True == False
        pass

    def test_tasseled_cap_indices(self):
        assert True == False
        pass

    def test_file_writer(self):
        assert True == False
        pass
"""
