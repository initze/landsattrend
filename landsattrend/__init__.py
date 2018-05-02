# -*- coding: utf-8 -*-

"""
Description
"""


from landsattrend import utils
from landsattrend.config_study_sites import study_sites
from landsattrend.file_preprocessor import FilePreProcessor
from landsattrend.mosaic import Mosaic, MosaicFiltered, MosaicNewOnly
from landsattrend.processing_funcs import LocPreProcessor, LocPreProcessorDEM, \
    Processor, DataStack, auto_prlist
from landsattrend.version import __version__
