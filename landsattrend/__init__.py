# -*- coding: utf-8 -*-

"""
Description
"""


from landsattrend.processing_funcs import LocPreProcessor, LocPreProcessorDEM, \
    Processor, DataStack, auto_prlist
from landsattrend.mosaic import Mosaic, MosaicFiltered, MosaicNewOnly
from landsattrend.config_study_sites import study_sites
from landsattrend.file_preprocessor import FilePreProcessor
from .version import __version__