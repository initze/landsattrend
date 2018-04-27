import datetime
import glob
import os
import shutil
import time

import numpy as np

from .config_study_sites import study_sites
from landsattrend.utils import Masking, MaskingNG, WRS_Mover


class FilePreProcessor(object):
    """
    Class to run pre processing operations for final processing:
    clip and file operations
    """
    def __init__(self, in_dir, processing_dir,
                 dst_dir = None,
                 mode='index',
                 compr_file_ext='tar.gz',
                 study_site='none',
                 archiving_dir='auto',
                 dump_dir='none',
                 delete_compressed=False,
                 processing_order='auto',
                 processing_type='sr'):
        self.mode = mode
        self.in_dir = in_dir
        self.processing_dir = processing_dir
        self.dst_dir = dst_dir
        self.compr_file_ext = compr_file_ext
        self.study_site = study_site
        self.archiving_dir = archiving_dir
        self.dump_dir = dump_dir
        self.delete_compressed = delete_compressed
        self.processing_order = processing_order
        self.processing_type = processing_type
        self.find_compressed_files()
        self.make_basenames()
        self.make_processing_dir_list()
        self.make_masked_file_names()
        self.make_archiving_paths()
        self.set_dst_dir()
        self.make_succ_masks()
        self.file_checks()
        self.rearrange_order()

    def find_compressed_files(self):
        """
        List all files with specified file-extension within data directory
        :return: list
        """
        pattern = os.path.join(self.in_dir, '*.' + self.compr_file_ext)
        self.flist_compressed = glob.glob(pattern)

    def make_basenames(self):
        """
        list basenames of all input-files, without extension
        :return: list
        """
        bnlist = [os.path.basename(f) for f in self.flist_compressed]
        self.list_basenames = [f.split('.'+self.compr_file_ext)[-2] for f in bnlist]
        self.list_basenames_short = [f.split('-')[0] for f in self.list_basenames]

    def make_processing_dir_list(self):
        """
        list full path of dataset specific processing dirs
        :return: list
        """
        self.processing_dir_list = np.array([os.path.join(self.in_dir, f) for f in self.list_basenames])

    def make_masked_file_names(self):
        """
        list full filenames of masked output images
        :return: list
        """
        self.masked_file_names = np.array([os.path.join(pd, 'tmp', bn+'_masked.tif')
                                           for pd, bn in zip(self.processing_dir_list, self.list_basenames_short)])

    def set_dst_dir(self):
        """
        set destination dir, change if destination dir is not indicated, but study site
        :return:
        """
        if (not self.dst_dir) :
            self.dst_dir = study_sites[self.study_site]['data_dir']

    def make_archiving_paths(self):
        """
        list filepath for archiving
        :return:
        """
        if self.archiving_dir != 'auto':
            if not os.path.exists(self.archiving_dir):
                os.makedirs(self.archiving_dir)
        self.archive_file_list = np.array([os.path.join(self.archiving_dir, os.path.basename(bn)) for bn in self.flist_compressed])

    def make_succ_masks(self):
        """
        create masks for indication of successful processing
        :return:
        """
        self.n_datasets = len(self.list_basenames)
        self.succ_uncompressed = np.zeros((self.n_datasets), dtype=np.bool)
        self.succ_archived = np.zeros((self.n_datasets), dtype=np.bool)
        self.succ_masked = np.zeros((self.n_datasets), dtype=np.bool)
        self.succ_moved = np.zeros((self.n_datasets), dtype=np.bool)

    def run_uncompress(self, index=0):
        """
        run extraction of archived datasets
        :param index: int
        :return:
        """
        # TODO: Check if already exists
        if self.mode == 'index':
            execstr1 = '7z e {0} -o{1} -aoa'.format(self.flist_compressed[index], self.in_dir)
            tarfile = os.path.join(self.in_dir, self.list_basenames[index])
            execstr2 = '7z e {0}.tar -o{0} -aoa'.format(tarfile)
            os.system(execstr1)
            os.system(execstr2)
            os.remove(tarfile+'.tar')
            # check for content of file
            # check if it corresponds to output
        self.succ_uncompressed[index] = True

    def run_archive_files(self, index=0):
        """
        trigger archiving (move) of archived datasets
        :param index: int
        :return:
        """
        if not self.archive_file_exists[index]:
            shutil.move(self.flist_compressed[index], self.archive_file_list[index])
            self.succ_archived[index] = True
        else:
            print("file exists")

    def run_mask(self, index=0):
        """
        trigger mask application of uncompressed datasets and cleanup directory
        :param index: int
        :return:
        """
        m = Masking(self.processing_dir_list[index],
                    self.masked_file_names[index],
                    valid_val=[0, 1], dst_nodata=0,
                    processing_type=self.processing_type)
        if all([m.mask_ok, m.files_ok, m.meta_ok]):
            m.make_mask()
            m.save_raster()
            m.compress_raster()
            m.cleanup_dir()

        else:
            m = MaskingNG(self.processing_dir_list[index],
            self.masked_file_names[index],
            valid_val=[0, 1], dst_nodata=0,
            processing_type=self.processing_type)
            if all([m.mask_ok, m.files_ok, m.meta_ok]):
                m.make_mask()
                m.save_raster()
                m.compress_raster()
                m.cleanup_dir()

        if all([m.masked, m.exported, m.compressed]):
            self.succ_masked[index] = True

    def run_move_files(self, index=0):
        """
        trigger move of processed data to specified directory with WRS-2 path/row substructure
        :param index: int
        :return:
        """
        move = WRS_Mover(self.processing_dir_list[index], dst_dir=self.dst_dir)
        move.move()
        self.succ_moved[index] = move.moved

    def file_checks(self):
        """
        Make boolean masks if archive files and output data exist
        :return:
        """
        # move archive_file_exists
        self.output_file_exists = np.empty_like(self.flist_compressed, dtype=np.bool)
        self.archive_file_exists = np.empty_like(self.flist_compressed, dtype=np.bool)
        for i in range(len(self.flist_compressed)): # make ndatasets variable
            mover = WRS_Mover(self.processing_dir_list[i], dst_dir=self.dst_dir)
            self.output_file_exists[i] = os.path.exists(mover.outfile)
            self.archive_file_exists[i] = os.path.exists(self.archive_file_list[i])
        pass

    def rearrange_order(self):
        """
        Rearrange listorder according to preferred ordering
        'auto' : standard processing - no reordering
        'archive' : process existing data first
        'process' : process new data first
        :return:
        """
        if self.processing_order not in ['auto', 'archive', 'process']:
            self.processing_order = 'auto'
            return
        elif self.processing_order == 'archive':
            self.index = np.argsort(~self.output_file_exists)
        elif self.processing_order == 'process':
            self.index = np.argsort(self.output_file_exists)
        else:
            self.index = np.arange(0, self.n_datasets, dtype=np.uint)

    # TODO: needs improved check for existing files already archived files nor taken into consideration
    def run_full_process(self):
        """
        Run entire process over each file
        :return:
        """
        counter = 1
        for i in self.index:
            mover = WRS_Mover(self.processing_dir_list[i], dst_dir=self.dst_dir) # move up to other func
            print("\nProcessing Dataset:{1}/{2} {0}".format(self.flist_compressed[i], counter, self.n_datasets))
            if not os.path.exists(mover.outfile):
                self.run_uncompress(index=i)
                self.run_archive_files(index=i)
                self.run_mask(index=i)
                self.run_move_files(index=i)
            else:
                self.run_archive_files(index=i)
            counter += 1

    # TODO: Make report file with non-processed files
    # TODO: Make list of successful processing
    def make_report(self):
        """
        Make processing report
        :return:
        """
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
        self.report_file = os.path.join(self.in_dir, 'report_{0}.txt'.format(tstamp))
        text = 'Processing Report\n\n'
        text += "Errors:\n"
        for m, txt in zip([self.succ_uncompressed, self.succ_archived, self.succ_masked, self.succ_moved],
                     ["Failed File extraction:\n", "Failed Archiving:\n", "Failed Masking:\n", "Failed File Moving:\n"]):
            text += txt
            for f in np.array(self.list_basenames)[~m]:
                text += f + '\n'

        with open(self.report_file, 'w') as src:
            src.write(text)