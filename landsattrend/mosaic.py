import glob
import os
import pandas as pd
import datetime
import rasterio
from .helper_funcs import array_to_file
from osgeo import gdal
import numpy as np


class Mosaic(object):
    """Class to create a mosaic of tiled trend images"""

    def __init__(self, infolder, indices=None,
                 tile_dir=r'tiles',
                 mosaic_dir=r'.',
                 visual_file=r'011_tcrgb_mos.tif'):
        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndmi', 'ndwi', 'nobs']
        self.indices = indices
        self.mosaic_dir = os.path.normpath(os.path.join(infolder, mosaic_dir))
        self.tile_dir = os.path.normpath(os.path.join(infolder, tile_dir))
        self.visual_file = os.path.join(self.mosaic_dir, visual_file)
        self.process = True
        self.df = pd.DataFrame()
        self.df_tmp = None
        self._set_fnames()
        self._get_visual_timestamp()
        self._make_export_filenames()

    @staticmethod
    def get_basename(inpath):
        return '_'.join(os.path.basename(inpath).split('_')[:-1])

    @staticmethod
    def get_timestamp(file_path):
        if os.path.exists(file_path):
            return datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        else:
            return datetime.datetime(1900, 1, 1)

    def _set_fnames(self):
        self.FNAMES = {'tcb': '001_tcb_mos',
                       'tcg': '002_tcg_mos',
                       'tcw': '003_tcw_mos',
                       'ndvi': '004_ndvi_mos',
                       'ndwi': '005_ndwi_mos',
                       'ndmi': '006_ndmi_mos',
                       'nobs': '007_nobs_mos',
                       'tcrgb': '011_tcrgb_mos',
                       'ndrgb': '012_ndrgb_mos',
                       'mask': '021_mask'}

    # TODO: check real functionality
    def _get_visual_timestamp(self):
        self._visual_timestamp = self.get_timestamp(self.visual_file)

    def _make_export_filenames(self):
        """
        function to setup file names for the file export
        """
        self.list_txt_path = {}
        self.list_vrt_path = {}
        for i in self.indices:
            self.list_txt_path[i] = self.mosaic_dir + r'\{i}.txt'.format(i=self.FNAMES[i])
            self.list_vrt_path[i] = self.mosaic_dir + r'\{i}.vrt'.format(i=self.FNAMES[i])
        self.rgb_vrt_file = self.mosaic_dir + r'\{name}.vrt'.format(name=self.FNAMES['tcrgb'])
        self.rgb_tif_file = self.mosaic_dir + r'\{name}.tif'.format(name=self.FNAMES['tcrgb'])
        self.rgb_tif_file_tmp = self.mosaic_dir + r'\{name}2.tif'.format(name=self.FNAMES['tcrgb'])
        self.rgb_png_file = self.mosaic_dir + r'\{name}.png'.format(name=self.FNAMES['tcrgb'])
        self.mask_file = self.mosaic_dir + r'\{name}.tif'.format(name=self.FNAMES['mask'])

    def make_filelist(self):
        """Function to create file list"""
        for i in self.indices:
            flist = glob.glob(r'{indir}\*{i}*.tif'.format(indir=self.tile_dir, i=i))
            basenames = [self.get_basename(f) for f in flist]
            timestamps = [self.get_timestamp(f) for f in flist]
            self.df_tmp = pd.DataFrame(index=basenames, columns=[i, i + '_tstamp'])
            self.df_tmp[i] = flist
            self.df_tmp[i + '_tstamp'] = timestamps
            self.df = self.df.join(self.df_tmp, how='outer')

    def export_to_file_lists(self, filt=True):
        """filter and export filelists"""
        self._make_export_filenames()
        for i in self.indices:
            self.df.to_csv(self.mosaic_dir + r'\{i}.txt'.format(i=self.FNAMES[i]),
                           header=False, index=False, columns=[i])

    def make_vrt(self):
        if self.process:
            # Needs to get fixed
            for i in self.indices:
                os.system(r'gdalbuildvrt -srcnodata 0 -vrtnodata 0 -overwrite {p} -input_file_list {ft}'.format(
                    p=self.list_vrt_path[i], ft=self.list_txt_path[i]))
            outfiles = ' '.join([self.list_vrt_path[i] for i in ['tcb', 'tcg', 'tcw']])
            os.system(r'gdalbuildvrt -separate -srcnodata 0 -vrtnodata 0 {r} {out}'.format(r=self.rgb_vrt_file,
                                                                                           out=outfiles))

    def make_rgb_image(self, scaling_low=-0.12, scaling_high=0.12, format='GTiff'):
        if self.process:
            execstr = 'gdal_translate -a_nodata none -scale {s_l} {s_h} 1 255 -ot Byte -of {fmt} {r} {p}'.format(
                s_l=scaling_low,
                s_h=scaling_high,
                p=self.rgb_tif_file_tmp,
                r=self.rgb_vrt_file,
                fmt=format)
            os.system(execstr)

    def extract_mask(self):
        print("Extracting mask")
        if os.path.exists(self.rgb_vrt_file):
            vrt_master = self.rgb_vrt_file
        else:
            vrt_master = self.list_vrt_path['ndvi']
        src = rasterio.open(vrt_master)
        msk = src.read_masks(1)
        array_to_file(msk, self.mask_file, vrt_master, dtype=gdal.GDT_Byte, compress=True)

    def apply_mask(self):
        """

        :return:
        """
        print("Applying mask")
        data = rasterio.open(self.rgb_tif_file_tmp).read()
        lo_msk = np.where(data == 0)
        data[lo_msk] = 1

        src_msk = np.squeeze(rasterio.open(self.mask_file).read())
        m = np.array(src_msk > 0, dtype=np.float)
        data = (data * m)

        array_to_file(data, self.rgb_tif_file, self.rgb_tif_file_tmp, dtype=gdal.GDT_Byte, compress=True,
                      noData=True, noDataVal=0)

    def cleanup(self):
        # TODO: Check if relations are OK
        filelist = list(self.list_txt_path.values()) + [self.rgb_tif_file_tmp]
        for f in filelist:
            if os.path.exists(f):
                os.remove(f)


class MosaicNewOnly(Mosaic):

    def _set_fnames(self):
        self.FNAMES = {'tcb': '101_tcb_mos',
                       'tcg': '102_tcg_mos',
                       'tcw': '103_tcw_mos',
                       'ndvi': '104_ndvi_mos',
                       'ndwi': '105_ndwi_mos',
                       'ndmi': '106_ndmi_mos',
                       'nobs': '107_nobs_mos',
                       'tcrgb': '111_tcrgb_mos',
                       'ndrgb': '112_ndrgb_mos',
                       'mask': '121_mask'}

    def export_to_file_lists(self, filt=True):
        """filter and export filelists"""
        self._make_export_filenames()
        for i in self.indices:
            df_filt = self.df[self.df['{i}_tstamp'.format(i=i)] > self._visual_timestamp]
            if len(df_filt) == 0:
                self.process = False
            else:
                df_filt.to_csv(self.mosaic_dir + r'\{i}.txt'.format(i=self.FNAMES[i]), header=False, index=False,
                               columns=[i])


class MosaicFiltered(Mosaic):
    def __init__(self, infolder, indices=None,
                 tile_dir=r'tiles',
                 mosaic_dir=r'.',
                 visual_file=r'011_tcrgb_mos.tif',
                 tiles=None,
                 outname='T2_Z052_22-29'):
        if indices is None:
            indices = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndmi', 'ndwi', 'nobs']
        if tiles is None:
            tiles = ['24_6']
        self.indices = indices
        self.mosaic_dir = os.path.normpath(os.path.join(infolder, mosaic_dir))
        self.tile_dir = os.path.normpath(os.path.join(infolder, tile_dir))
        self.visual_file = os.path.join(self.mosaic_dir, visual_file)
        self.process = True
        self.outname = outname
        self.df = pd.DataFrame()
        self._set_fnames()
        self._get_visual_timestamp()
        self.tiles = tiles

    def make_filelist(self):
        """Function to create file list"""
        for i in self.indices:
            flist = []
            for t in self.tiles:
                f = glob.glob(os.path.join(self.tile_dir, '*{t}_{i}*.tif'.format(t=t, i=i)))
                if len(f) > 0:
                    flist.append(f[0])
            # flist = [glob.glob(os.path.join(self.tile_dir, '*{t}_{i}*.tif'.format(t=t, i=i)))[0] for t in self.tiles]
            basenames = [self.get_basename(f) for f in flist]
            self.df_tmp = pd.DataFrame(index=basenames, columns=[i])
            self.df_tmp[i] = flist
            self.df = self.df.join(self.df_tmp, how='outer')

    def export_to_raster(self):
        for i in self.indices:
            execstring = 'gdal_translate -of GTiff {vrt} {of}'.format(vrt=self.list_vrt_path[i],
                                                                      of=self.list_tif_path[i])
            os.system(execstring)

    def export_to_file_lists(self, filt=True):
        """filter and export filelists"""
        self._make_export_filenames()
        for i in self.indices:
            self.df.to_csv(self.list_txt_path[i], header=False, index=False, columns=[i])

    def _make_export_filenames(self):
        """
        function to setup file names for the file export
        """
        self.list_txt_path = {}
        self.list_vrt_path = {}
        self.list_tif_path = {}
        for i in self.indices:
            self.list_txt_path[i] = os.path.join(self.mosaic_dir, '{on}_{i}.txt'.format(on=self.outname, i=i.upper()))
            self.list_vrt_path[i] = os.path.join(self.mosaic_dir, '{on}_{i}.vrt'.format(on=self.outname, i=i.upper()))
            self.list_tif_path[i] = os.path.join(self.mosaic_dir, '{on}_{i}.tif'.format(on=self.outname, i=i.upper()))
        self.rgb_vrt_file = self.mosaic_dir + r'\{name}.vrt'.format(name=self.FNAMES['tcrgb'])
        self.rgb_tif_file = os.path.join(self.mosaic_dir, '{on}_RGB.tif'.format(on=self.outname))
        self.rgb_tif_file_tmp = self.mosaic_dir + r'\{name}2.tif'.format(name=self.FNAMES['tcrgb'])
        self.rgb_png_file = self.mosaic_dir + r'\{name}.png'.format(name=self.FNAMES['tcrgb'])
        self.mask_file = self.mosaic_dir + r'\{name}.tif'.format(name='mask')
