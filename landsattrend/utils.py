import csv
import datetime
import glob
import os
import shutil
import subprocess
from xml.dom import minidom

import bottleneck as bn
import fiona
import geopandas
import geopandas as gpd
import joblib
import netCDF4
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from fiona.crs import from_epsg
from osgeo import gdal, gdal_array as ga, ogr, osr
from scipy import stats
from scipy.stats import itemfreq
from shapely.geometry import Polygon, mapping


from landsattrend.config_study_sites import study_sites


def median_slopes(x1, x2, y1, y2):
    a = (y2 - y1) / (x2 - x1)
    return np.median(a, axis=0)

def slope(x1, x2, y1, y2):
    return (y2 - y1) / (x2 - x1)

def crossindex(data, nodata=0):
    """
    function to create crossindices for valid data of 1-D array
    """
    ind = np.where(data != nodata)[0]
    l = len(ind)
    r = ind[:-1]
    c = ind[1:]
    r = np.repeat(r, l - 1).reshape((l - 1, l - 1))
    c = np.tile(c, l - 1).reshape((l - 1, l - 1))
    c = c[np.triu_indices_from(c)].ravel()
    r = r[np.triu_indices_from(r)].ravel()
    return r, c

def non_duplicates(d, dates):
    """
    :param d: n-d masked array (eg. 4 dimensional tasseled cap stack
    :param dates: imagedates with potentially duplicate values
    :return: reduced image data, reduced imagedates
    """
    # find frequency of used imagedates - find where images were used at least twice
    ifreq = itemfreq(dates)
    ifreq = ifreq[ifreq[:, 1] > 1]

    # calculate masked mean of duplicate image dates and apply to data
    # collect reduced, non-used indices and remove from data-stack
    idx_del = []
    for ifr in ifreq:
        idx = np.where(dates == ifr[0])[0]  # find indices where imdates have at least duplicate values
        d[idx[0]] = d[idx].mean(axis=0)
        idx_del += list(idx[1:])
    valid_idx = np.array(np.setxor1d(list(range(len(dates))), idx_del), dtype=np.uint16)  # remove duplicate layers

    return d[valid_idx], dates[valid_idx], valid_idx

def get_n_observations(data, imdates, decision='<'):
    """
    returns number of observations from 3-d boolean mask
    :param data:
    :param imdates:
    :param decision:
    :return:
    """
    if len(imdates) != len(data):
        print("incompatible sizes")
        pass
    val, idx, counts = np.unique(imdates, return_index=True, return_counts=True)
    for v, i, c in zip(val, idx, counts):
        if c > 1:
            data[i] = data[[i, i + 1]].all(axis=0)
            # gt = np.where(mask[i] > mask[i+1])
            # mask[i][gt] = mask[i+1][gt]
    return np.invert(data[idx]).sum(axis=0)

def theil_sen(x, y, full=False):
    """
    x: imagedates in ordinal dates
    y: values to calculate slope
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(y) == 0:
        return 0

    r, c = crossindex(y)
    slp = slope(x[r], x[c], y[r], y[c])
    if full:
        return slp
    else:
        return np.median(slp)

def theil_sen2(x, y, full=False):
    """
    x: imagedates in ordinal dates
    y: values to calculate slope
    """

    if y.mask.mean() == 1:
        return 0

    slp, ict, ci_l, ci_u = stats.mstats.theilslopes(y, x)
    return slp, ci_l, ci_u

def trend_image(image_stack, image_dates, tiling=True, tile_size=(1, 1), mode='ts', factor=3650):
    """
    calculates trend of image values
    input:
    image_stack: 3-D array, array/list of image dates (ordinal)
    image_dates

    output:
    slope_image: 2-D array of slope
    """
    ndim = image_stack.ndim
    if ndim == 3:
        nds, nrows, ncols = image_stack.shape
        image_stack = image_stack.reshape((nds, -1))
    image_stack = image_stack.T
    # iterate over each image tile
    ind = np.where((~image_stack.mask).sum(axis=1) > 3)[0]
    if len(ind) != 0:
        slope_image = np.zeros((4, nrows * ncols))
        slope_image[:, ind] = np.array([theil_sen(image_dates, d) for d in image_stack[:, ind]])

    if ndim == 3:
        slope_image = np.reshape(slope_image, (nrows, ncols))
    # return images (slope per decade)
    return slope_image * factor

def trend_image2(image_stack, image_dates, processing_mask=None, factor=3650, obs_min=4):
    """
    calculates trend of image values
    input:
    image_stack: 3-D array, array/list of image dates (ordinal)
    image_dates: 1-D array
    processing_mask: Boolean mask to check which location should be processed

    output:
    slope_image: 3-D array of slope, intercept and confidence intervals of slope
    """

    # re-distribute data for further analysis
    ndim = image_stack.ndim
    if ndim == 3:
        nds, nrows, ncols = image_stack.shape
        image_stack = image_stack.reshape((nds, -1))
    image_stack = image_stack.T

    # define output-image
    slope_image = np.zeros((4, nrows * ncols))

    # Check which parts do contain data
    if isinstance(processing_mask, np.ndarray):
        ind = np.where(processing_mask.ravel())[0]
    else:
        ind = np.where((~image_stack.mask).sum(axis=1) >= obs_min)[0]
    if len(ind) != 0:
        # iterate over each image tile
        # TODO: throws Error in single mode in some locations (median year)
        # Use Index for better logging
        local_results = []
        for d in image_stack[ind]:
            try:
                local_results.append(theilslopes_ma_local(d, image_dates))
            except Exception as e:
                local_results.append((0, 0, 0, 0))
        slope_image[:, ind] = np.array(local_results).T

        # slope_image[:, ind] = np.array([theilslopes_ma_local(d, image_dates) for d in image_stack[ind]]).T
        slope_image[[0, 2, 3]] *= factor

    if ndim == 3:
        slope_image = np.reshape(slope_image, (4, nrows, ncols))

    return slope_image

def sig2_calc(n, z):
    return np.sqrt(1 / 18. * n * (n - 1) * (2 * n + 5)) * z

def theilslopes_local(y, x=None):
    """
    """
    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]

    medslope = np.median(slopes)
    medinter = np.median(y) - medslope * np.median(x)
    # Now compute confidence intervals

    z = -1.9599639845400545

    nt = len(slopes)  # N in Sen (1968)
    ny = len(y)  # n in Sen (1968)
    # Equation 2.6 in Sen (1968):

    sigmaz = sig2_calc(ny, z)

    Ru = min(int(np.round((nt - sigmaz) / 2.)), nt - 1)
    Rl = max(int(np.round((nt + sigmaz) / 2.)) - 1, 0)

    # bn.partition takes only necessary indices, much faster than a full sort
    delta = [bn.partition(slopes, Rl + 1)[Rl], bn.partition(slopes, Ru + 1)[Ru]]

    return medslope, medinter, delta[0], delta[1]

def theilslopes_ma_local(y, x=None, alpha=0.95):
    y = np.ma.asarray(y).flatten()
    if x is None:
        x = np.ma.arange(len(y), dtype=float)
    else:
        x = np.ma.asarray(x).flatten()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths ! (%s<>%s)" % (len(y), len(x)))

    m = np.ma.mask_or(np.ma.getmask(x), np.ma.getmask(y))
    y._mask = x._mask = m
    # Disregard any masked elements of x or y
    y = y.compressed()
    x = x.compressed().astype(float)
    # We now have unmasked arrays so can use `stats.theilslopes`
    return theilslopes_local(y, x)


def get_datafolder(study_site, coords, epsg='auto'):
    # TODO: find better method --> spatial select?
    if epsg == 'auto':
        epsg = study_sites[study_site]['epsg']
    gdf = gpd.GeoDataFrame()
    gdf.crs = {'init': 'epsg:{0}'.format(epsg)}
    gdf['geometry'] = [shapely.geometry.Point(coords[0], coords[1])]
    fn = gpd.read_file(study_sites[study_site]['fishnet_file'])
    joined = gpd.sjoin(gdf.to_crs(epsg=study_sites[study_site]['epsg']), fn)
    if len(joined) == 1:
        return study_sites[study_site]['processing_dir'] + r'\{0}_{1}_{2}'.format(study_site, joined.iloc[0].row, joined.iloc[0].path)
    else:
        raise ValueError("No Spatial Intersection of Point")


def sortlist(inlist):
    inlist = np.array(inlist)
    tmplist = [fn[9:16] for fn in inlist]
    d_ord = [datetime.datetime.strptime(f, "%Y%j").toordinal() for f in tmplist]
    return inlist[np.argsort(d_ord)]


def sensorlist(flist):
    """
    Function to get Landsat sensor from file name
    Checks version of ESPA Landsat filenaming version
    :param flist: list
    :return: list
    """
    sl_v1 = {'LT4': 'TM', 'LT5': 'TM', 'LE7': 'ETM', 'LC8': 'OLI'}
    sl_v2 = {'LT04': 'TM', 'LT05': 'TM', 'LE07': 'ETM', 'LC08': 'OLI'}
    #TODO ADD distinction between naming types
    sensor_list = []
    for f in flist:
        if len(f.split('_masked')[0]) == 16:
            sensor_list.append(sl_v1[f[:3]])
        elif len(f.split('_masked')[0]) == 22:
            sensor_list.append(sl_v2[f[:4]])
        else:
            pass
    return sensor_list


def get_foldernames(path, global_path=False):
    """
    :param path:
    :param global_path:
    :return:
    """
    dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if global_path:
        return [os.path.join(path, directory) for directory in dirs]
    else:
        return dirs


def array_to_file(inarray, outfile, samplefile,
                  metadata=None,
                  dtype=gdal.GDT_Float32,
                  compress=True,
                  quiet=True,
                  outresolution=None,
                  noData=False,
                  noDataVal=0):
    """
    This function exports a numpy.array to a physical Raster File.
    Input:
        array: numpy array to export
        outfile: filepath of output dataset
        samplefile: file with same properties (datatype, size) as outputfile. Projection information will be read from
                    this file.
        dtype: output dtype ('float', 'int', 'uint')
    """
    if not quiet:
        print("Exporting array to file...")
    # read sample dataset parameters
    sdataset = gdal.Open(samplefile, gdal.GA_ReadOnly)
    proj = sdataset.GetProjection()
    gt = sdataset.GetGeoTransform()
    # set manual outputresolution
    if outresolution:
        gt = list(gt)
        gt[1] = outresolution[0]
        gt[5] = -outresolution[1]
        gt = tuple(gt)

    #############################################
    ncols = inarray.shape[-1]
    nrows = inarray.shape[-2]
    if not quiet:
        print(ncols)
        print(nrows)
    #############################################
    driver = gdal.GetDriverByName("GTiff")
    if inarray.ndim == 3:
        if compress:
            outds = driver.Create(outfile, ncols, nrows, inarray.shape[0], dtype, ['COMPRESS=LZW'])
        else:
            outds = driver.Create(outfile, ncols, nrows, inarray.shape[0], dtype)
        if not quiet:
            print('3-dimensional file created')
    elif inarray.ndim == 2:
        if compress:
            outds = driver.Create(outfile, ncols, nrows, 1, dtype, ['COMPRESS=LZW'])
        else:
            outds = driver.Create(outfile, ncols, nrows, 1, dtype)
        if not quiet:
            print('2-dimensional file created')
    else:
        print("Error")
        return

    outds.SetProjection(proj)
    outds.SetGeoTransform(gt)
    # add metadata if given
    if (metadata != None) & (isinstance(metadata, dict)):
        outds.SetMetadata(metadata)
        if not quiet:
            print("Metadata added")
    if not quiet:
        print(inarray.shape)
    if inarray.ndim == 3:
        for i in range(inarray.shape[0]):
            outBand = outds.GetRasterBand(i + 1)
            outBand.WriteArray(inarray[i, :, :])
            if noData:
                outBand.SetNoDataValue(noDataVal)
            #outBand.WriteArray(array[:,:,i])
    elif inarray.ndim == 2:
        outBand = outds.GetRasterBand(1)
        outBand.WriteArray(inarray[:, :])
        if noData:
            outBand.SetNoDataValue(noDataVal)
    else:
        return

    del (sdataset, outBand, outds)
    if not quiet:
        print("Export completed!")
    return True


def isodate_to_doy(isodate):
    """
    reformat isodate format to day of year
    """
    return datetime.datetime.strptime(isodate, '%Y-%m-%d').strftime('%j')


def get_esd(doy, filepath='P:\\initze\\888_scripts\\esd.csv'):
    """
    read earth sun distance from csv file
    """
    f = open(filepath)
    fx = csv.reader(f, delimiter=',')
    for row in fx:
        it = 0
        for val in row:
            it += 1
            if val == str(doy):
                esd = row[it]
    f.close()
    return float(esd)


def load_data(path, indices=None, xoff=0, yoff=0, xsize=None, ysize=None, factor = 1.,
              filter=True, filetype='tif', outdict=False, **kwargs):
    """
    :param path:
    :param indices:
    :param xoff:
    :param yoff:
    :param xsize:
    :param ysize:
    :param factor:
    :param filter:
    :param filetype:
    :param kwargs:
    :return:
    """
    if indices is None:
        indices = ['tc', 'ndvi', 'ndwi', 'ndmi']
    os.chdir(path)
    flist = glob.glob('*.{0}'.format(filetype))
    flist = sortlist(flist)  # sort images by specific naming pattern
    print(len(flist))
    # filter by years
    idx = filter_year(imdates_from_filename(flist), **kwargs)
    flist = flist[idx]
    print(len(flist))

    idx = filter_month(imdates_from_filename(flist), **kwargs)
    flist = flist[idx]
    print(len(flist))

    s_l = sensorlist(flist)
    print(len(flist))

    # set-up data structures
    ds = []
    imdates = []
    imdates_doy = []

    # load data and extract its properties
    for f in flist:
        dsx = ga.LoadFile(f, xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)
        if dsx.shape[0] == 7:
            dsx = dsx[1:]
        # added masking of zero_values
        dsx = np.ma.masked_values(dsx, 0)
        dsx.mask = dsx.mask.any(axis=0)
        ds.append(dsx)
        imdates.append(datetime.datetime.strptime("{0}-{1}".format(f[9:13], f[13:16]), "%Y-%j").toordinal())
        imdates_doy.append(int(f[13:16]))
    ds = np.ma.MaskedArray(ds)
    imdates = np.array(imdates, dtype=np.int)
    imdates_doy = np.array(imdates_doy, dtype=np.int)

    # apply factor
    ds = ds * 1./factor

    # filter empty files
    """
    valid_idx = np.where((ds.mask[:,0]).reshape(ds.shape[0],-1).mean(axis=1)<1)[0]
    ds = ds[valid_idx]
    imdates = imdates[valid_idx]
    imdates_doy = imdates_doy[valid_idx]
    print len(idx), "images for calculation"
    """

    # apply filter
    if filter:
        ds, imdates, idx = non_duplicates(ds, imdates)
        imdates_doy, flist, s_l = [np.array(dataset)[idx] for dataset in [imdates_doy, flist, s_l]]

    # Index Calculation
    ind = []

    print("start index")

    if 'tc' in indices:
        tc = [tasseled_cap(i, s) for i, s in zip(ds, s_l)]
        tc = np.ma.masked_equal(tc, 0)
        ind.append(tc)

    if 'ndvi' in indices:
        ndvi_ = [ndvi(i, 3, 2) for i, s in zip(ds, s_l)]
        ndvi_ = np.ma.masked_equal(ndvi_, 0)
        ind.append(ndvi_)

    if 'ndwi' in indices:
        ndwi = [ndvi(i, 1, 3) for i, s in zip(ds, s_l)]
        ndwi = np.ma.masked_equal(ndwi, 0)
        ind.append(ndwi)

    if 'ndmi' in indices:
        ndmi = [ndvi(i, 3, 4) for i, s in zip(ds, s_l)]
        ndmi = np.ma.masked_equal(ndmi, 0)
        ind.append(ndmi)

    if 'ndbr' in indices:
        ndbr = [ndvi(i, 3, 5) for i, s in zip(ds, s_l)]
        ndbr = np.ma.masked_equal(ndbr, 0)
        ind.append(ndbr)

    if outdict:
        outdata = dict(data=ds,
                       indices=dict(list(zip(indices, ind))),
                       imdates=imdates,
                       imdates_doy=imdates_doy,
                       flist=flist,
                       sensor=s_l)

        return outdata
    else:
        return ds, imdates, imdates_doy, ind


def filter_year(imdates, startyr=0, endyr=datetime.datetime.now().year, **kwargs):
    dt = np.array([datetime.datetime.fromordinal(dt).year for dt in imdates])
    return np.where((dt >= startyr) & (dt <= endyr))[0]


def filter_month(imdates, startmth=6, endmth=9, **kwargs):
    if 'startmth' in list(kwargs.keys()):
        startmth = kwargs.startmth
    if 'endtmth' in list(kwargs.keys()):
        endmth = kwargs.startmth
    dt = np.array([datetime.datetime.fromordinal(dt).month for dt in imdates])
    return np.where((dt >= startmth) & (dt <= endmth))[0]


def imdates_from_filename(flist):
    """
    :param flist:
    :return:
    """
    return [datetime.datetime.strptime("{0}-{1}".format(f[9:13], f[13:16]), "%Y-%j").toordinal() for f in flist]


def get_rp_from_fishnet(infile):
    """
    :param infile:
    :return:
    """
    with fiona.open(infile) as src:
        out = []
        for feat in src:
            p = feat['properties']
            out.append('{0}_{1}'.format(p['row'], p['path']))
    return out


def tiling(xsize, ysize, xstepsize, ystepsize):
    """
    :param xsize:
    :param ysize:
    :param xstepsize:
    :param ystepsize:
    :return:
    """
    x, y = np.mgrid[0:xsize:xstepsize, 0:ysize:ystepsize]
    x = x.ravel()
    y = y.ravel()
    dist_x = xsize-x
    dist_x[dist_x > xstepsize] = xstepsize
    dist_y = ysize-y
    dist_y[dist_y > ystepsize] = ystepsize

    return x, y, dist_x, dist_y


def compress_folders(indir, delete=True):
    """
    :param indir: path to directory
    :param delete: switch to delete inputfolder
    :return: None
    """
    flist = get_foldernames(indir, global_path=True)
    for f in flist:
        subprocess.call('7z a -mx=1 {0}.7z {0}'.format(f), shell=True)
        if delete:
            shutil.rmtree(f)


def compress_geotiff(infolder, delete=True, direct=False):
    """
    :param infolder: path to directory with subfolder structure for masked WRS-tiles
    :param delete: switch to delete inputfiles
    :param direct:
    :return: None
    """
    if direct:
        files = glob.glob('{0}/*.tif'.format(infolder))
        print(files)
        for infile in files:
            outfile = infile[:-4] + r'_cmp.tif'
            execstring = 'gdal_translate -co COMPRESS=LZW {0} {1}'.format(infile, outfile)
            os.system(execstring)
            if delete:
                os.remove(infile)
                shutil.move(outfile, infile)
    else:
        indirs = get_foldernames(infolder, global_path=False)

        for indir in indirs:
            infile = os.path.join(indir, "tmp", indir.split('-')[0] + '_masked.tif')
            outfile = os.path.join(indir, "tmp", indir.split('-')[0] + '_masked_compr.tif')

            if not os.path.exists(infile):
                print("Input file does not exist, skip!")
                continue

            # check if already compressed, if yes skip

            execstring = 'gdal_translate -co COMPRESS=LZW {0} {1}'.format(infile, outfile)
            os.system(execstring)
            if delete:
                os.remove(infile)
                shutil.move(outfile, infile)


def uncompress_folders(indir, delete=True, parallel=False):
    """
    :param indir: path to directory with 7z-file archives
    :param delete: switch to delete inputfiles
    :return:
    """
    flist = glob.glob(os.path.join(indir, '*.7z'))
    if parallel:
        joblib.Parallel(n_jobs=10)(joblib.delayed(subprocess.call)('7z x -o{1} {0}'.format(f, indir), shell=True) for f in flist)
        if delete:
            [os.remove(f) for f in flist]
    else:
        for f in flist:
            print(f)
            subprocess.call('7z x -o{1} {0}'.format(f, indir), shell=True)
            if delete:
                os.remove(f)


class Masking(object):
    def __init__(self, infolder, dst_file, valid_val=None, dst_nodata=0, cleanup=True, processing_type='sr'):
        if valid_val is None:
            valid_val = [0, 1]
        self.infolder = infolder
        self.dst_file = dst_file
        self.valid_val = valid_val
        self.dst_nodata = dst_nodata
        self.cleanup = cleanup
        self.processing_type = processing_type
        self.meta_ok = False
        self.mask_ok = False
        self.files_ok = False
        self.masked = False
        self.exported = False
        self.compressed = False
        self._find_all_files()
        self._find_metadata()
        if self.meta_ok:
            self._find_basename()
            self._find_maskfile()
            self._find_bands()
            self._make_filelist()

    def _find_all_files(self):
        self.all_files = glob.glob(self.infolder + '/*.*')

    def _find_metadata(self):
        md = glob.glob(self.infolder + '/*.xml')
        md = [y for y in md if '.aux.' not in y]
        if len(md) != 1:
            return
        self.metadata = os.path.abspath(md[0])
        self.meta_ok = os.path.exists(self.metadata)

    def _find_basename(self):
        self.basename = os.path.split(self.metadata)[-1].split('.')[-2]

    def _find_maskfile(self):
        self.maskfile = os.path.abspath(os.path.join(self.infolder, self.basename + '_cfmask.tif'))
        self.mask_ok =  os.path.exists(self.maskfile)

    def _find_bands(self):
        self.sensor = get_sensor_from_metadata(self.metadata)
        self.bands = bands[self.sensor]

    def _make_filelist(self):
        self.filelist = []
        for band in self.bands:
            f = '{0}_{1}_band{2}.tif'.format(self.basename, self.processing_type, band)
            f = os.path.join(self.infolder, f)
            self.filelist.append(f)
            self.files_ok = np.all([os.path.exists(f) for f in self.filelist])

    def make_mask(self):
        self._load_data()
        self._load_mask()
        self._calc_mask_indices()
        self._apply_mask()

    def _load_data(self):
        self.data = np.array([ga.LoadFile(f) for f in self.filelist])

    def _load_mask(self):
        self.mask = ga.LoadFile(self.maskfile)

    def _calc_mask_indices(self):
        self.m_idx = ~(np.in1d(self.mask, self.valid_val).reshape(self.mask.shape))

    def _apply_mask(self):
        self.data[:, self.m_idx] = self.dst_nodata
        self.masked = True

    def save_raster(self):
        tmpfolder = os.path.dirname(self.dst_file)
        if self.masked:
            if not os.path.exists(tmpfolder):
                os.makedirs(tmpfolder)
            array_to_file(self.data, self.dst_file, self.filelist[0], dtype=gdal.GDT_UInt16, compress=False)
            self.exported = True

    def compress_raster(self):
        if self.exported:
            tmpfile = self.dst_file + '_tmp.tif'
            shutil.move(self.dst_file, tmpfile)
            os.system("gdal_translate -co COMPRESS=LZW {0} {1}".format(tmpfile, self.dst_file))
            os.remove(tmpfile)
            self.compressed = True

    def cleanup_dir(self):
        if self.cleanup:
            [os.remove(f) for f in self.all_files]

    def processing_assessment(self):
        if self.masked:
            print("Masking: OK")
        if self.exported:
            print("File Export: OK")
        if self.compressed:
            print("File Compression: OK")


class MaskingNG(Masking):

    def _find_maskfile(self):
        """
        find pixel_qa file
        :return:
        """
        self.maskfile = os.path.abspath(os.path.join(self.infolder, self.basename + '_pixel_qa.tif'))
        self.mask_ok = os.path.exists(self.maskfile)

    def _calc_mask_indices(self):
        masks_bin = self._check_mask_vals(self.mask)
        self.m_idx = ~np.any(masks_bin[[1, 2]], axis=0)

    @staticmethod
    def _check_mask_vals(decimal_mask):
        """function that translates Landsats pixel_qa values into bits
        returns boolean if mask needs to be applied --> True"""
        r, c = decimal_mask.shape
        m = np.unpackbits(np.array(decimal_mask, dtype=np.uint8)).reshape(r, c, 8)
        m = np.flipud(np.rollaxis(m, 2, 0))
        return m


class WRS_Mover(object):
    def __init__(self, src_dir, dst_dir=None, study_site=None):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.study_site = study_site
        self.moved = False
        self.get_pr()
        self.set_dst()
        self.make_dst()
        self.make_outfile_path()

    #TODO: update for new landsat naming convention
    #TODO: TESTING needed
    def get_pr(self):
        """
        get WRS-2 path and row values from directory name
        :return:
        """
        self.bn = os.path.basename(self.src_dir)
        # check if file has new or old landsat naming style
        if len(self.bn.split('-')[0]) == 22:
            # new style with 22 digit naming
            self.p = self.bn[4:7]
            self.r = self.bn[8:10]
        else:
            # old style
            self.p = self.bn[3:6]
            self.r = self.bn[7:9]

    def set_dst(self):
        """

        :return:
        """
        if (not self.dst_dir) and self.study_site:
            self.dst_dir = study_sites[self.study_site]['data_dir']

    def make_dst(self):
        """

        :return:
        """
        self.outpath = os.path.join(self.dst_dir, 'p{0}_r{1}'.format(self.p, self.r))
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

    def make_outfile_path(self):
        """

        :return:
        """
        self.outfile = os.path.join(self.outpath, self.bn)

    def move(self):
        """

        :return:
        """
        if not os.path.exists(self.outfile):
            shutil.move(self.src_dir, self.outpath)
            self.moved = True


def load_ts_from_coords(datadir, coordinates):
    """
    :param datadir:
    :param coordinates:
    :return:
    """
    f_0 = glob.glob(os.path.join(datadir, '*.tif'))[0]
    if f_0 == None:
        raise IOError('No Data available')
    with rasterio.open(f_0) as src:
        cc, rr = src.index(*coordinates)
        print(rr, cc)
        print(coordinates)
    if (rr in range(1000)) and (cc in range(1000)):
        print("Loading Data")
        data = load_data(datadir, xoff=rr, xsize=1, yoff=cc, ysize=1, indices=['tc', 'ndvi', 'ndwi', 'ndmi'], factor=10000., outdict=True)
    else:
        print("Coordinates outside the image extent")
    return data


md_specs = {'"TM"' : {'acq_date': 'DATE_ACQUIRED',
                      'sun_el':'SUN_ELEVATION',
                      'lmin':'RADIANCE_MINIMUM_BAND_',
                      'lmax':'RADIANCE_MAXIMUM_BAND_',
                      'qcalmin':'QUANTIZE_CAL_MIN_BAND_',
                      'qcalmax':'QUANTIZE_CAL_MAX_BAND_',
                      'esun':{1:1983., 2:1796., 3:1536., 4:1031., 5:220., 7:83.49},
                      'thermal':''
                      },
            '"ETM"': {'acq_date': 'DATE_ACQUIRED',
                      'sun_el':'SUN_ELEVATION',
                      'lmin':'RADIANCE_MINIMUM_BAND_',
                      'lmax':'RADIANCE_MAXIMUM_BAND_',
                      'qcalmin':'QUANTIZE_CAL_MIN_BAND_',
                      'qcalmax':'QUANTIZE_CAL_MAX_BAND_',
                      'esun':{1:1997., 2:1812., 3:1533., 4:1039., 5:230.8, 7:84.90, 8:1362.},
                      'thermal':'_VCID_1'
                      },
            '"OLI_TIRS"':{'acq_date': 'DATE_ACQUIRED',
                          'sun_el':'SUN_ELEVATION'
                          }
            }
metadata_trends = {'br':{'DS_INFO': 'Slope of Landsat Tasseled Cap Brightness'},
                   'gr':{'DS_INFO': 'Slope of Landsat Tasseled Cap Greenness'},
                   'we':{'DS_INFO': 'Slope of Landsat Tasseled Cap Wetness'},
                   'ndvi':{'DS_INFO': 'Slope of NDVI'},
                   'ndwi':{'DS_INFO': 'Slope of NDWI'},
                   'ndsi':{'DS_INFO': 'Slope of NDMI'}
                   }
bands = {'TM':[1,2,3,4,5,7], 'ETM':[1,2,3,4,5,7], 'OLI_TIRS':[2,3,4,5,6,7]}


def ls_get_metadata(filepath, feature):
    """Extract value of metadata feature
    -----------------
    Input:
    filepath : string - path to metadatafile (e.g. './LC82070232013192LGN00_MTL.txt')
    feature : string - name of metadata feature (e.g. 'SUN_ELEVATION')
    -----------------
    Output
    item : string - Output item of chosen metadata feature
    """
    f = open(filepath)
    meta = f.readlines()
    f.close()
    item = [x for x in meta if feature in x][0]
    out = item.replace(" ", "").replace("\n","").split('=')[-1]
    try:
        out = float(out)
    except:
        out = str(out)
    return out


def dn_to_radiance(raster, Lmin, Lmax, Qcalmin, Qcalmax):
    """
    functions calculates Landsat5-7 DNs to Radiance
    rho_lambda_s : TOA planetary reflectance, without correction for solar angle.  Note that ρλ' does not contain a correction for the sun angle.
    m_rho :  Band-specific multiplicative rescaling factor from the metadata (REFLECTANCE_MULT_BAND_x, where x is the band number)
    a_rho : Band-specific additive rescaling factor from the metadata (REFLECTANCE_ADD_BAND_x, where x is the band number)
    image : Image DN
    """
    radiance = np.zeros_like(raster, dtype=np.float32)
    nonzero = np.where(raster != 0)
    radiance[nonzero] = ((Lmax - Lmin) / (Qcalmax-Qcalmin)) * (raster[nonzero]-Qcalmin) + Lmin
    return radiance


def ls8_to_radiance(raster, ml, al):
    """
    functions calculates Landsat8 DNs to Radiance
    -----------------
    Input:
    raster : numpy array - 2-dimensional numpy array with DN
    ml : float - Band-specific multiplicative rescaling factor from the metadata (RADIANCE_MULT_BAND_x, where x is the band number)
    al : string - Band-specific additive rescaling factor from the metadata (RADIANCE_ADD_BAND_x, where x is the band number)
    -----------------
    Output
    radiance : numpy-array - 2-dimensional numpy array with TOA spectral radiance values (w/(m^2 * srad * µm))
    """
    radiance = np.zeros_like(raster, dtype=np.float32)
    nonzero = np.where(raster != 0)
    radiance[nonzero] = ml * raster[nonzero] + al
    return radiance


def radiance_to_reflectance(radiance, solar_el, esun, es_dist=1):
    """
    functions calculates Landsat8 DNs to Reflectance
    rho_lambda_s : TOA planetary reflectance, without correction for solar angle.  Note that ρλ' does not contain a correction for the sun angle.
    m_rho :  Band-specific multiplicative rescaling factor from the metadata (REFLECTANCE_MULT_BAND_x, where x is the band number)
    a_rho : Band-specific additive rescaling factor from the metadata (REFLECTANCE_ADD_BAND_x, where x is the band number)
    image : Image DN
    """
    pi = 3.141592653589793
    reflectance = np.zeros_like(radiance, dtype=np.float32)
    nonzero = np.where(radiance != 0)
    reflectance[nonzero] = (pi * radiance[nonzero] * es_dist**2) / (esun * np.sin(np.radians(solar_el)))
    return reflectance


def ls8_to_reflectance(raster, solar_el, m_rho=2.0000e-05, a_rho=-0.1):
    """
    functions calculates Landsat8 DNs to Reflectance
    rho_lambda_s : TOA planetary reflectance, without correction for solar angle.  Note that ρλ' does not contain a correction for the sun angle.
    m_rho :  Band-specific multiplicative rescaling factor from the metadata (REFLECTANCE_MULT_BAND_x, where x is the band number)
    a_rho : Band-specific additive rescaling factor from the metadata (REFLECTANCE_ADD_BAND_x, where x is the band number)
    image : Image DN
    """
    reflectance = np.zeros_like(raster, dtype=np.float32)
    nonzero = np.where(raster != 0)
    reflectance[nonzero] = (m_rho * raster[nonzero] + a_rho) / np.sin(np.radians(solar_el))
    return reflectance


def tasseled_cap(im, sensor = 'OLI'):
    """
    Function for the calculation of tasseled cap components of Landsat at-satellite reflectances
    """
    if im.ndim != 3:
        print("wrong number of dimensions")
        return 1

    if sensor == 'TM':
        #Crist (1985). A TM Tasseled Cap Equivalent Transformation for Reflectance Factor Data.
        factor_b = [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]
        factor_g = [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446]
        factor_w = [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]
        bands = list(range(6))

    elif sensor =='ETM':
        #Huang et al. (2002). Derivation of a tasselled cap transformation based on Landsat 7 at-satellite reflectance.
        factor_b = [0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596]
        factor_g = [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630]
        factor_w = [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388]
        bands = list(range(6))

    elif sensor in ['OLI', 'OLI_TIRS']:
        #Ali Baig et al. (2014). Derivation of a tasselled cap transformation based on Landsat 8 at-satellite reflectance.
        factor_b = [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872]
        factor_g = [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608]
        factor_w = [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]
        bands = list(range(6))

    else:
        print("Unknown input sensor")
        return 1

    outim = np.ma.zeros((3, im.shape[1], im.shape[2]), dtype=np.float)

    #brightness
    for f, band in zip(factor_b, bands):
        outim[0] += f * im[band]
    #greenness
    for f, band in zip(factor_g, bands):
        outim[1] += f * im[band]
    #wetness
    for f, band in zip(factor_w, bands):
        outim[2] += f * im[band]

    outim.mask = im[0].mask # set mask
    return outim


def ndvi(im, bandnir, bandred, nodata=0):
    ind = im.sum(axis=0).nonzero()
    im_ndvi = np.zeros(im.shape[1:], dtype=np.float)
    im_ndvi[ind] = (im[bandnir][ind] - im[bandred][ind]) / (im[bandnir][ind] + im[bandred][ind])
    return im_ndvi


def bai(im, bandnir, bandred, nodata=0):
    """
    Burn Area Index
    Chuvieco, E., M. Pilar Martin, and A. Palacios. “Assessment of Different Spectral Indices in the Red-Near-Infrared
    Spectral Domain for Burned Land Discrimination.” Remote Sensing of Environment 112 (2002): 2381-2396.
    :param im:
    :param bandnir:
    :param bandred:
    :param nodata:
    :return:
    """
    ind = im.sum(axis=0).nonzero()
    bai = np.zeros(im.shape[1:], dtype=np.float)
    bai[ind] = (1. / ((0.1 - im[bandred][ind])**2 + (0.06 - im[bandnir][ind])))
    return bai


def align_px_to_ls(startvalue):
    """align pixel locations to standard pixel edges of ls"""
    return startvalue - (startvalue % 30 - 15)


def get_sensor_from_metadata(filepath):
    xmldoc = minidom.parse(filepath)
    sl = xmldoc.getElementsByTagName('instrument')[0]
    sensor = sl.childNodes[0].data
    return sensor


def get_ullr_from_json(json):
    """

    :param json:
    :return:
    """
    crd = np.array(json['geometry']['coordinates'][0][:4])
    ulx, uly, lrx, lry = crd[[0,2]].ravel()
    return ulx, uly, lrx, lry


def merge_pr(df):
    """
    function to create pr_string for data frame with path and row features
    """
    return '{r}_{p}'.format(r=df['row'], p=df['path'])


def load_gt(zone, gt_layer=r'E:\05_Vector\01_GroundTruth\01_GT_manual_AK_LL.shp', cirange=False):
    """
    function to load ground truth data and its associated trend values to a Pandas Dataframe"""

    flist = glob.glob(r'F:\06_Trendimages\{zone}_2016_2mths\1999-2014\tiles'.format(zone=zone))
    fn_file = study_sites[zone]['fishnet_file']

    # setup column names
    new_cols = []
    idxlist = ['tcb', 'tcg', 'tcw', 'ndvi', 'ndwi', 'ndmi']
    idlist = ['slp', 'ict', 'cil', 'ciu']

    #load shapefiles to dataframes, reproject to 4326 if necessary
    gp_df_groundtruth = geopandas.GeoDataFrame.from_file(gt_layer)
    #gt_cols = list(gp_df_groundtruth.columns)
    gp_df_fishnet = geopandas.GeoDataFrame.from_file(fn_file).to_crs({'init':'epsg:4326'})

    # spatial join fishnet file and ground truth data - keep only GT data that are inside fishnet
    joined = geopandas.tools.sjoin(gp_df_groundtruth, gp_df_fishnet)
    # remove unnecessary columns
    joined_filt = joined[list(gp_df_groundtruth.columns) + ['row', 'path']]
    # calculate pr_string
    joined_filt.loc[:, 'pr_string'] = joined_filt.apply(merge_pr, axis=1)

    # get all tile pr_strings where there are data
    prlist = joined_filt['pr_string'].unique()

    # reproject dataframe to current zone (UTM)
    df_temp_reprojected = joined_filt.to_crs({'init':'epsg:326{zid}'.format(zid=zone[-2:])})

    # add columns to dataframe with NaN as default value
    for idx in idxlist:
        for i in idlist:
            new_cols.append(idx + '_' + i)
            joined_filt.loc[:, idx + '_' + i] = np.nan

    # iterate over each tile (pr)
    for pr in prlist[:]:
        # create filtered dataframe of GT points only within currently selected tile
        local = df_temp_reprojected[df_temp_reprojected.pr_string==pr]
        # get UTM coordinates
        crds = [(crd.x, crd.y) for crd in local.geometry.values]
        # iterate over index (Tasselled Cap etc.)
        for idx in idxlist:
            # create list of feature/column names
            feat_name = [idx + ext for ext in ('_slp', '_ict', '_cil', '_ciu')]
            # find tile and index specific trend image
            f = glob.glob(r'F:\06_Trendimages\{zone}_2016_2mths\1999-2014\tiles\trendimage_*{0}*{index}*'.format(pr,
                                                                                                           index=idx,
                                                                                                           zone=zone))
            # check if image is found
            if len(f) != 0:
                # open file and get raster values from Ground Truth points
                with rasterio.open(f[0]) as src:
                    v = [smp for smp in src.sample(crds)]
                    # create temporary data frame and update record in parent file for the zone
                    xx = pd.DataFrame(data=v, columns=feat_name, index=local.index)
                    joined_filt.update(xx, join='left')
                    # calc cirange if indicated
                    if cirange:
                        joined_filt[idx + '_cirange'] = joined_filt[idx + '_ciu'] - joined_filt[idx + '_cil']
    # remove points where no data could be retrieved
    data_frame = joined_filt[joined_filt.ndvi_slp.notnull()]
    # indicate zone where ground truth is located
    data_frame['zone'] = zone

    return data_frame


def combine_idxlist(idxlist, extlist):
    outlist = []
    for idx in idxlist:
        for ext in extlist:
            outlist.append(idx + ext)
    return outlist


def load_era_interim(filepath):
    """
    Function to extract data from ntcdf file with monthly 't2m' and 'p' for analysis of paper 3
    :param filepath: str - path to nc file
    :return: latitude, longitude, time, temperature, precipitation
    """

    ds = netCDF4.Dataset(r'F:/21_Paper03_Analysis/06_ExternalData/Climate/ERA_interim_global/ERA_interim.nc')

    lats = ds.variables['latitude'][:]  # extract/copy the data
    lons = ds.variables['longitude'][:]
    time = ds.variables['time'][:]
    t = ds.variables['t2m'][:] - 273.15
    p = ds.variables['tp'][:]

    base_date = datetime.datetime(1900,1,1,0,0,0)
    time_corr = [base_date + datetime.timedelta(hours=tm) for tm in time]
    time_corr = np.array(time_corr)[list(range(0, len(time_corr), 2))]

    shp = t.shape
    t_res = t.reshape(shp[0]/2,2,shp[1], shp[2]).mean(axis=1)

    shp = p.shape
    p_res = p.reshape(shp[0]/2,2,shp[1], shp[2]).sum(axis=1)*30*10000

    time = pd.DataFrame(data=time_corr, columns=['datetime'])
    time['year'] = [t.year for t in time_corr]
    time['month'] = [t.month for t in time_corr]
    ds.close()
    return lats, lons, time, t_res, p_res


def get_statistics(data):
    """
    get regional statistics from lake change dataset and write into pandas Series
    :param data:
    :return:
    """
    A = data.sum()
    index = {'Percentage Net Change' : (A['area_end_ha'] / A['area_start_ha'] - 1) * 100,
             'Absolute Net Change' : A['netchange_ha'],
             'Water area T0' : A['area_start_ha'],
             'Water area T1' : A['area_end_ha'],
             'Gross Area Increase [ha]' : A['grossincrease_ha'],
             'Gross Area Increase [%]' : A['grossincrease_ha'] / A['area_start_ha'] * 100,
             'Gross Area Decrease [ha]' : A['grossdecrease_ha'],
             'Gross Area Decrease [%]' : A['grossdecrease_ha'] / A['area_end_ha'] * 100,
             'Stable water area [ha]' : A['stablewater_ha'],
             'Number of Lakes' : len(data),
             'Mean Size T0' : data['area_start_ha'].mean(),
             'Mean Size T1' : data['area_end_ha'].mean(),
             'Median Size T0' : data['area_start_ha'].median(),
             'Median Size T1' : data['area_end_ha'].median(),
             'Max Size T0' : data['area_start_ha'].max(),
             'Max Size T1' : data['area_end_ha'].max(),
             'Gain - quadruple' : sum(( data['area_end_ha']  / data['area_start_ha']) > 4),
             'Gain - quadruple [%]' : (sum(( data['area_end_ha']  / data['area_start_ha']) > 4)) / float(len(data)) * 100.,
             'Loss to quarter' : sum(( data['area_start_ha'] / data['area_end_ha']) > 4),
             'Loss to quarter [%]' : (sum(( data['area_start_ha'] / data['area_end_ha']) > 4))  / float(len(data)) * 100.,
             'stable Lakes' : (((data['stablewater_ha'] / data[['stablewater_ha','grossdecrease_ha', 'grossincrease_ha']].sum(axis=1)) * 100) > 95).sum()
             }
    return pd.Series(index)


def geom_sr_from_point(x, y, epsg):
    """
    create geometry from 4 corner coordinates and a specified epsg code
    returns ogr.Geometry and ogr.SpatiaReference of the geometry
    """
    wkt_point = 'POINT({0} {1})'.format(x, y)
    geom_point = ogr.CreateGeometryFromWkt(wkt_point)
    sr_point = ogr.osr.SpatialReference()
    sr_point.ImportFromEPSG(int(epsg))
    geom_point.AssignSpatialReference(sr_point)
    return geom_point, sr_point


def geom_sr_from_bbox(ulx, uly, lrx, lry, epsg):
    """
    create geometry from 4 corner coordinates and a specified epsg code
    returns ogr.Geometry and ogr.SpatiaReference of the geometry
    """
    wkt_bbox = 'POLYGON(({0} {1}, {2} {1}, {2} {3}, {0} {3}, {0} {1}))'.format(ulx, uly, lrx, lry)
    geom_bbox = ogr.CreateGeometryFromWkt(wkt_bbox)
    sr_bbox = ogr.osr.SpatialReference()
    sr_bbox.ImportFromEPSG(int(epsg))
    geom_bbox.AssignSpatialReference(sr_bbox)
    return geom_bbox, sr_bbox


def geom_from_wrs2Bounds(infile, path, row):
    """
    return ogr.Geometry of WRS-2 Landsat tiles
    """
    ds = ogr.Open(infile)
    lyr = ds.GetLayerByIndex(0)
    lyr.GetFeatureCount()
    fc = lyr.GetFeatureCount()
    fci = list(range(0, fc))
    for i in fci:
        feat = lyr.GetFeature(i)
        p, r = feat.GetField(feat.GetFieldIndex('PATH')), feat.GetField(feat.GetFieldIndex('ROW'))
        if (int(p), int(r)) == (int(path), int(row)):
            print(p, r)
            outgeom = feat.GetGeometryRef().Clone()

    return outgeom


def geom_from_fishnet(infile, row, col):
    """
    return corner coordinates of project specific tile-layer
    """
    ds = ogr.Open(infile)
    lyr = ds.GetLayerByIndex(0)
    lyr.GetFeatureCount()
    fc = lyr.GetFeatureCount()
    fci = list(range(0, fc))
    for i in fci:
        feat = lyr.GetFeature(i)
        r, p = feat.GetField(feat.GetFieldIndex('row')), feat.GetField(feat.GetFieldIndex('path'))
        if (r, p) == (row, col):
            geom = feat.GetGeometryRef().Clone()
            return geom


def intersect_geoms(geom1, geom2):
    """
    check if geometries of different projections intersect
    """
    sr = geom1.GetSpatialReference()
    geom2.TransformTo(sr)
    return geom1.Intersects(geom2)


def get_bounds(infile, row, col):
    """
    return corner coordinates of project specific tile-layer
    -----------------
    input:
    infile : path to input vectorfile
    row : row id of fishnet vectorfile
    col : column id of fishnet vectorfile
    -----------------
    output:
    xmin, xmax, ymin, ymax
    """
    ds = ogr.Open(infile)
    lyr = ds.GetLayerByIndex(0)
    lyr.GetFeatureCount()
    fc = lyr.GetFeatureCount()
    fci = list(range(0, fc))
    for i in fci:
        feat = lyr.GetFeature(i)
        r, p = feat.GetField(feat.GetFieldIndex('row')), feat.GetField(feat.GetFieldIndex('path'))
        if (r, p) == (row, col):
            xmin, xmax, ymin, ymax = feat.GetField('XMIN'), feat.GetField('XMAX'), feat.GetField('YMIN'), feat.GetField(
                'YMAX')
    ds = None
    if not xmin or not ymin:
        print("indicated path and row not available")
        pass
    return xmin, xmax, ymin, ymax


def epsg_from_raster(infile):
    """
    returns epsg code from rasterfile

    :param
    infile: filepath to rasterfile (str)
    :return: epsg code (str)
    """
    ds = gdal.Open(infile)
    epsg = ds.GetProjection().split('"')[-2]
    ds = None
    return epsg


def reproject_on_the_fly(dsin, outepsg, xmin, ymax, no_data=0, xbuffer=10, ybuffer=10):
    """
    input:
    dsin : path to rasterfile (str)
    outepsg : epsg-code of outfile (int)
    xmin : minimum x-coordinate (float)
    ymax : maximum y-corrdinate (float)
    xbuffer : buffer in number of pixels in x-direction (int)
    ybuffer : buffer in number of pixels in y-direction (int)
    """
    # open dataset
    if not os.path.exists(dsin):
        print("file does not exist")
        return
    print(dsin)
    ds = gdal.Open(dsin)

    # get number of bands
    n_bands = ds.RasterCount

    # set no_data value for each band
    [ds.GetRasterBand(band).SetNoDataValue(no_data) for band in range(1, n_bands + 1)]

    # Get old geotransform and create new one
    geo_t = ds.GetGeoTransform()
    new_geo = (xmin-(geo_t[1]*xbuffer), geo_t[1], geo_t[2], ymax-(geo_t[5]*ybuffer), geo_t[4], -30)

    # define and get spatial references
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(outepsg)

    # create dataset in memory only
    mem_drv = gdal.GetDriverByName('MEM')
    # FIX
    dest = mem_drv.Create('', 1020, 1020, n_bands, gdal.GDT_Float32)
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(new_cs.ExportToWkt())

    # Reproject
    gdal.ReprojectImage(ds, dest, old_cs.ExportToWkt(), new_cs.ExportToWkt(), gdal.GRA_Cubic)

    # read array content
    rb = np.array([dest.GetRasterBand(i).ReadAsArray() for i in range(1, n_bands + 1)])

    # remove datasets from memory
    dest = None
    ds = None
    return rb


def geom_from_rasterfile(filepath):
    """

    :param filepath:
    :return:
    """
    ds = gdal.Open(filepath)
    gt = ds.GetGeoTransform()  # corner coordinates and resolution
    X = (gt[0], gt[0] + gt[1] * ds.RasterXSize)  # create bbox wkt
    Y = (gt[3], gt[3] + gt[5] * ds.RasterYSize)  # create bbox wkt
    wkt_raster = 'POLYGON(({0} {1}, {2} {1}, {2} {3}, {0} {3}, {0} {1}))'.format(X[0], Y[0], X[1], Y[1])
    geom_raster = ogr.CreateGeometryFromWkt(wkt_raster)
    # assign spatial reference of bbox
    sr_raster = ogr.osr.SpatialReference()
    sr_raster.ImportFromWkt(ds.GetProjection())
    geom_raster.AssignSpatialReference(sr_raster)
    ds = None
    return geom_raster


def geom_sr_from_vectorfile(filepath, fid=None):
    """

    :param filepath:
    :param fid:
    :return:
    """
    ds = ogr.Open(filepath)
    lyr = ds.GetLayerByIndex(0)
    fc = lyr.GetFeatureCount()
    sr = lyr.GetSpatialRef()
    #outgeom = ogr.Geometry(ogr.wkbGeometryCollection)
    #outgeom.AssignSpatialReference(sr)
    tmp_feat = lyr.GetFeature(0)
    outgeom = tmp_feat.geometry().Clone()

    if fid is None and fc > 1:
        for fid in range(1, fc):
            feat = lyr.GetFeature(fid)
            geom = feat.geometry()
            outgeom = outgeom.Union(geom)
    else:
        if fid > fc:
            raise ValueError("FID too large, only {0} features exist! FIDs start at 0".format(fc))

    return outgeom, sr


def export_raster_geom(ingeoms, outfilename, epsg, fp=None):
    """

    :param ingeoms:
    :param outfilename:
    :param epsg:
    :param fp:
    :return:
    """
    if not fp:
        fp = [' '] * len(ingeoms)
    # create driver and file
    ext = outfilename.lower()[-4:]
    drname = {'.shp': 'ESRI Shapefile', '.kml': 'KML'}
    # check if file is shp or kml
    if ext in list(drname.keys()):
        dr = ogr.GetDriverByName(drname[ext])
    else:
        raise ValueError("Filetype not supported, must be in {0} !".format(list(drname.keys())))

    ds = dr.CreateDataSource(outfilename)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)

    lyr = ds.CreateLayer("Raster_Boundaries", srs=sr)
    fddef = ogr.FieldDefn("Filepath", ogr.OFTString)
    lyr.CreateField(fddef)
    ldef = lyr.GetLayerDefn()

    for g, n in zip(ingeoms, fp):
        feat = ogr.Feature(ldef)
        g.TransformTo(sr)
        feat.SetGeometry(g)
        feat.SetField(0, n)
        lyr.CreateFeature(feat)
    ds.Destroy()
    return 0


def get_raster_size(path):
    ds = gdal.Open(path)
    return ds.RasterYSize, ds.RasterXSize


def create_fishnet(study_site, step=30000):
    """
    :param study_site: string
    :param step:
    :return:
    """
    # check if study site exists, otherwise pass
    if study_site not in list(study_sites.keys()):
        print("Study site is not defined")
        pass

    # define schema of vectorfile
    schema = {'geometry': 'Polygon',
         'properties':{
            'ID':'int',
            'XMIN':'int',
            'XMAX':'int',
            'YMIN':'int',
            'YMAX':'int',
            'row':'int',
            'path':'int'
         }}

    # get calculation properties
    ss = study_sites[study_site]
    outfile = ss['fishnet_file']
    epsg = ss['epsg']
    bbox = ss['bbox']
    xstart, xstop, ystart, ystop = [align_px_to_ls(crd) for crd in bbox]

    # sort in case bounds are swapped
    xstart, xstop = np.sort([xstart, xstop])
    ystart, ystop = np.sort([ystart, ystop])

    # set up coordinates
    path = np.arange(xstart, xstop, step)
    p_idx = list(range(len(path)))
    row = np.arange(ystop-step, ystart-step, -step)
    r_idx = list(range(len(row)))
    idx = 0

    with fiona.open(outfile, mode='w', driver='ESRI Shapefile', schema=schema, crs=from_epsg(epsg)) as source:
        for r, ri in zip(row, r_idx):
            for p, pi in zip(path, p_idx):
                xmin, xmax, ymin, ymax = p, p+step, r, r+step
                p = Polygon(([xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]))
                idx += 1

                source.write({'geometry':mapping(p),
                        'properties':{
                        'ID':idx,
                        'XMIN':xmin,
                        'XMAX':xmax,
                        'YMIN':ymin,
                        'YMAX':ymax,
                        'row':ri,
                        'path':pi
                }})

    pass


def get_datafolder_old(study_site, coords):
    # TODO: find better method --> spatial select?
    epsg = study_sites[study_site]['epsg']
    geom_pt, sr_pt = geom_sr_from_point(coords[0], coords[1], epsg)
    with fiona.open(study_sites[study_site]['fishnet_file']) as src:
        for s in src:
            ulx, uly, lrx, lry = get_ullr_from_json(s)
            geom_bbox, osr_bbox = geom_sr_from_bbox(ulx, uly, lrx, lry, epsg)
            if geom_bbox.Intersects(geom_pt):
                row, path = (s['properties']['row'], s['properties']['path'])
                break
    pdir = study_sites[study_site]['processing_dir'] + r'\{0}_{1}_{2}'.format(study_site, row, path)
    return pdir


def global_to_local_coords(datadir, coordinates, force_inside=True):
    """
    :param datadir: string
    :param coordinates: tuple/array/list
    :return:
    """
    f_0 = glob.glob(os.path.join(datadir, '*.tif'))[0]
    if f_0 is None:
        raise IOError('No Data available')
    with rasterio.open(f_0) as src:
        cc, rr = src.index(*coordinates)
        hgt = src.height
        wth = src.width

    if force_inside:
        if (rr < hgt) & (rr >= 0) & (cc < wth) & (cc >= 0):
            return rr, cc
        else:
            raise ValueError('Coordinates are not inside the image!')

    else:
        return rr, cc


def reproject_coords(epsg_in, epsg_out, coordinates):
    """

    :param epsg_in: int or str - EPSG code of source
    :param epsg_out: int or str - EPSG code of destination
    :param coordinates: (tuple, list, array) - source coordinates
    :return:
    """
    #print epsg_in
    inProj = pyproj.Proj(init='epsg:{e}'.format(e=epsg_in))
    outProj = pyproj.Proj(init='epsg:{e}'.format(e=epsg_out))
    coordinates_transformed = pyproj.transform(inProj, outProj, coordinates[0], coordinates[1])
    return coordinates_transformed


def coord_raster(filepath, epsg_in=4326, epsg_out=4326):
    """
    Function returns coordinates per pixel
    :param filepath: str
    :param epsg_out: int
    :return: longitude, latitude
    """
    with rasterio.open(filepath) as src:
        gt = src.get_transform()
        #epsg_in = int(src.get_crs().values()[0].split(':')[-1])
        coords = np.zeros((src.width, src.height), dtype=np.float)
        yy, xx = np.mgrid[:src.width, :src.width]
        x_coord = (gt[0] + gt[1] * xx).ravel()
        y_coord = (gt[3] + gt[5] * yy).ravel()
        lon, lat = reproject_coords(int(epsg_in), int(epsg_out), (x_coord, y_coord))
        return lon.reshape(src.width, src.height), lat.reshape(src.width, src.height)