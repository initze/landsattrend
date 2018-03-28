import os
import csv
import glob
import subprocess
import shutil
import fiona
import joblib
import rasterio
import datetime
import geopandas
import pandas as pd
import netCDF4

from osgeo import gdal
from osgeo import gdal_array as ga

from . import lstools
from .trend_funcs import *
from .config_study_sites import study_sites
from .spatial_funcs import geom_sr_from_bbox


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


def load_data(path, indices = ['tc', 'ndvi', 'ndwi', 'ndmi'], xoff=0, yoff=0, xsize=None, ysize=None, factor = 1.,
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
        tc = [lstools.tasseled_cap(i, s) for i, s in zip(ds, s_l)]
        tc = np.ma.masked_equal(tc, 0)
        ind.append(tc)

    if 'ndvi' in indices:
        ndvi = [lstools.ndvi(i, 3, 2) for i, s in zip(ds, s_l)]
        ndvi = np.ma.masked_equal(ndvi, 0)
        ind.append(ndvi)

    if 'ndwi' in indices:
        ndwi = [lstools.ndvi(i, 1, 3) for i, s in zip(ds, s_l)]
        ndwi = np.ma.masked_equal(ndwi, 0)
        ind.append(ndwi)

    if 'ndmi' in indices:
        ndmi = [lstools.ndvi(i, 3, 4) for i, s in zip(ds, s_l)]
        ndmi = np.ma.masked_equal(ndmi, 0)
        ind.append(ndmi)

    if 'ndbr' in indices:
        ndbr = [lstools.ndvi(i, 3, 5) for i, s in zip(ds, s_l)]
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
    def __init__(self, infolder, dst_file, valid_val=[0, 1], dst_nodata=0, cleanup=True, processing_type='sr'):
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
        self.sensor = lstools.get_sensor_from_metadata(self.metadata)
        self.bands = lstools.bands[self.sensor]

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

def get_datafolder(study_site, coords):
    """

    :param study_site:
    :param coords:
    :return:geom_sr_from_point
    """
    epsg = study_sites[study_site]['epsg']
    geom_pt, sr_pt = (coords[0], coords[1], epsg)
    row = None
    with fiona.open(study_sites[study_site]['fishnet_file']) as src:
        for s in src:
            ulx, uly, lrx, lry = get_ullr_from_json(s)
            geom_bbox, osr_bbox = geom_sr_from_bbox(ulx, uly, lrx, lry, epsg)
            if geom_bbox.Intersects(geom_pt):
                row, path = (s['properties']['row'], s['properties']['path'])
                break
    if row != None:
        pdir = study_sites[study_site]['processing_dir'] + r'\{0}_{1}_{2}'.format(study_site, row, path)
    else:
        pass
    return pdir


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