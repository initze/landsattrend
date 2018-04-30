import glob
import os

import fiona
import numpy as np
import pyproj
import rasterio
from osgeo import ogr, osr, gdal
from shapely.geometry import Polygon

from landsattrend.config_study_sites import study_sites


def reproject_coords(epsg_in, epsg_out, coordinates):
    """
    :param epsg_in: int or str - EPSG code of source
    :param epsg_out: int or str - EPSG code of destination
    :param coordinates: (tuple, list, array) - source coordinates
    :return:
    """
    inProj = pyproj.Proj(init='epsg:{e}'.format(e=epsg_in))
    outProj = pyproj.Proj(init='epsg:{e}'.format(e=epsg_out))
    coordinates_transformed = pyproj.transform(inProj, outProj, coordinates[0], coordinates[1])
    return coordinates_transformed

def coord_raster(filepath, epsg_in=4326, epsg_out=4326):
    """
    Function returns coordinates per pixel
    :param filepath: str
    :param epsg_in: int
    :param epsg_out: int
    :return: longitude, latitude
    """
    with rasterio.open(filepath) as src:
        gt = src.get_transform()
        yy, xx = np.mgrid[:src.width, :src.width]
        x_coord = (gt[0] + gt[1] * xx).ravel()
        y_coord = (gt[3] + gt[5] * yy).ravel()
        lon, lat = reproject_coords(int(epsg_in), int(epsg_out), (x_coord, y_coord))
        return lon.reshape(src.width, src.height), lat.reshape(src.width, src.height)

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
        else:
            outgeom = None

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

    with fiona.open(outfile, mode='w', driver='ESRI Shapefile', schema=schema, crs=fiona.crs.from_epsg(epsg)) as source:
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
    Transform global geographic/projected coordinates to local raster coordinates
    :param datadir: string
    :param coordinates: tuple/array/list
    :param force_inside: boolean
    :return:
    """
    flist = glob.glob(os.path.join(datadir, '*.tif'))
    for f_0 in flist:
        try:
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
        except:
            continue

def get_ullr_from_json(json):
    """
    :param json:
    :return:
    """
    crd = np.array(json['geometry']['coordinates'][0][:4])
    ulx, uly, lrx, lry = crd[[0,2]].ravel()
    return ulx, uly, lrx, lry

def align_px_to_ls(startvalue):
    """align pixel locations to standard pixel edges of ls"""
    return startvalue - (startvalue % 30 - 15)