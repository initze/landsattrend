# -*- coding: utf-8 -*-
import numpy as np
from xml.dom import minidom

#########################################################################
# constants #############################################################
#########################################################################

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

# define metadata items for output products
metadata_trends = {'br':{'DS_INFO': 'Slope of Landsat Tasseled Cap Brightness'},
                   'gr':{'DS_INFO': 'Slope of Landsat Tasseled Cap Greenness'},
                   'we':{'DS_INFO': 'Slope of Landsat Tasseled Cap Wetness'},
                   'ndvi':{'DS_INFO': 'Slope of NDVI'},
                   'ndwi':{'DS_INFO': 'Slope of NDWI'},
                   'ndsi':{'DS_INFO': 'Slope of NDMI'}
                   }


#definition of optical bands of each sensor
bands = {'TM':[1,2,3,4,5,7], 'ETM':[1,2,3,4,5,7], 'OLI_TIRS':[2,3,4,5,6,7]}

#########################################################################
# data import and organisation ##########################################
#########################################################################

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


#########################################################################
# band conversions ######################################################
#########################################################################

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