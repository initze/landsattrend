import numpy as np
from scipy import stats
from scipy.stats import itemfreq
import bottleneck as bn
from numba import jit


def median_slopes(x1, x2, y1, y2, full=False):
    a = (y2-y1)/(x2-x1)
    return np.median(a, axis=0)


def slope(x1, x2, y1, y2, full=False):
    return (y2-y1)/(x2-x1)


def crossindex(data, nodata=0):
    """
    function to create crossindices for valid data of 1-D array
    """
    ind = np.where(data != nodata)[0]
    l = len(ind)
    r = ind[:-1]
    c = ind[1:]
    r = np.repeat(r, l-1).reshape((l-1, l-1))
    c = np.tile(c, l-1).reshape((l-1, l-1))
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
    ifreq = ifreq[ifreq[:,1] > 1]

    # calculate masked mean of duplicate image dates and apply to data
    # collect reduced, non-used indices and remove from data-stack
    idx_del = []
    for ifr in ifreq:
        idx = np.where(dates == ifr[0])[0] # find indices where imdates have at least duplicate values
        d[idx[0]] = d[idx].mean(axis=0)
        idx_del += list(idx[1:])
    valid_idx = np.array(np.setxor1d(list(range(len(dates))), idx_del), dtype=np.uint16) # remove duplicate layers

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
            data[i] = data[[i, i+1]].all(axis=0)
            #gt = np.where(mask[i] > mask[i+1])
            #mask[i][gt] = mask[i+1][gt]
    return np.invert(data[idx]).sum(axis=0)


def theil_sen(x, y, full=False):
    """
    x: imagedates in ordinal dates
    y: values to calculate slope
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # remove duplicate dates
    """
    ind = non_duplicates(x, y) # FIX
    x = x[ind]
    y = y[ind]
    """
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
                local_results.append((0,0,0,0))
        slope_image[:, ind] = np.array(local_results).T

        #slope_image[:, ind] = np.array([theilslopes_ma_local(d, image_dates) for d in image_stack[ind]]).T
        slope_image[[0,2,3]] *= factor

    if ndim == 3:
        slope_image = np.reshape(slope_image, (4, nrows, ncols))

    return slope_image

@jit
def sig2_calc(n, z ):
    return np.sqrt(1/18. * n * (n-1) * (2*n +5)) * z

@jit
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

    nt = len(slopes)       # N in Sen (1968)
    ny = len(y)            # n in Sen (1968)
    # Equation 2.6 in Sen (1968):

    sigmaz = sig2_calc(ny, z)

    Ru = min(int(np.round((nt - sigmaz)/2.)), nt-1)
    Rl = max(int(np.round((nt + sigmaz)/2.)) - 1, 0)

    # bn.partition takes only necessary indices, much faster than a full sort
    delta = [bn.partition(slopes, Rl+1)[Rl], bn.partition(slopes, Ru+1)[Ru]]

    return medslope, medinter, delta[0], delta[1]


@jit
def theilslopes_ma_local(y, x=None, alpha=0.95):
    y = np.ma.asarray(y).flatten()
    if x is None:
        x = np.ma.arange(len(y), dtype=float)
    else:
        x = np.ma.asarray(x).flatten()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths ! (%s<>%s)" % (len(y),len(x)))

    m = np.ma.mask_or(np.ma.getmask(x), np.ma.getmask(y))
    y._mask = x._mask = m
    # Disregard any masked elements of x or y
    y = y.compressed()
    x = x.compressed().astype(float)
    # We now have unmasked arrays so can use `stats.theilslopes`
    return theilslopes_local(y, x)