import xarray as xr
import numpy as np 
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.spatial import cKDTree as KDTree
import matplotlib
import matplotlib.pyplot as plt 
import xrft
import multiprocessing
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Use 'Agg' backend for matplotlib
matplotlib.use('Agg')

from .utils import shared_array

def spec_3d(da, w0, w1, detrend='constant'):
    """
    Compute the 3D power spectral density (PSD) of the input data array.

    Parameters:
    - da: xarray.DataArray, input data array.
    - w0: float, lower frequency bound.
    - w1: float, upper frequency bound.
    - detrend: str, detrending method.

    Returns:
    - psd: xarray.DataArray, power spectral density within the specified frequency bounds.
    """
    # Rechunk the data array
    chunks = {d: da[d].size for d in da.dims}
    signal = da.chunk(chunks)

    # Compute PSD using xrft
    psd = xrft.power_spectrum(signal, detrend=detrend, window=True).compute()

    # Return PSD within specified frequency bounds
    return psd.where((psd[psd.dims[0]] < w1) & (psd[psd.dims[0]] > w0), drop=True)

def normalize(psd, kx, ky, kmin=0.01, kmax=10, nphi=360, nk=200):
    """
    Normalize the power spectral density (PSD) over a log-polar grid.

    Parameters:
    - psd: numpy.ndarray, input PSD.
    - kx: numpy.ndarray, x-axis frequencies.
    - ky: numpy.ndarray, y-axis frequencies.
    - kmin: float, minimum frequency.
    - kmax: float, maximum frequency.
    - nphi: int, number of angles for the log-polar grid.
    - nk: int, number of radial points for the log-polar grid.

    Returns:
    - psd_norm: numpy.ndarray, normalized PSD.
    """
    # Define log-polar grid
    phi = np.deg2rad(np.linspace(0., 360., num=nphi, endpoint=True))
    alpha = (kmax / kmin) ** (1. / (nk - 1.))
    k = kmin * alpha ** np.arange(nk)
    kx2d, ky2d = np.meshgrid(kx, ky)
    
    # Bilinear interpolation
    psd_norm = np.zeros_like(psd)
    for i in range(psd.shape[0]):
        intfunc = RectBivariateSpline(ky, kx, psd[i], kx=1, ky=1)
        intkx = k[np.newaxis, :] * np.sin(phi[:, np.newaxis])
        intky = k[np.newaxis, :] * np.cos(phi[:, np.newaxis])
        polpsd = np.zeros((nphi, nk), dtype=psd[i].dtype)
        indint = np.where((intkx >= kx[0]) & (intkx <= kx[-1]) & (intky >= ky[0]) & (intky <= ky[-1]))
        polpsd[indint] = intfunc(intkx[indint], intky[indint], grid=False)

        # Normalization
        polpsd = polpsd / polpsd.sum(axis=1)[:, np.newaxis]

        # Polar coordinates
        _xl = k[np.newaxis, :] * np.cos(phi[:, np.newaxis])
        _yl = k[np.newaxis, :] * np.sin(phi[:, np.newaxis])

        # Interpolate to cartesian grid
        psd_norm[i] = griddata((_xl.ravel(), _yl.ravel()), polpsd.ravel(), (kx2d.ravel(), ky2d.ravel()), method='cubic').reshape((kx2d.shape))

    return psd_norm

def interp(X, z, q, nnear=6, eps=0, p=1, weights=None, dmin=None):
    """
    Interpolate data using k-nearest neighbors.

    Parameters:
    - X: numpy.ndarray, input coordinates.
    - z: numpy.ndarray, input values.
    - q: numpy.ndarray, query points.
    - nnear: int, number of nearest neighbors.
    - eps: float, approximation factor.
    - p: int, power parameter for inverse distance weighting.
    - weights: numpy.ndarray, optional weights.
    - dmin: float, minimum distance.

    Returns:
    - interpol: numpy.ndarray, interpolated values at query points.
    """
    tree = KDTree(X)
    q = np.asarray(q)
    qdim = q.ndim
    if qdim == 1:
        q = np.array([q])
    distances, ix = tree.query(q, k=nnear, eps=eps, distance_upper_bound=nnear*dmin)
    interpol = np.zeros((len(distances),) + np.shape(z[0]))
    jinterpol = 0
    for dist, ix in zip(distances, ix):
        ix0 = ix[ix < z.size]
        dist0 = dist[ix < z.size]
        if nnear == 1:
            wz = z[ix0]
        elif dist[0] < 1e-10:
            wz = z[ix0[0]]
        else:  # weight z by 1/dist
            w = 1 / dist0**p
            if weights is not None:
                w *= weights[ix0]  # >= 0
            w /= np.sum(w)
            wz = np.dot(w, z[ix0])
        interpol[jinterpol] = wz
        jinterpol += 1
            
    return interpol if qdim > 1 else interpol[0]

def J(psd, w, kx, ky, k2d, kx2d, ky2d, u, v):
    """
    Compute the cost function J for given velocity components u and v.

    Parameters:
    - psd: xarray.DataArray, power spectral density.
    - w: numpy.ndarray, frequency values.
    - kx: numpy.ndarray, x-axis frequencies.
    - ky: numpy.ndarray, y-axis frequencies.
    - k2d: numpy.ndarray, 2D array of frequency magnitudes.
    - kx2d: numpy.ndarray, 2D array of x-axis frequencies.
    - ky2d: numpy.ndarray, 2D array of y-axis frequencies.
    - u: float, velocity component in x-direction.
    - v: float, velocity component in y-direction.

    Returns:
    - res: float, cost function value.
    """
    g = 9.81
    res = 0
    for iw in range(psd.shape[0]):
        disp_uv = np.sqrt(g * k2d) + kx2d * u + ky2d * v
        cs = plt.contour(kx, ky, disp_uv, [w[iw]])
        plt.close()
        collections = cs.collections[0].get_paths()

        psd1d = psd[iw].values.ravel()
        idx = psd1d > 0.1 * np.nanmax(psd1d)
        psd1d = psd1d[idx]
        coords1d = np.vstack((kx2d.ravel()[idx], ky2d.ravel()[idx])).T
        kxmin = kx2d.ravel()[idx].min()
        kxmax = kx2d.ravel()[idx].max()
        kymin = ky2d.ravel()[idx].min()
        kymax = ky2d.ravel()[idx].max()
        for p1 in collections:
            coor_p1 = p1.vertices
            coor_p1[:, 0][(coor_p1[:, 0] < kxmin) | (coor_p1[:, 0] > kxmax)] = np.nan
            coor_p1[:, 1][(coor_p1[:, 1] < kymin) | (coor_p1[:, 1] > kymax)] = np.nan
            isNaN = np.isnan(coor_p1.sum(axis=1))
            coor_p1 = coor_p1[~isNaN]
            psd_interp = interp(coords1d, psd1d, coor_p1, dmin=kx[1] - kx[0])
            res += psd_interp.sum()
    
    return res

def get_newdu(du, accuracy):
    """
    Compute a new step size for the velocity components.

    Parameters:
    - du: float, current step size.
    - accuracy: float, desired accuracy.

    Returns:
    - new_du: float, new step size.
    """
    return max(du / 10, accuracy)

def compute_uv_bin(ds, ib, jb, bin_y, bin_x, bin_y_step, bin_x_step, umap=None, vmap=None, Jmap=None, ulim=[-2, 2], vlim=[-2, 2], w0=0.5, w1=1.0, du=0.1, dv=0.1, accuracy=0.01, norm=True, Print=False):
    """
    Compute the velocity components u and v for a specific bin of the dataset.

    Parameters:
    - ds: xarray.Dataset, input dataset.
    - ib: int, bin index in y-direction.
    - jb: int, bin index in x-direction.
    - bin_y: numpy.ndarray, y-axis bin centers.
    - bin_x: numpy.ndarray, x-axis bin centers.
    - bin_y_step: float, bin step size in y-direction.
    - bin_x_step: float, bin step size in x-direction.
    - umap: shared_array, optional output array for u-component.
    - vmap: shared_array, optional output array for v-component.
    - Jmap: shared_array, optional output array for cost function values.
    - ulim: list, u-component limits.
    - vlim: list, v-component limits.
    - w0: float, lower frequency bound.
    - w1: float, upper frequency bound.
    - du: float, initial step size for u-component.
    - dv: float, initial step size for v-component.
    - accuracy: float, desired accuracy.
    - norm: bool, whether to normalize the PSD.
    - Print: bool, whether to print intermediate results.

    Returns:
    - None or (u0, v0): computed velocity components.
    """
    # Select data for the current bin
    ds_bin = ds.sel({
        'x': slice(bin_x[jb] - bin_x_step / 2, bin_x[jb] + bin_x_step / 2),
        'y': slice(bin_y[ib] - bin_y_step / 2, bin_y[ib] + bin_y_step / 2)
    }).copy().load()

    # Compute PSD
    da_psd = spec_3d(ds_bin.data, w0, w1)

    # Get space/time frequencies
    w = da_psd.freq_t.values * 2 * np.pi
    kx = da_psd.freq_x.values * 2 * np.pi
    ky = da_psd.freq_y.values * 2 * np.pi
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2d = np.sqrt(kx2d**2 + ky2d**2)

    # Normalize PSD
    psd = da_psd.copy()
    if norm:
        psd.data = normalize(da_psd.values, kx, ky)

    # Initialize velocity components
    u0, v0 = 0, 0
    u_list = np.arange(ulim[0], ulim[1] + du / 2, du)
    v_list = np.arange(vlim[0], vlim[1] + dv / 2, dv)

    # Iteratively adjust step size until desired accuracy is achieved
    while du > accuracy or dv > accuracy:
        J0 = 0
        for u in u_list:
            for v in v_list:
                Jtest = J(psd, w, kx, ky, k2d, kx2d, ky2d, u, v)
                if Print:
                    print('({:.2f}, {:.2f}) : {:.2E} |  ({:.2f}, {:.2f}) : {:.2E})'.format(u, v, Jtest, u0, v0, J0))
                if Jtest > J0:
                    u0, v0 = u, v
                    J0 = Jtest
        
        ulim[0] = u0 - du
        ulim[1] = u0 + du
        vlim[0] = v0 - dv
        vlim[1] = v0 + dv
        du = get_newdu(du, accuracy)
        dv = get_newdu(dv, accuracy)
        u_list = np.arange(ulim[0], ulim[1] + du / 2, du)
        v_list = np.arange(vlim[0], vlim[1] + dv / 2, dv)
    
    # Final pass with refined step sizes
    J0 = 0
    for u in u_list:
        for v in v_list:
            Jtest = J(psd, w, kx, ky, k2d, kx2d, ky2d, u, v)
            if Print:
                print('({:.2f}, {:.2f}) : {:.2E} |  ({:.2f}, {:.2f}) : {:.2E})'.format(u, v, Jtest, u0, v0, J0))
            if Jtest > J0:
                u0, v0 = u, v
                J0 = Jtest

    # Store results
    if Jmap is not None:
        Jmap[ib, jb] = J0

    if umap is not None and vmap is not None:
        umap[ib, jb] = u0
        vmap[ib, jb] = v0
    else:
        return u0, v0

def run_current_estimation(path_in, num_pixels, ulim, vlim, du, dv, accuracy, w0, w1, num_threads, norm, path_out):
    """
    Run the current estimation process on the input dataset.

    Parameters:
    - path_in: str, path to the input dataset.
    - num_pixels: int, number of pixels per bin.
    - ulim: list, u-component limits.
    - vlim: list, v-component limits.
    - du: float, initial step size for u-component.
    - dv: float, initial step size for v-component.
    - accuracy: float, desired accuracy.
    - w0: float, lower frequency bound.
    - w1: float, upper frequency bound.
    - num_threads: int, number of threads for parallel processing.
    - norm: bool, whether to normalize the PSD.
    - path_out: str, path to the output file.

    Returns:
    - None
    """
    # Open input dataset
    ds = xr.open_dataset(path_in).load()
    t = ds.t.values
    x = ds.x.values
    y = ds.y.values
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Create grid for current maps
    bin_x_step = num_pixels * (x[1] - x[0])
    bin_y_step = num_pixels * (y[1] - y[0])
    bin_x = np.arange(0, x.max() - bin_x_step / 2 + dx, bin_x_step / 2)
    bin_x = np.concatenate((-bin_x[::-1], bin_x[1:]))
    bin_y = np.arange(0, y.max() - bin_y_step / 2 + dy, bin_y_step / 2)
    bin_y = np.concatenate((-bin_y[::-1], bin_y[1:]))

    # Initialize mapped u & v    
    umap = shared_array((bin_y.size, bin_x.size))
    vmap = shared_array((bin_y.size, bin_x.size))
    Jmap = shared_array((bin_y.size, bin_x.size))

    # Create jobs for parallel processing
    jobs = []
    for ib in range(bin_y.size):
        for jb in range(bin_x.size):
            jobs.append(multiprocessing.Process(target=compute_uv_bin, args=(ds, ib, jb, bin_y, bin_x, bin_y_step, bin_x_step, umap, vmap, Jmap, ulim, vlim, w0, w1, du, dv, accuracy, norm)))
            if len(jobs) == num_threads:
                # Start jobs
                for job in jobs:
                    job.start()
                # Join jobs
                for job in jobs:
                    job.join()
                jobs = []
    if len(jobs) > 0:
        # Start remaining jobs
        for job in jobs:
            job.start()
        # Join remaining jobs
        for job in jobs:
            job.join()

    # Write output to netCDF file
    dsout = xr.Dataset(
        {
            'u': (('y', 'x'), umap),
            'v': (('y', 'x'), vmap),
            'J': (('y', 'x'), Jmap)
        },
        coords={'x': ('x', bin_x), 'y': ('y', bin_y)}
    )
    dsout.to_netcdf(path_out)
    dsout.close()
