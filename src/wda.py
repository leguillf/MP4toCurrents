import xarray as xr
import numpy as np 
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.spatial import cKDTree as KDTree
import matplotlib
import matplotlib.pyplot as plt 
import xrft
import multiprocessing
import gc
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio


# Ignore warnings
warnings.filterwarnings("ignore")

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

    g = 9.81 # gravitational acceleration
    res = 0 # Initialize cost function value

    # Iterate over the frequencies
    for iw in range(psd.shape[0]):
        # Compute the dispersion relation
        disp_uv = np.sqrt(g * k2d) - kx2d * u - ky2d * v
        # Find the indices of the frequencies that are close to the dispersion relation
        w0 = w[iw]
        wmin = (w0 + w[iw-1]) / 2 if iw > 0 else 0
        wmax = (w0 + w[iw+1]) / 2 if iw < psd.shape[0] - 1 else np.inf
        indices = (disp_uv>=wmin) & (disp_uv<wmax) 
        # Extract the corresponding coordinates from kx2d and ky2d
        x_contour = kx2d[indices]
        y_contour = ky2d[indices]
        # Extract the corresponding PSD values
        psd1d = psd[iw].values.ravel()
        # Filter the PSD values and coordinates to only include those above a certain threshold
        idx = psd1d > 0.1 * np.nanmax(psd1d) 
        psd1d = psd1d[idx]
        coords1d = np.vstack((kx2d.ravel()[idx], ky2d.ravel()[idx])).T
        # Interpolate the PSD values onto the contour coordinates
        coords = np.vstack((x_contour, y_contour)).T
        psd_interp = interp(coords1d, psd1d, coords, nnear=6, dmin=kx[1] - kx[0])
        # Compute the cost function value
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

def compute_uv_bin(da_psd, c, umap=None, vmap=None, Jmap=None, ulim=[-2, 2], vlim=[-2, 2], w0=0.5, w1=1.0, du=0.1, dv=0.1, accuracy=0.01, norm=True, name_save='', Print=False):
    """
    Compute the velocity components u and v for a specific bin of the dataset.

    Parameters:
    - ds: xarray.Dataset, input dataset.
    - c: int, bin index in y-direction.
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
    
    # Get space/time frequencies
    w = da_psd.freq_t.values * 2 * np.pi
    kx = da_psd.freq_x.values * 2 * np.pi
    ky = da_psd.freq_y.values * 2 * np.pi
    kx2d, ky2d = np.meshgrid(kx, ky)
    k2d = np.sqrt(kx2d**2 + ky2d**2)

    # Normalize PSD
    psd = da_psd.copy().load()
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

    # Plot results
    fig, ax = plt.subplots()
    def update(iw):
        ax.clear()
        w0 = w[iw]
        disp_uv = np.sqrt(9.81 * k2d) - kx2d * u0 - ky2d * v0
        mesh = ax.pcolormesh(kx2d, ky2d, psd[iw], cmap='Reds', shading='auto')

        # Draw contour
        contour = ax.contour(kx2d, ky2d, disp_uv, levels=[w0], colors='black')
        
        # Create dummy line for legend
        dummy_line = plt.Line2D([], [], color='black', label=f'u = {u0:.2f}m/s, v = {v0:.2f}m/s')
        
        ax.legend(handles=[dummy_line], loc='upper right')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('kx (m-1)')
        ax.set_ylabel('ky (m-1)')
        ax.set_title(f'w = {w0/(2*np.pi):.2f} s-1')
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(w), interval=200)
    # Save animation to GIF using Pillow (requires `pillow` package)
    anim.save(f'{name_save}_{u0:.2f}_{v0:.2f}.gif', writer='pillow', fps=5)

    # Store results
    if Jmap is not None:
        Jmap[c] = J0
    
    print('(u = {:.2f}, v = {:.2f}) | J = {:.2E}'.format(u0, v0, J0))

    if umap is not None and vmap is not None:
        umap[c] = u0
        vmap[c] = v0
    else:
        return u0, v0

def run_current_estimation(path_in, num_pixels, num_times, ulim, vlim, du, dv, accuracy, w0, w1, num_threads, norm, dir_out, file_out):
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

    # Create grid for current maps
    if num_pixels is None:
        bin_x_step = x[-1] - x[0]
        bin_y_step = y[-1] - y[0]
    else:
        bin_x_step = (x[1] - x[0]) * num_pixels
        bin_y_step = (y[1] - y[0]) * num_pixels
    if num_times is None:
        bin_t_step = t[-1] - t[0]
    else:
        bin_t_step = (t[1] - t[0]) * num_times

    
    bin_x = np.arange(0, x.max(), bin_x_step / 2)
    bin_x = np.concatenate((-bin_x[::-1], bin_x[1:]))
    bin_y = np.arange(0, y.max(), bin_y_step / 2)
    bin_y = np.concatenate((-bin_y[::-1], bin_y[1:]))
    if bin_x[-1] + bin_x_step / 2 > x.max():
        bin_x[-1] = x.max() - bin_x_step/2
    if bin_y[-1] + bin_y_step / 2 > y.max():
        bin_y[-1] = y.max() - bin_y_step/2
    bin_t = np.arange(bin_t_step / 2, t.max(), bin_t_step / 2)

    # Initialize mapped u & v    
    print('Shape:', (bin_t.size, bin_y.size, bin_x.size))
    umap = shared_array((bin_t.size, bin_y.size, bin_x.size))
    vmap = shared_array((bin_t.size, bin_y.size, bin_x.size))
    Jmap = shared_array((bin_t.size, bin_y.size, bin_x.size))

    # Create jobs for parallel processing
    jobs = []
    c = -1
    for tb in range(bin_t.size):
        for ib in range(bin_y.size):
            for jb in range(bin_x.size):
                c += 1
                # Select data for the current bin
                ds_bin = ds.sel({
                    'x': slice(bin_x[jb] - bin_x_step / 2, bin_x[jb] + bin_x_step / 2),
                    'y': slice(bin_y[ib] - bin_y_step / 2, bin_y[ib] + bin_y_step / 2),
                    't': slice(bin_t[tb] - bin_t_step / 2, bin_t[tb] + bin_t_step / 2)
                }).copy()

                # Compute PSD
                da_psd = spec_3d(ds_bin.data, w0, w1).load()
                jobs.append(multiprocessing.Process(target=compute_uv_bin, args=(da_psd, c, 
                                                                                 umap, vmap, Jmap, 
                                                                                 ulim, vlim, w0, w1, du, dv, accuracy, norm, 
                                                                                 f'{dir_out}/anim_{tb}_{ib}_{jb}')))
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
    
    umap = np.asarray(umap.get_obj()).reshape((bin_t.size, bin_y.size, bin_x.size))
    vmap = np.asarray(vmap.get_obj()).reshape((bin_t.size, bin_y.size, bin_x.size))
    Jmap = np.asarray(Jmap.get_obj()).reshape((bin_t.size, bin_y.size, bin_x.size))

    # Write output to netCDF file
    dsout = xr.Dataset(
        {
            'u': (('t', 'y', 'x'), umap),
            'v': (('t', 'y', 'x'), vmap),
            'J': (('t', 'y', 'x'), Jmap)
        },
        coords={'x': ('x', bin_x), 'y': ('y', bin_y), 't': ('t', bin_t)}
    )
    dsout.to_netcdf(f'{dir_out}/{file_out}')
    dsout.close()
