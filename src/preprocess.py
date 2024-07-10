import glob, re 
import subprocess
import numpy as np 
import scipy.fftpack as fp
from matplotlib.image import imread
from scipy.signal import convolve2d
import multiprocessing
import os 
import xarray as xr 

from .utils import shared_array, gaspari_cohn


def MP4toPNG(input_file, output_pattern, fps):
    """
    Convert MP4 video to PNG frames.
    
    Parameters:
    input_file (str): Path to the input MP4 file.
    output_pattern (str): Pattern for naming output PNG files.
    fps (int): Frames per second for extracting frames.
    """
    # Extract PNG frames from MP4 using ffmpeg
    command = f"ffmpeg -i {input_file} -vf fps={fps} {output_pattern}_%d.png"
    _ = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
    
    # Rename files to be sorted in alphabetical order
    files = glob.glob(f"{output_pattern}_*.png")
    num0 = len(str(len(files)))  # Number of digits for zero-padding
    for file in files:
        # Get index of frame
        idx = re.search(f'{output_pattern}_(.*).png', file).group(1)
        # Rename by filling with zeros for each index
        file_renamed = f'{output_pattern}_{str(idx).zfill(num0)}.png'
        if file_renamed != file:
            os.system(f'mv {file} {file_renamed}')

def low_pass(data, ground_spacing, cutoff):
    """
    Apply a low-pass filter to data.
    
    Parameters:
    data (np.ndarray): Input 2D data array.
    ground_spacing (float): Ground spacing in the data.
    cutoff (float): Cutoff frequency for the low-pass filter.
    
    Returns:
    np.ndarray: Low-pass filtered data.
    """
    ny, nx = data.shape

    # Make data periodic in space
    data_extended = np.empty((3 * ny, 3 * nx))
    data_extended[ny:2 * ny, nx:2 * nx] = +data
    data_extended[0:ny, nx:2 * nx] = +data[::-1, :]
    data_extended[2 * ny:3 * ny, nx:2 * nx] = +data[::-1, :]
    data_extended[:, 0:nx] = data_extended[:, nx:2 * nx][:, ::-1]
    data_extended[:, 2 * nx:3 * nx] = data_extended[:, nx:2 * nx][:, ::-1]

    # 2D wavenumber
    kx = np.fft.fftfreq(3 * nx, ground_spacing)
    ky = np.fft.fftfreq(3 * ny, ground_spacing)
    k, l = np.meshgrid(kx, ky)
    wavnum2D = np.sqrt(k ** 2 + l ** 2)

    # Kernel
    kernel = np.zeros((3 * ny, 3 * nx))
    kernel[wavnum2D < 1 / cutoff] = 1

    # Spatial window
    winy = np.ones(3 * ny)
    winy[:ny] = gaspari_cohn(np.arange(0, ny, 1), ny)[::-1]
    winy[2 * ny:] = gaspari_cohn(np.arange(0, ny), ny)
    winx = np.ones(3 * nx)
    winx[:nx] = gaspari_cohn(np.arange(0, nx, 1), nx)[::-1]
    winx[2 * nx:] = gaspari_cohn(np.arange(0, nx), nx)
    window = winy[:, np.newaxis] * winx[np.newaxis, :]
    data_win = data_extended * window

    # Fourier transform
    data_hat = fp.fft2(data_win)

    # Filtering
    data_extended_filtered = kernel * data_hat
    data_ls = np.real(fp.ifft2(data_extended_filtered))[ny:2 * ny, nx:2 * nx]

    return data_ls

def process(file, i, band, flag_downscale_movie, downscaling, resolution, cutoff, output_arr):
    """
    Process a single image file.
    
    Parameters:
    file (str): Path to the image file.
    i (int): Index of the image.
    band (int): Color band to process.
    flag_downscale_movie (bool): Whether to downscale the movie.
    downscaling (int): Downscaling factor.
    resolution (float): Resolution of the ground grid.
    cutoff (float): Cutoff frequency for the low-pass filter.
    output_arr (multiprocessing.Array): Shared array to store the output.
    """
    
    img = imread(file)[::-1, :, band]  # Reverse y-axis to make it ascendant

    # Downscaling
    if flag_downscale_movie:
        img_down = convolve2d(img, np.ones((downscaling, downscaling)) / downscaling ** 2)
        img_down = img_down[::downscaling, ::downscaling]
    else:
        img_down = +img

    # Glint Normalization
    img_ls = low_pass(img_down, resolution, cutoff)  # Extract glint
    img_corr = (img_down - img_ls) / img_ls  # Normalization

    # Fill output array
    output_arr[i, :, :] = img_corr[1:-1, 1:-1]

def run_preprocess(files_pattern, band, flag_downscale_movie, downscaling, resolution_movie, num_threads, fps, dir_out):
    """
    Run preprocessing on a set of image files.
    
    Parameters:
    files_pattern (str): Pattern for input files.
    band (int): Color band to process.
    flag_downscale_movie (bool): Whether to downscale the movie.
    downscaling (int): Downscaling factor.
    resolution_movie (float): Resolution of the ground grid.
    num_threads (int): Number of threads for parallel processing.
    fps (int): Frames per second for the output.
    dir_out (str): Output directory for the processed data.
    """
    
    files_in = sorted(glob.glob(f"{files_pattern}_*.png"))
    nt = len(files_in)

    # Define ground grid
    file = files_in[0]
    img = imread(file)[::-1, :, band]  # Reverse y-axis to make it ascendant
    if flag_downscale_movie:
        img_down = convolve2d(img, np.ones((downscaling, downscaling)) / downscaling ** 2)
        img_down = img_down[::downscaling, ::downscaling]
    else:
        img_down = +img
    y_img = np.arange(-downscaling * img_down.shape[0] / 2, downscaling * img_down.shape[0] / 2, downscaling)
    x_img = np.arange(-downscaling * img_down.shape[1] / 2, downscaling * img_down.shape[1] / 2, downscaling)
    x_grnd = resolution_movie * x_img[1:-1]
    y_grnd = resolution_movie * y_img[1:-1]
    resolution = x_grnd[1] - x_grnd[0]
    Ny, Nx = 1, 1
    while 2 * Ny <= y_grnd.size:
        Ny *= 2
    while 2 * Nx <= x_grnd.size:
        Nx *= 2
    cutoff = min(Ny, Nx) * resolution / 2
    output_arr = shared_array([nt, y_grnd.size, x_grnd.size])

    # Run preprocessing
    jobs = []
    for i, file in enumerate(files_in):
        jobs.append(multiprocessing.Process(target=process, args=(file, i, band, flag_downscale_movie, downscaling, resolution, cutoff, output_arr)))
        if len(jobs) == num_threads:
            # Start jobs
            for job in jobs:
                job.start()
            # Join jobs
            for job in jobs:
                job.join()
            jobs = []
    if len(jobs) > 0:
        # Start jobs
        for job in jobs:
            job.start()
        # Join jobs
        for job in jobs:
            job.join()

    # Remove time mean
    mean_arr = output_arr.mean(axis=0)
    output_arr -= mean_arr
    
    # Create and save dataset
    ds = xr.Dataset({'data': (('t', 'y', 'x'), output_arr)}, coords={'t': ('t', np.arange(0, nt / fps, 1 / fps)), 'y': ('y', y_grnd), 'x': ('x', x_grnd)})
    ds.to_netcdf(os.path.join(dir_out, 'data.nc'))
