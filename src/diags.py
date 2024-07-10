import glob
import xarray as xr 
import numpy as np 
import matplotlib.pylab as plt 
from matplotlib.image import imread


def plot_current_map(png_files_pattern, resolution_movie, path_current_map):

    # Read one image
    img = imread(glob.glob(png_files_pattern+'*')[0])[::-1,:,:]
    xmin = -resolution_movie * img.shape[1]/2
    xmax = resolution_movie * img.shape[1]/2
    ymin = -resolution_movie * img.shape[0]/2
    ymax = resolution_movie * img.shape[0]/2

    # Open current map
    ds = xr.open_dataset(path_current_map)
    x = ds.x.values
    y = ds.y.values
    u = ds.u.values
    v = ds.v.values
    U = np.sqrt(u**2+v**2)
    a = u/v

    # Plot current velocities on top of one RGB image
    plt.figure(figsize=(10,5))
    plt.imshow(img, extent=[xmin,xmax,ymin,ymax])
    q = plt.quiver(x,y,u,v,U,scale=10,cmap='Blues_r')
    cbar = plt.colorbar(q,shrink=.5)
    cbar.ax.set_title('[m/s]')
    plt.xlabel('[m]',size=12)
    plt.ylabel('[m]',size=12)

    plt.savefig(path_current_map[:-2] + 'png', bbox_inches='tight')
    plt.show()

    