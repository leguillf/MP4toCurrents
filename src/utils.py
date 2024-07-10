import numpy as np
import multiprocessing
import ctypes

def shared_array(shape):
    """
    Create a shared array that can be accessed by multiple processes.
    
    Parameters:
    shape (tuple): Shape of the desired shared array.
    
    Returns:
    np.ndarray: Numpy array that is shared among multiple processes.
    """
    
    # Create a shared array in memory with the given shape and type double (c_double)
    shared_array_base = multiprocessing.Array(ctypes.c_double, int(np.prod(shape)))
    
    # Convert the shared array to a numpy array
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    
    # Reshape the numpy array to the specified shape
    shared_array = shared_array.reshape(*shape)
    
    return shared_array

def gaspari_cohn(r, c):
    """
    Calculate the Gaspari-Cohn function values for a given radius and cutoff.
    
    Parameters:
    r (np.ndarray): Array of radii.
    c (float): Cutoff distance.
    
    Returns:
    np.ndarray: Values of the Gaspari-Cohn function for the given radii.
    """
    
    # Normalize radius by the cutoff distance
    ra = 2 * np.abs(r) / c
    
    # Initialize the result array with zeros
    gp = np.zeros_like(ra)
    
    # Compute values for the range where 0 <= ra <= 1
    i = np.where(ra <= 1.)[0]
    gp[i] = -0.25 * ra[i]**5 + 0.5 * ra[i]**4 + 0.625 * ra[i]**3 - (5./3.) * ra[i]**2 + 1.
    
    # Compute values for the range where 1 < ra <= 2
    i = np.where((ra > 1.) * (ra <= 2.))[0]
    gp[i] = (1./12.) * ra[i]**5 - 0.5 * ra[i]**4 + 0.625 * ra[i]**3 + (5./3.) * ra[i]**2 - 5. * ra[i] + 4. - (2./3.) / ra[i]
    
    return gp
