import os

import numpy as np
from astropy.io import fits
from scipy.io import readsav


__all__ = ['make_iter', 'load_parameter']


def make_iter(*args):
    """Returns each inputted argument, wrapping in a list if not already iterable

    Parameters
    ----------
    *args
        Arguments to make iterable.

    Returns
    -------
    iterables
        `*args` converted to iterables.
    """
    iterables = []  # Holder to return
    for parameter in args:
        try:  # Will work if iterable
            _ = (i for i in parameter)
        except TypeError:  # Not iterable
            parameter = [parameter]  # Wrap in a list
        iterables = iterables + [parameter]  # Append to list to return
    return iterables


def load_parameter(parameter, wl=None):
    """Load parameters from file, optionally evaluating variables from strings

    Loads the parameter from string or file.

    Parameters
    ----------
    parameter : str
        Parameter to load, either string of Python list/number or filename string.
    wl : float, optional, default = None
        Central line core wavelength to replace 'wl' in strings.

    Returns
    -------
    value : ndarray or list of floats
        Value of parameter in easily computable format (not string)
    """
    if os.path.exists(parameter):  # If the parameter is a real file

        ext = os.path.splitext(parameter)[1]  # File extension

        if ext.lower() in ['.fits', '.fit', '.fts']:  # Extension suggests FITS file

            # Read data from the primary HDU of the FITS file
            hdul = fits.open(parameter)  # Open with mmap
            value = hdul[0].data.copy()  # Copy out of mmap
            hdul.close()  # Close the file

        elif ext.lower() in ['.csv', '.txt']:  # Extension suggests CSV file

            # Read CSV file
            value = np.loadtxt(parameter, delimiter=',')  # Assumes a delimiter
            if wl is not None:  # If `wl` is specified, try a replacement
                try:
                    value = eval(str(value), {'__builtins__': None}, {'wl': wl, 'inf': float('inf')})
                except TypeError:  # Only allowed to process `wl` and `inf` variables for security reasons
                    raise SyntaxError("parameter string contains illegal variables")
                except SyntaxError:
                    raise SyntaxError("parameter string '{}' contains a syntax error".format(parameter))

        elif ext.lower() in ['.npy', '.npz']:  # Extension suggests NumPy array

            value = np.load(parameter)

        elif ext.lower() in ['.sav']:  # Extension suggests IDL SAVE file

            value = list(readsav(parameter).values())[0]

        else:  # Extension not matched
            raise ValueError("loaded parameters can only have file extensions: '.fits', '.fit', '.fts', '.csv', "
                             "'.txt', '.npy', '.npz', '.sav', got '%s'" % ext.lower())

    else:  # Must not be a file (or the filename is incorrect!)

        # Convert to list, calculate relative to central line core (`wl`)
        value = eval(str(parameter), {'__builtins__': None}, {'wl': wl, 'inf': float('inf')})
        try:
            value = [float(val) for val in value]  # Make sure all values are floats
        except TypeError:
            value = float(value)

    return value
