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

    Examples
    --------
    >>> make_iter(1)
    [[1]]

    >>> make_iter(1, 2, 3)
    [[1], [2], [3]]

    >>> make_iter(1, [2], 3)
    [[1], [2], [3]]

    It is intended that a list of arguments be passed to the function for conversion:

    >>> make_iter(*[1, [2], 3])
    [[1], [2], [3]]

    Remember that strings are already iterable!

    >>> make_iter(*[[1, 2, 3], (4, 5, 6), "a"])
    [[1, 2, 3], (4, 5, 6), 'a']
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
        Parameter to load, either string of Python list/number or filename string. Supported filename extensions are
        '.fits', '.fit', '.fts', '.csv', '.txt', '.npy', '.npz', and '.sav'. If the file does not exist, it will assume
        the string is a Python expression.
    wl : float, optional, default = None
        Central line core wavelength to replace 'wl' in strings. Will only replace occurrences in the `parameter`
        variable itself or in files with extension ".csv" or ".txt". When using `wl`, also use 'inf' and 'nan' as
        required.

    Returns
    -------
    value : ndarray or list of floats
        Value of parameter in easily computable format (not string)

    Examples
    --------
    >>> load_parameter("wl + 4.2", wl=7.1)
    11.3

    >>> load_parameter("[wl + 4.2, 5.2 - inf, 5 > 3]", wl=7.1)
    [11.3, -inf, 1.0]

    Filenames are given as follows:

    >>> x = load_parameter("datafile.csv", wl=12.4)

    >>> x = load_parameter("datafile.fits")

    If the file does not exist, the function will assume that the string is a Python expression, possibly leading to an
    error:

    >>> load_parameter("nonexistant.csv")
    TypeError: 'NoneType' object is not subscriptable
    """
    if os.path.exists(parameter):  # If the parameter is a real file

        ext = os.path.splitext(parameter)[1]  # File extension

        if ext.lower() in ['.fits', '.fit', '.fts']:  # Extension suggests FITS file

            # Read data from the primary HDU of the FITS file
            hdul = fits.open(parameter)  # Open with mmap
            value = hdul[0].data.copy()  # Copy out of mmap
            hdul.close()  # Close the file

        elif ext.lower() in ['.csv', '.txt']:  # Extension suggests CSV file

            # Read CSV file (assumes a ',' delimiter)
            if wl is not None:  # If `wl` is specified, try a replacement
                value = str(list(np.loadtxt(parameter, delimiter=',', dtype=object))).replace('\'', '')
                try:
                    value = eval(str(value), {'__builtins__': None},
                                 {'wl': wl, 'inf': float('inf'), 'nan': float('nan')})
                except TypeError:  # Only allowed to process `wl` and `inf` variables for security reasons
                    raise SyntaxError("parameter string contains illegal variables")
                except SyntaxError:
                    raise SyntaxError("parameter string '{}' contains a syntax error".format(parameter))
            else:
                value = np.loadtxt(parameter, delimiter=',', dtype=float)

        elif ext.lower() in ['.npy', '.npz']:  # Extension suggests NumPy array

            value = np.load(parameter)

        elif ext.lower() in ['.sav']:  # Extension suggests IDL SAVE file (assumes relevant data in first variable)

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
