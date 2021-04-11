import os
from shutil import copyfile

import numpy as np
from astropy.io import fits
from scipy.io import readsav


__all__ = ['make_iter', 'load_parameter', 'merge_results']


def make_iter(*args):
    """Returns each inputted argument, wrapping in a list if not already iterable.

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
    """Load parameters from file, optionally evaluating variables from strings.

    Loads the parameter from string or file.

    Parameters
    ----------
    parameter : str
        Parameter to load, either string of Python list/number or filename string. Supported filename extensions are
        '.fits', '.fit', '.fts', '.csv', '.txt', '.npy', '.npz', and '.sav'. If the file does not exist, it will assume
        the string is a Python expression.
    wl : float, optional, default=None
        Central line core wavelength to replace 'wl' in strings. Will only replace occurrences in the `parameter`
        variable itself or in files with extension ".csv" or ".txt". When using `wl`, also use 'inf' and 'nan' as
        required.

    Returns
    -------
    value : numpy.ndarray or list of floats
        Value of parameter in easily computable format (not string).

    Examples
    --------
    >>> load_parameter("wl + 4.2", wl=7.1)
    11.3

    >>> load_parameter("[wl + 4.2, 5.2 - inf, 5 > 3]", wl=7.1)
    [11.3, -inf, 1.0]

    Filenames are given as follows:

    >>> x = load_parameter("datafile.csv", wl=12.4)  # doctest: +SKIP

    >>> x = load_parameter("datafile.fits")  # doctest: +SKIP

    If the file does not exist, the function will assume that the string is a Python expression, possibly leading to an
    error:

    >>> load_parameter("nonexistant.csv")
    Traceback (most recent call last):
     ...
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


def merge_results(filenames, output):
    """Merges files generated by the :meth:`mcalf.models.FitResults.save` method.

    Parameters
    ----------
    filenames : list of str, length>1
        List of FITS files generated by :meth:`mcalf.models.FitResults.save` method.
    output : str
        Name of FITS file to save merged input files to. Will be clobbered.

    Notes
    -----
    See :meth:`mcalf.models.FitResults` for details on the output FITS file data structure.
    """
    if not isinstance(filenames, list) or len(filenames) <= 1:
        raise TypeError("`filenames` must be a list of length greater than 1.")

    # Verification headers (initialise and give keys)
    verification = {
        'PRIMARY': {
            'NTIME': None,
            'NROWS': None,
            'NCOLS': None,
            'TIME': None,
        },
        'PARAMETERS': {
            'NPARAMS': None,
        },
        'CLASSIFICATIONS': {
        },
        'PROFILE': {
            'PROFILES': None
        },
        'SUCCESS': {
        },
        'CHI2': {
        },
        'VLOSA': {
            'VTYPE': None,
            'UNIT': None,
        },
        'VLOSQ': {
            'VTYPE': None,
            'UNIT': None,
        },
    }

    # Values if not fitted (or unsuccessful)
    unset_value = {
        'PRIMARY': '__SKIP__',
        'PARAMETERS': np.nan,
        'CLASSIFICATIONS': -1,
        'PROFILE': 0,
        'SUCCESS': False,
        'CHI2': np.nan,
        'VLOSA': np.nan,
        'VLOSQ': np.nan,
    }

    # Open the output file for updating
    main_hdul = fits.open(filenames[0], mode='readonly')

    # Record the order for easy access {'NAME': index, ...}
    main_index = {main_hdul[v].name: v for v in range(len(main_hdul))}

    # Remove optional keys if not present in first file
    for optional_key in ['VLOSA', 'VLOSQ']:
        if optional_key not in main_index.keys():
            verification.pop(optional_key)

    # Check that the expected HDUs are present
    if main_index.keys() != verification.keys():
        raise ValueError(f"Unexpected HDU name in {filenames[0]}.")

    # Get expected values for the headers from the first file
    for name in verification.keys():
        for attribute in verification[name].keys():
            verification[name][attribute] = main_hdul[main_index[name]].header[attribute]

    # Load the initial arrays
    arrays = {name: main_hdul[main_index[name]].data.copy() for name in verification.keys()}

    # Close the first input file
    main_hdul.close()

    # Copy across the remainder of the FITS files
    for filename in filenames[1:]:
        with fits.open(filename, mode='readonly') as hdul:

            # Check that the expected HDUs are present in `filename`
            input_index = {hdul[v].name: v for v in range(len(hdul))}
            if input_index.keys() != verification.keys():
                raise ValueError(f"Unexpected HDUs in {filename}.")

            for name in verification.keys():  # Loop through the HDUs

                # Verify that the important header items match
                for attribute, expected_value in verification[name].items():
                    if hdul[input_index[name]].header[attribute] != expected_value:
                        # TODO: Handle the case where there are different profiles in each file
                        raise ValueError(f"FITS attribute {attribute} for {name} HDU in {filename} is different.")

                # Create aliases for the input and output arrays
                output_array = arrays[name]
                input_array = hdul[input_index[name]].data

                # Choose the function to test if data is being overwritten
                invalid = unset_value[name]
                if invalid == '__SKIP__':  # PRIMARY HDU (do nothing)
                    continue
                elif np.isnan(invalid):  # floats (can only overwrite nan)
                    test_function = _nan_test
                elif isinstance(invalid, bool) and not invalid:  # bool (can only overwrite False)
                    test_function = _false_test
                elif isinstance(invalid, (int, np.integer)) and invalid == -1:
                    test_function = _minus_one_test
                elif isinstance(invalid, (int, np.integer)) and invalid == 0:
                    test_function = _zero_test
                else:
                    raise ValueError(f"Unexpected invalid value {invalid}.")

                # Verify that no data is being overwritten
                should_edit = test_function(input_array)
                would_edit = output_array[should_edit]
                if np.sum(test_function(would_edit)) != 0:
                    raise ValueError(f"Overlapping values in {name} HDU at {filename}.")

                # Merge `input_array` onto output
                output_array[np.where(should_edit)] = input_array[np.where(should_edit)]

    # Copy the first FITS input to the output file
    copyfile(filenames[0], output)

    # Open the output file for updating
    with fits.open(output, mode='update') as output_hdul:
        for hdu in output_hdul:
            hdu.data = arrays[hdu.name]


def _nan_test(x):
    """Finds where not NaN.

    False if index is NaN.

    Parameters
    ----------
    x : array_like
        Array to search.

    Returns
    -------
    array : array of bool
        Whether corresponding index is not NaN.
    """
    return ~np.isnan(x)


def _false_test(x):
    """Finds where not False (where is True).

    Parameters
    ----------
    x : array_like
        Array to search.

    Returns
    -------
    array : array of bool
        Whether corresponding index is True. (Is not False.)

    Notes
    -----
    Converts to bool dtype as integer could have been given.
    """
    return x.astype(bool)


def _minus_one_test(x):
    """Finds where not -1.

    Parameters
    ----------
    x : array_like
        Array to search.

    Returns
    -------
    array : array of bool
        Whether corresponding index is not -1.
    """
    return x != -1


def _zero_test(x):
    """Finds where not 0.

    Parameters
    ----------
    x : array_like
        Array to search.

    Returns
    -------
    array : array of bool
        Whether corresponding index is not 0.
    """
    return x != 0
