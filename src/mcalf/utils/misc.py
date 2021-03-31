import os
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units
from scipy.io import readsav


__all__ = ['make_iter', 'load_parameter', 'merge_results', 'hide_existing_labels', 'calculate_axis_extent',
           'calculate_extent']


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


def hide_existing_labels(plot_settings, axes=None, fig=None):
    """Hides labels for each dictionary provided if label already exists in legend.

    Parameters
    ----------
    plot_settings : dict of {str: dict}
        Dictionary of lines to be plotted. Values must be dictionaries with a 'label'
        entry that this function my append with a '_' to hide the label.
    axes : list of matplotlib.axes.Axes, optional, default=None
        List of axes to extract lines labels from. Extracts axes from `fig` if omitted.
    fig : matplotlib.figure.Figure, optional, default=None
        Figure to take line labels from. Uses current figure if omitted.

    Notes
    -----
    Only the ``plot_settings[*]['label']`` values are uses to assess if a label has already
    been used. Other `plot_settings` parameters such as `color` are ignored.

    Examples
    --------

    Import plotting package:

    >>> import matplotlib.pyplot as plt

    Define various plot settings:

    >>> plot_settings = {
    ...     'LineA': {'color': 'r', 'label': 'A'},
    ...     'LineB': {'color': 'g', 'label': 'B'},
    ...     'LineC': {'color': 'b', 'label': 'C'},
    ... }

    Create a figure and plot two lines on the first axes:

    >>> fig, axes = plt.subplots(1, 2)
    >>> axes[0].plot([0, 1], [0, 1], **plot_settings['LineA'])  # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> axes[0].plot([0, 1], [1, 0], **plot_settings['LineB'])  # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    Set labels already used to be hidden if used again:

    >>> hide_existing_labels(plot_settings)

    Anything already used will have an underscore prepended:

    >>> [x['label'] for x in plot_settings.values()]
    ['_A', '_B', 'C']

    Plot two lines on the second axes:

    >>> axes[1].plot([0, 1], [0, 1], **plot_settings['LineB'])  # Label hidden  # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> axes[1].plot([0, 1], [1, 0], **plot_settings['LineC'])  # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    Show the figure with the legend:

    >>> fig.legend(ncol=3, loc='upper center')  # doctest: +ELLIPSIS
    <matplotlib.legend.Legend object at 0x...>
    >>> plt.show()
    >>> plt.close()
    """
    # Get axes:
    if axes is None:
        if fig is None:
            fig = plt.gcf()
        axes = fig.get_axes()

    # Get plotted labels:
    lines = []
    for ax in axes:
        lines.extend(ax.get_lines())
    existing = [line.get_label() for line in lines]

    # Hide labels already plotted:
    for name in plot_settings:
        if plot_settings[name]['label'] in existing:
            plot_settings[name]['label'] = '_' + plot_settings[name]['label']


def calculate_axis_extent(resolution, px, offset=0, unit="Mm"):
    """Calculate the extent from a resolution value along a particular axis.

    Parameters
    ----------
    resolution : float or astropy.units.quantity.Quantity
        Length of each pixel. Unit defaults to `unit` is not an astropy quantity.
    px : int
        Number of pixels extent is being calculated for.
    offset : int or float, default=0
        Number of pixels from the 0 pixel to the first pixel. Defaults to the first
        pixel being at 0 length units. For example, in a 1000 pixel wide dataset,
        setting offset to -500 would place the 0 Mm location at the centre.
    unit : str, default="Mm"
        Default unit string to use if `res` is not an astropy quantity.

    Returns
    -------
    first : float
        First extent value.
    last : float
        Last extent value.
    unit : str
        Unit of extent values.
    """

    # Ensure a valid spatial and pixel resolution is provided
    if not isinstance(resolution, (float, astropy.units.quantity.Quantity)):
        raise TypeError('`resolution` values must be either floats or astropy quantities'
                        f', got {type(resolution)}.')
    if not isinstance(px, (int, np.integer)):
        raise TypeError(f'`px` must be an integer, got {type(px)}.')
    if not isinstance(offset, (float, int, np.integer)):
        raise TypeError(f'`offset` must be an float or integer, got {type(offset)}.')

    # Update the default unit if a quantity is provided
    if isinstance(resolution, astropy.units.quantity.Quantity):
        unit = resolution.unit.to_string(astropy.units.format.LatexInline)
        resolution = float(resolution.value)  # Remove the unit

    # Calculate the extent values
    first = offset * resolution
    last = (px + offset) * resolution

    return first, last, unit


def calculate_extent(shape, resolution, offset=(0, 0), ax=None, dimension=None, **kwargs):
    """Calculate the extent from a particular data shape and resolution.

    This function assumes a lower origin is being used with matplotlib.

    Parameters
    ----------
    shape : tuple[int]
        Shape (y, x) of the :class:`numpy.ndarray` of the data being plotted.
        First integer corresponds to the y-axis and the second integer is for the x-axis.
    resolution : tuple[float] or astropy.units.quantity.Quantity
        A 2-tuple (x, y) containing the length of each pixel in the x and y direction respectively.
        If a value has type :class:`astropy.units.quantity.Quantity`, its axis label will
        include its attached unit, otherwise the unit will default to Mm.
        The `ax` parameter must be specified to set its labels.
        If `resolution` is None, this function will immediately return None.
    offset : tuple[float] or int, length=2, optional, default=(0, 0)
        Two offset values (x, y) for the x and y axis respectively.
        Number of pixels from the 0 pixel to the first pixel. Defaults to the first
        pixel being at 0 length units. For example, in a 1000 pixel wide dataset,
        setting offset to -500 would place the 0 Mm location at the centre.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which axis labels will be plotted.
        Defaults to not printing axis labels.
    dimension : str or tuple[str] or list[str], length=2, optional, default=None
        If an `ax` (and `resolution`) is provided, use this string as the `dimension name`
        that appears before the ``(unit)`` in the axis label.
        A 2-tuple (x, y) or list [x, y] can instead be given to provide a different name
        for the x-axis and y-axis respectively.
        Defaults is equivalent to ``dimension=('x-axis', 'y-axis')``.
    **kwargs : dict, optional
        Extra keyword arguments to pass to :func:`calculate_axis_extent`.

    Returns
    -------
    extent : tuple[float], length=4
        The extent value that will be passed to matplotlib functions with a lower origin.
        Will return None if `resolution` is None.
    """
    # Calculate a specific extent if a resolution is specified
    if resolution is not None:

        # Validate relevant parameters
        for n, v in (('shape', shape), ('resolution', resolution), ('offset', offset)):
            if not isinstance(v, tuple) or len(v) != 2:
                raise TypeError(f'`{n}` must be a tuple of length 2.')

        # Calculate extent values, and extract units
        ypx, xpx = shape
        l, r, x_unit = calculate_axis_extent(resolution[0], xpx, offset=offset[0], **kwargs)
        b, t, y_unit = calculate_axis_extent(resolution[1], ypx, offset=offset[1], **kwargs)

        # Optionally set the axis labels
        if ax is not None:

            # Extract the dimension name
            if isinstance(dimension, (tuple, list)):  # different value for each dimension
                if len(dimension) != 2:
                    raise TypeError('`dimension` must be a tuple or list of length 2.')
                x_dim = str(dimension[0])
                y_dim = str(dimension[1])
            elif dimension is None:  # default values
                x_dim, y_dim = 'x-axis', 'y-axis'
            else:  # single value for both dimensions
                x_dim = y_dim = str(dimension)
            ax.set_xlabel(f'{x_dim} ({x_unit})')
            ax.set_ylabel(f'{y_dim} ({y_unit})')

        return l, r, b, t  # extent

    return None  # default extent
