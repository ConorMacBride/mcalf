import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units


__all__ = ['hide_existing_labels', 'calculate_axis_extent', 'calculate_extent', 'class_cmap']


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
        Default unit string to use if `resolution` is not an astropy quantity.

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
            elif isinstance(dimension, str):  # single value for both dimensions
                x_dim = y_dim = str(dimension)
            else:
                raise TypeError('`dimension` must be a tuple or list of length 2.')
            ax.set_xlabel(f'{x_dim} ({x_unit})')
            ax.set_ylabel(f'{y_dim} ({y_unit})')

        return l, r, b, t  # extent

    return None  # default extent


def class_cmap(style, n):
    """Create a listed colormap for a specific number of classifications.

    Parameters
    ----------
    style : str
        The named matplotlib colormap to extract a :class:`~matplotlib.colors.ListedColormap`
        from. Colours are selected from `vmin` to `vmax` at equidistant values
        in the range [0, 1]. The :class:`~matplotlib.colors.ListedColormap`
        produced will also show bad classifications and classifications
        out of range in grey.
        The 'original' style is a special case used since early versions
        of this code. It is a hardcoded list of 5 colours. When the number
        of classifications exceeds 5, ``style='viridis'`` will be used.
    n : int
        Number of colours (i.e., number of classifications) to include in
        the colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap generated for classifications.
    """

    # Validate `n`
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f'`n` must be an integer, got {type(n)}.')

    # Choose colours
    if style == 'original' and n <= 5:  # original colours
        cmap_colors = np.array(['#0072b2', '#56b4e9', '#009e73', '#e69f00', '#d55e00'])[:n]
    else:
        if style == 'original':
            style = 'viridis'  # fallback for >5 classifications
        c = mpl.cm.get_cmap(style)  # query in equal intervals from [0, 1]
        cmap_colors = np.array([c(i / (n - 1)) for i in range(n)])

    # Generate colormap
    cmap = mpl.colors.ListedColormap(cmap_colors)
    cmap.set_over(color='#999999', alpha=1)
    cmap.set_under(color='#999999', alpha=1)

    return cmap
