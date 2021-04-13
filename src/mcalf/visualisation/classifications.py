import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mcalf.utils.smooth import mask_classifications
from mcalf.utils.plot import calculate_extent, class_cmap


__all__ = ['plot_classifications', 'bar', 'plot_class_map', 'init_class_data']


def plot_classifications(spectra, labels, nrows=None, ncols=None, nlines=20, style='original', cmap=None,
                         show_labels=True, plot_settings={}, fig=None):
    """Plot spectra grouped by their labelled classification.

    Parameters
    ----------
    spectra : ndarray, ndim=2
        Two-dimensional array with dimensions [spectra, wavelengths].
    labels : ndarray, ndim=1, length of `spectra`
        List of classifications for each spectrum in `spectra`.
    nrows : int, optional, default=None
        Number of rows. Defaults to rows of max width 3 axes.
        Special case: four plots will be in a 2x2 grid.
        Only one of `nrows` and `ncols` can be specified.
    ncols : int, optional, default=None
        Number of columns. Defaults to rows of max width 3 axes.
        Special case: four plots will be in a 2x2 grid.
        Only one of `nrows` and `ncols` can be specified.
    nlines : int, optional, default=20
        Maximum number of lines per classification plot.
    style : str, optional, default='original'
        The named matplotlib colormap to extract a :class:`~matplotlib.colors.ListedColormap`
        from. Colours are selected from `vmin` to `vmax` at equidistant values
        in the range [0, 1]. The :class:`~matplotlib.colors.ListedColormap`
        produced will also show bad classifications and classifications
        out of range in grey.
        The default 'original' is a special case used since early versions
        of this code. It is a hardcoded list of 5 colours. When the number
        of classifications exceeds 5, ``style='viridis'`` will be used.
    cmap : callable, optional, default=None
        Function that returns a colour for each input from zero to num. classifications.
        This parameter overrides any cmap requested via the `style` parameter.
        Return value is passed to the `color` parameter of
        :func:`matplotlib.pyplot.axes.Axes.plot`.
    show_labels : bool, optional, default=True
        Whether to label the axes with the corresponding classifications.
    plot_settings : dict, optional, default={}
        Dictionary of keyword arguments to pass to :func:`matplotlib.pyplot.axes.Axes.plot`.
    fig : matplotlib.figure.Figure, optional, default=None
        Figure into which the classifications will be plotted.
        Defaults to the current figure.

    Returns
    -------
    gs : matplotlib.gridspec.GridSpec
        The grid layout subplots are placed on within the figure.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.plot_classifications
    """
    if fig is None:
        fig = plt.gcf()

    # Validate parameters

    for n, v in (('spectra', spectra), ('labels', labels)):
        if not isinstance(v, np.ndarray):
            raise TypeError(f'`{n}` must be a numpy.ndarray, got {type(v)}.')

    if not spectra.ndim == 2:
        raise TypeError('`spectra` must be a 2D array.')

    if not labels.ndim == 1 or not issubclass(labels.dtype.type, np.integer):
        raise TypeError('`labels` must be a 1D array of integers.')

    if len(spectra) != len(labels):
        raise ValueError('`spectra` and `labels` must be the same length along the first dimension.')

    if nrows is not None and ncols is not None:
        raise ValueError('Both `nrows` and `ncols` cannot be given together.')

    for n, v in (('nrows', nrows), ('ncols', ncols)):
        if v is not None and not isinstance(v, (int, np.integer)):
            raise TypeError(f'`{n}` must be an integer, got {type(v)}.')

    if not isinstance(nlines, (int, np.integer)) or nlines <= 0:
        raise TypeError('`nlines` must be a positive integer.')

    # Find and count unique classifications
    classifications = np.unique(labels)
    n = len(classifications)  # number of subplots
    if n == 0:
        return None

    # Set `nrows` and `ncols`
    if ncols is None and nrows is None:
        if n == 1:
            ncols = 1
        elif n == 2 or n == 4:
            ncols = 2
        else:
            ncols = 3
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))

    # Verify `nrows` and `ncols`
    if (nrows - 1) * ncols >= n:
        raise ValueError('`nrows` is larger than it needs to be.')
    if nrows * (ncols - 1) >= n:
        raise ValueError('`ncols` is larger than it needs to be.')

    gs = GridSpec(nrows, ncols, figure=fig, wspace=0)

    # Configure the color map
    if cmap is None:
        cmap = class_cmap(style, n)

    for i in range(n):

        ax = fig.add_subplot(gs[i])

        c = classifications[i]
        lines = spectra[labels == c]
        if len(lines) > nlines:  # crop if too big
            lines = lines[:nlines]

        # Whether to crop the y-axis to [0, 1]
        limit_y = False
        if np.nanmin(lines) > -1e-6 and np.nanmax(lines) < 1 + 1e-6:
            limit_y = True

        color = cmap(i)  # extract the single color from the listed colormap
        for l in lines:
            ax.plot(l, color=color, **plot_settings)

        if limit_y:  # if data within range [0, 1]
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])  # only show that intensity is scaled [0, 1]

        ax.set_xticks([])  # no wavelengths plotted
        ax.margins(0)  # no "gaps" at line ends

        for loc, spine in ax.spines.items():
            if loc != 'left':
                spine.set_color('none')  # don't draw spine

        if show_labels:
            ax.set_title(f'classification {str(c)}')

    return gs


def bar(class_map=None, vmin=None, vmax=None, reduce=True, style='original', cmap=None, ax=None, data=None):
    """Plot a bar chart of the classification abundances.

    Parameters
    ----------
    class_map : numpy.ndarray[int], ndim=2 or 3
        Array of classifications. If the array is three-dimensional, it is assumed
        that the first dimension is time, and a time average classification will be plotted.
        The time average is the most common positive (valid) classification at each pixel.
    vmin : int, optional, default=None
        Minimum classification integer to plot. Must be greater or equal to zero.
        Defaults to min positive integer in `class_map`.
    vmax : int, optional, default=None
        Maximum classification integer to plot. Must be greater than zero.
        Defaults to max positive integer in `class_map`.
    reduce : bool, optional, default=True
        Whether to perform the time average described in `class_map` info.
    style : str, optional, default='original'
        The named matplotlib colormap to extract a :class:`~matplotlib.colors.ListedColormap`
        from. Colours are selected from `vmin` to `vmax` at equidistant values
        in the range [0, 1]. The :class:`~matplotlib.colors.ListedColormap`
        produced will also show bad classifications and classifications
        out of range in grey.
        The default 'original' is a special case used since early versions
        of this code. It is a hardcoded list of 5 colours. When the number
        of classifications exceeds 5, ``style='viridis'`` will be used.
    cmap : str or matplotlib.colors.Colormap, optional, default=None
        Parameter to pass to matplotlib.axes.Axes.imshow. This parameter
        overrides any cmap requested via the `style` parameter.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which the velocity map will be plotted.
        Defaults to the current axis of the current figure.
    data : dict, optional, default=None
        Dictionary of common classification plotting settings generated by
        :func:`init_class_data`. If present, all other parameters are ignored
        except and `ax`.

    Returns
    -------
    b : matplotlib.container.BarContainer
        The object returned by :func:`matplotlib.axes.Axes.bar` after plotting abundances.

    See Also
    --------
    mcalf.models.ModelBase.classify_spectra : Classify spectra.
    mcalf.utils.smooth.average_classification : Average a 3D array of classifications.

    Notes
    -----
    Visualisation assumes that all integers between `vmin` and `vmax` are valid
    classifications, even if they do not appear in `class_map`.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.bar
    """
    if ax is None:
        ax = plt.gca()

    if data is None:
        if class_map is None:  # `class_map` must always be provided
            raise TypeError("bar() missing 1 required positional argument: 'class_map'")
        data = init_class_data(class_map, vmin=vmin, vmax=vmax, reduce=reduce, style=style, cmap=cmap)

    # Count for each classification
    d = data['class_map'].flatten()
    counts = np.array([len(d[d == i]) for i in data['classes']])
    d = counts / len(d) * 100  # Convert to percentage

    b = ax.bar(data['classes'], d, color=data['cmap'](np.arange(len(data['classes']))))

    ax.set(xlabel='classification', ylabel='abundance (%)',
           xticks=data['classes'], xticklabels=data['classes'])

    return b


def plot_class_map(class_map=None, vmin=None, vmax=None, resolution=None, offset=(0, 0), dimension='distance',
                   style='original', cmap=None, show_colorbar=True, colorbar_settings=None, ax=None, data=None):
    """Plot a map of the classifications.

    Parameters
    ----------
    class_map : numpy.ndarray[int], ndim=2 or 3
        Array of classifications. If the array is three-dimensional, it is assumed
        that the first dimension is time, and a time average classification will be plotted.
        The time average is the most common positive (valid) classification at each pixel.
    vmin : int, optional, default=None
        Minimum classification integer to plot. Must be greater or equal to zero.
        Defaults to min positive integer in `class_map`.
    vmax : int, optional, default=None
        Maximum classification integer to plot. Must be greater than zero.
        Defaults to max positive integer in `class_map`.
    resolution : tuple[float] or astropy.units.quantity.Quantity, optional, default=None
        A 2-tuple (x, y) containing the length of each pixel in the x and y direction respectively.
        If a value has type :class:`astropy.units.quantity.Quantity`, its axis label will
        include its attached unit, otherwise the unit will default to Mm.
        If `resolution` is None, both axes will be ticked with the default pixel value
        with no axis labels.
    offset : tuple[float] or int, length=2, optional, default=(0, 0)
        Two offset values (x, y) for the x and y axis respectively.
        Number of pixels from the 0 pixel to the first pixel. Defaults to the first
        pixel being at 0 length units. For example, in a 1000 pixel wide dataset,
        setting offset to -500 would place the 0 Mm location at the centre.
    dimension : str or tuple[str] or list[str], length=2, optional, default='distance'
        If an `ax` (and `resolution`) is provided, use this string as the `dimension name`
        that appears before the ``(unit)`` in the axis label.
        A 2-tuple (x, y) or list [x, y] can instead be given to provide a different name
        for the x-axis and y-axis respectively.
    style : str, optional, default='original'
        The named matplotlib colormap to extract a :class:`~matplotlib.colors.ListedColormap`
        from. Colours are selected from `vmin` to `vmax` at equidistant values
        in the range [0, 1]. The :class:`~matplotlib.colors.ListedColormap`
        produced will also show bad classifications and classifications
        out of range in grey.
        The default 'original' is a special case used since early versions
        of this code. It is a hardcoded list of 5 colours. When the number
        of classifications exceeds 5, ``style='viridis'`` will be used.
    cmap : str or matplotlib.colors.Colormap, optional, default=None
        Parameter to pass to matplotlib.axes.Axes.imshow. This parameter
        overrides any cmap requested via the `style` parameter.
    show_colorbar : bool, optional, default=True
        Whether to draw a colorbar.
    colorbar_settings : dict, optional, default=None
        Dictionary of keyword arguments to pass to :func:`matplotlib.figure.Figure.colorbar`.
        Ignored if `show_colorbar` is False.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which the velocity map will be plotted.
        Defaults to the current axis of the current figure.
    data : dict, optional, default=None
        Dictionary of common classification plotting settings generated by
        :func:`init_class_data`. If present, all other parameters are ignored
        except `show_colorbar` and `ax`.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The object returned by :func:`matplotlib.axes.Axes.imshow` after plotting `class_map`.

    See Also
    --------
    mcalf.models.ModelBase.classify_spectra : Classify spectra.
    mcalf.utils.smooth.average_classification : Average a 3D array of classifications.

    Notes
    -----
    Visualisation assumes that all integers between `vmin` and `vmax` are valid
    classifications, even if they do not appear in `class_map`.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.plot_class_map
    """
    if ax is None:
        ax = plt.gca()

    if data is None:
        if class_map is None:  # `class_map` must always be provided
            raise TypeError("plot_class_map() missing 1 required positional argument: 'class_map'")
        data = init_class_data(class_map, vmin=vmin, vmax=vmax,
                               resolution=resolution, offset=offset, dimension=dimension,
                               style=style, cmap=cmap, colorbar_settings=colorbar_settings, ax=ax)

    # Plot `class_map` (using special imshow `data` keyword)
    im = ax.imshow('class_map', cmap='cmap', vmin='plot_vmin', vmax='plot_vmax',
                   origin='lower', extent='extent', interpolation='nearest',
                   data=data)

    if show_colorbar:
        ax.get_figure().colorbar(im, **data['colorbar_settings'])

    return im


def init_class_data(class_map, vmin=None, vmax=None, reduce=True, resolution=None, offset=(0, 0),
                    dimension='distance', style='original', cmap=None, colorbar_settings=None, ax=None):
    """Initialise dictionary of common classification plotting data.

    Parameters
    ----------
    class_map : numpy.ndarray[int], ndim=2 or 3
        Array of classifications. If `reduce` is True (default) and the array is
        three-dimensional, it is assumed that the first dimension is time, and
        a time average classification will be calculated. The time average is
        the most common positive (valid) classification at each pixel.
    vmin : int, optional, default=None
        Minimum classification integer to include. Must be greater or equal to zero.
        Defaults to min positive integer in `class_map`. Classifications below this
        value will be set to -1.
    vmax : int, optional, default=None
        Maximum classification integer to include. Must be greater than zero.
        Defaults to max positive integer in `class_map`. Classifications above this
        value will be set to -1.
    reduce : bool, optional, default=True
        Whether to perform the time average described in `class_map` info.
    resolution : tuple[float] or astropy.units.quantity.Quantity, optional, default=None
        A 2-tuple (x, y) containing the length of each pixel in the x and y direction respectively.
        If a value has type :class:`astropy.units.quantity.Quantity`, its axis label will
        include its attached unit, otherwise the unit will default to Mm.
        If `resolution` is None, both axes will be ticked with the default pixel value
        with no axis labels.
    offset : tuple[float] or int, length=2, optional, default=(0, 0)
        Two offset values (x, y) for the x and y axis respectively.
        Number of pixels from the 0 pixel to the first pixel. Defaults to the first
        pixel being at 0 length units. For example, in a 1000 pixel wide dataset,
        setting offset to -500 would place the 0 Mm location at the centre.
    dimension : str or tuple[str] or list[str], length=2, optional, default='distance'
        If an `ax` (and `resolution`) is provided, use this string as the `dimension name`
        that appears before the ``(unit)`` in the axis label.
        A 2-tuple (x, y) or list [x, y] can instead be given to provide a different name
        for the x-axis and y-axis respectively.
    style : str, optional, default='original'
        The named matplotlib colormap to extract a :class:`~matplotlib.colors.ListedColormap`
        from. Colours are selected from `vmin` to `vmax` at equidistant values
        in the range [0, 1]. The :class:`~matplotlib.colors.ListedColormap`
        produced will also show bad classifications and classifications
        out of range in grey.
        The default 'original' is a special case used since early versions
        of this code. It is a hardcoded list of 5 colours. When the number
        of classifications exceeds 5, ``style='viridis'`` will be used.
    cmap : str or matplotlib.colors.Colormap, optional, default=None
        Parameter to pass to matplotlib.axes.Axes.imshow. This parameter
        overrides any cmap requested via the `style` parameter.
    colorbar_settings : dict, optional, default=None
        Dictionary of keyword arguments to pass to :func:`matplotlib.figure.Figure.colorbar`.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which the classification map will be plotted.
        Defaults to the current axis of the current figure.

    Returns
    -------
    data : dict
        Common classification plotting settings.

    See Also
    --------
    mcalf.visualisation.bar : Plot a bar chart of the classification abundances.
    mcalf.visualisation.plot_class_map : Plot a map of the classifications.
    mcalf.utils.smooth.mask_classifications : Mask 2D and 3D arrays of classifications.
    mcalf.utils.plot.calculate_extent : Calculate the extent from a particular data shape and resolution.
    mcalf.utils.plot.class_cmap : Create a listed colormap for a specific number of classifications.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.init_class_data
    """
    # Mask and average classification map according to the classification range and shape
    class_map, vmin, vmax = mask_classifications(class_map, vmin, vmax, reduce=reduce)

    # Create a list of the classifications
    classes = np.arange(vmin, vmax + 1, dtype=int)

    # Configure the color map
    if cmap is None:
        cmap = class_cmap(style, len(classes))

    # Calculate a specific extent if a resolution is specified
    extent = calculate_extent(class_map.shape, resolution, offset,
                              ax=ax, dimension=dimension)

    # Create and update colorbar settings
    cbar_settings = {'ax': [ax], 'ticks': classes}
    if colorbar_settings is not None:
        cbar_settings.update(colorbar_settings)

    data = {
        'class_map': class_map,
        'vmin': vmin,
        'vmax': vmax,
        'plot_vmin': vmin - 0.5,
        'plot_vmax': vmax + 0.5,
        'classes': classes,
        'extent': extent,
        'cmap': cmap,
        'colorbar_settings': cbar_settings,
    }

    return data
