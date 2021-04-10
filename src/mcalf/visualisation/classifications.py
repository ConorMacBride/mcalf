import glob

import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.gridspec import GridSpec

from mcalf.utils.smooth import mask_classifications
from mcalf.utils.plot import calculate_extent, class_cmap


__all__ = ['plot_classifications', 'bar', 'plot_class_map', 'init_class_data']


def plot_classifications(class_map, spectra, labels, extent=(0, 200, 0, 200), xticks=(0, 15, 3), yticks=(0, 15, 3),
                         xscale=0.725*0.097, yscale=0.725*0.097, output=None, figsize=None, dpi=600, fontfamily=None):
    """Plot the spectra separated into their classifications along with an example classified map.

    Must be 5 classifications.

    Parameters
    ----------
    class_map : ndarray, ndim=2
        Two-dimensional array of classifications.
    spectra : ndarray, ndim=2
        Two-dimensional array with dimensions [spectra, wavelengths].
    labels : ndarray, ndim=1, length of `spectra`
        List of classifications for each spectrum in `spectra`.
    output : str, optional, default = None
        If present, the filename to save the plot as.
    figsize : 2-tuple, optional, default = None
        Size of the figure.
    dpi : int, optional, default = 600
        The number of dots per inch. For controlling the quality of the outputted figure.
    fontfamily : str, optional, default = None
        If provided, this family string will be added to the 'font' rc params group.
    vmin : float, optional, default = -max(|`velmap`|)
        Minimum velocity to plot. If not given, will be -vmax, for vmax not None.
    vmax : float, optional, default = max(|`velmap`|)
        Maximum velocity to plot. If not given, will be -vmin, for vmin not None.
    extent : 4-tuple, optional, default = (0, 200, 0, 200)
        Region the `velmap` is cropped to.
    xticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the x-axis ticks in Mm.
    yticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the y-axis ticks in Mm.
    xscale : float, optional = 0.725 * 0.097
        Scaling factor between x-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    yscale : float, optional = 0.725 * 0.097
        Scaling factor between y-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    """

    if fontfamily is not None:
        plt.rc('font', family=fontfamily)
    fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=dpi)

    gs = GridSpec(2, 3, figure=fig, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    map_plot = fig.add_subplot(gs[1, 2])
    axes = [ax1, ax2, ax3, ax4, ax5, map_plot]

    # Optimised for readers with color blindness
    cmap = colors.ListedColormap(['#0072b2', '#56b4e9', '#009e73', '#e69f00', '#d55e00'])

    for classification in range(5):
        n_plots = 0  # Number plotted for this classification
        ax = axes[classification]  # Select the axis
        class_colour = cmap(classification)
        for j in range(len(labels)):
            if labels[j] == classification:
                n_plots += 1
                ax.plot(spectra[j], linewidth=0.5, color=class_colour)
                if n_plots >= 20:  # Only plot the first 20 of each classification
                    break
        ax.set_xticks([])  # No wavelengths plotted
        ax.set_yticks([0, 1])  # Only show that intensity is scaled [0, 1]
        ax.margins(0)

    ax = axes[-1]  # Classification map will be placed in the last axis

    cmap.set_bad(color='white')  # Background for masked points
    class_map_float = np.asarray(class_map, dtype=float)
    class_map_float[class_map == -1] = np.nan
    classif_img = ax.imshow(class_map_float[::-1], cmap=cmap, vmin=-0.5, vmax=4.5, interpolation='nearest')

    ax.set_xlim(*extent[:2]), ax.set_ylim(*extent[2:])

    xticks_Mm = np.arange(*xticks)
    xticks = (xticks_Mm / xscale) + extent[0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_Mm)
    ax.set_xlabel('Distance (Mm)')

    yticks_Mm = np.arange(*yticks)
    yticks = (yticks_Mm / yscale) + extent[2]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_Mm)
    ax.set_ylabel('Distance (Mm)')

    cbar = fig.colorbar(classif_img, ax=axes, ticks=[0, 1, 2, 3, 4], orientation='horizontal', shrink=1, pad=0)
    cbar.ax.set_xticklabels(['0\nabsorption', '1', '2', '3', '4\nemission'])

    plt.show()

    if output is not None and isinstance(output, str):
        fig.savefig(output, bbox_inches='tight', dpi=dpi)


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
    """
    if ax is None:
        ax = plt.gca()

    if data is None:
        if class_map is None:  # `class_map` must always be provided
            raise TypeError("plot_class_map() missing 1 required positional argument: 'class_map'")
        data = init_class_data(class_map, vmin=vmin, vmax=vmax, reduce=reduce, style=style, cmap=cmap)

    # Count for each classification
    d = data['class_map'].flatten()
    counts = np.array([len(d[d == i]) for i in data['classes']])
    d = counts / len(d) * 100  # Convert to percentage

    b = ax.bar(data['classes'], d, color=data['cmap'](data['classes']))

    ax.set(xlabel='classification', ylabel='abundance (%)', yscale='log', ylim=(0.01, 100))

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
        Axes into which the velocity map will be plotted.
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
        cbar_settings.update(cbar_settings)

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
