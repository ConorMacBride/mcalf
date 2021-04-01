import glob

import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.gridspec import GridSpec

from mcalf.utils.smooth import mask_classifications
from mcalf.utils.plot import calculate_extent, class_cmap


__all__ = ['plot_classifications', 'plot_distribution', 'plot_class_map']


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


def plot_distribution(class_map, overall_classes=None, classes=None, time_index=None, cadence=None,
                   xticks=(0, 15, 2), yticks=(0, 15, 2), xscale=0.725 * 0.097, yscale=0.725 * 0.097,
                   output=None, file_prefix='classmap_plot_', file_ext='png',
                   figsize=(5 * 3 / 2.5, 3 * 3 / 2.5), dpi=600, fontfamily=None, cache=False):
    """Plot an image of the classifications at a particular time along with bar charts of the classifications

    Parameters
    ----------
    class_map : ndarray, ndim=2 or 3
        Two-dimensional array of classifications. If three dimensions are given, the first dimension is assumed to
        represent the time.
    overall_classes : ndarray or bool, optional
        The percentage of spectra that belong to each classification in the overall dataset. If omitted, these will
        be calculated used all of the classifications given is `class_map`. If true is given, these will also be
        calculated in the same way and returned without any plotting done. (This returned array can then be used to
        speed up later calls of this function.)
    classes : ndarray, optional, default = ndarray of [0, 1, 2, 3, 4]
        Array of all the possible classifications in `class_map`.
    time_index : int, optional, default = 0
        The index of the time dimension of `class_map`, required if class_map is 3D. Also used for plotting the time.
    cadence : float, units = seconds, optional, default = None
        If given, the time index will be multiplied by this value and converted into a time in minutes on the plot.
        Otherwise, the `time_index` will be plotted without units.
    xticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the x-axis ticks in Mm.
    yticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the y-axis ticks in Mm.
    xscale : float, optional = 0.725 * 0.097
        Scaling factor between x-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    yscale : float, optional = 0.725 * 0.097
        Scaling factor between y-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    output : str or bool, optional, default = None
        If present, the filename to save the plot as. If omitted, the plot will not be saved. If true, the filename
        will be generated using the `time_index` along with the `file_prefix` and `file_ext`.
    file_prefix : str, optional, default = 'classmap_plot_'
        The prefix to use in the filename when `output` is true.
    file_ext : str, optional, default = 'png'
        The file extension (without the dot) to use when `output` is true.
    figsize : 2-tuple, optional, default = None
        Size of the figure.
    dpi : int, optional, default = 600
        The number of dots per inch. For controlling the quality of the outputted figure.
    fontfamily : str, optional, default = None
        If provided, this family string will be added to the 'font' rc params group.
    cache : bool, optional, default = False
        If true, the plot will not be regenerated if the output filename already exists.

    Returns
    -------
    overall_classes : ndarray
        If `overall_classes` is initially true, their calculated values will be returned.
    """
    if classes is None:
        classes = np.arange(5, dtype=int)

    if overall_classes is None or isinstance(overall_classes, bool):
        just_print_overall_classes = True if overall_classes else False
        overall_classes = class_map.flatten()
        counts = np.zeros(len(classes))
        for i in classes:
            counts[i] = len(overall_classes[overall_classes == i])
        overall_classes = counts / len(overall_classes) * 100  # Convert to percentage
        if just_print_overall_classes:
            return overall_classes

    if class_map.ndim == 3:
        if time_index is None:
            raise ValueError('A `time_index` must be specified as multiple time dimensions are in `class_map`.')
        class_map = class_map[time_index]
    else:
        if class_map.ndim != 2:
            raise ValueError('`class_map` must have either 2 or 3 dimensions, got %s' % class_map.ndim)
        if time_index is None:
            time_index = 0

    if isinstance(output, bool) and output:
        output = '{}{:05d}.{}'.format(file_prefix, time_index, file_ext)

    if cache and output is not None and len(glob.glob(output)) > 0:
        return 0

    if fontfamily is not None:
        plt.rc('font', family=fontfamily)

    time = time_index if cadence is None else time_index * cadence / 60
    time_unit = '' if cadence is None else ' min'
    time_prefix = 't = ' if cadence is None else ''

    cmap_colors = np.array(['#0072b2', '#56b4e9', '#009e73', '#e69f00', '#d55e00'])[:len(classes)]
    cmap = colors.ListedColormap(cmap_colors)
    extent = (0, len(class_map[0]), 0, len(class_map))
    cmap.set_bad(color='#999999', alpha=1)
    bar_colors = cmap(classes)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    plt.sca(ax1)
    im = plt.imshow(class_map, cmap=cmap, vmin=min(classes)-0.5, vmax=max(classes)+0.5, extent=extent,
                    interpolation='nearest')
    fig.colorbar(im, ax=ax1, ticks=classes, orientation='vertical', label='absorption' + ' ' * 41 + 'emission')
    ax1.set_title('Classifications at {}{:.2f}{}'.format(time_prefix, time, time_unit))

    xticks_Mm = np.arange(*xticks)
    xticks = (xticks_Mm / xscale) + extent[0]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks_Mm)
    ax1.set_xlabel('Distance (Mm)')

    yticks_Mm = np.arange(*yticks)
    yticks = (yticks_Mm / yscale) + extent[2]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks_Mm)
    ax1.set_ylabel('Distance (Mm)')

    current_classes = class_map.flatten()
    counts = np.zeros(len(classes))
    for i in classes:
        counts[i] = len(current_classes[current_classes == i])
    current_classes = counts / len(current_classes) * 100  # Convert to percentage

    plt.sca(ax2)
    plt.bar(classes, current_classes, color=bar_colors)
    ax2.set_title('Current Classes (%)')

    plt.sca(ax3)
    plt.bar(classes, overall_classes, color=bar_colors)
    ax3.set_title('Overall Classes (%)')

    for ax in [ax2, ax3]:
        ax.set_xlabel(None)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_yscale('log')
        ax.set_ylim(0.01, 100)

    plt.show()

    if output is not None and isinstance(output, str):
        fig.savefig(output, bbox_inches='tight', dpi=dpi)


def plot_class_map(class_map, vmin=None, vmax=None, resolution=None, offset=(0, 0),
                   style='original', cmap=None, show_colorbar=True, colorbar_settings=None, ax=None):
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

    # Mask and average classification map according to the classification range and shape
    class_map, vmin, vmax = mask_classifications(class_map, vmin, vmax)

    # Create a list of the classifications
    classes = np.arange(vmin, vmax + 1, dtype=int)

    # Configure the color map
    if cmap is None:
        cmap = class_cmap(style, len(classes))

    # Calculate a specific extent if a resolution is specified
    # TODO: Allow the `dimension` to be set by the user.
    extent = calculate_extent(class_map.shape, resolution, offset,
                              ax=ax, dimension='distance')

    # Plot `class_map` (update vmin/vmax to improve displayed colorbar endpoints)
    im = ax.imshow(class_map, cmap=cmap, vmin=vmin-0.5, vmax=vmax+0.5,
                   origin='lower', extent=extent, interpolation='nearest')

    if show_colorbar:
        cbar_settings = {'ax': [ax], 'ticks': classes}
        if colorbar_settings is not None:
            cbar_settings.update(cbar_settings)
        ax.get_figure().colorbar(im, **cbar_settings)

    return im
