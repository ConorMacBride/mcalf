import glob

import numpy as np
from matplotlib import pyplot as plt, colors, cm
from matplotlib.gridspec import GridSpec


__all__ = ['plot_classifications', 'plot_class_map', 'plot_averaged_class_map']


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


def plot_class_map(class_map, overall_classes=None, classes=None, time_index=None, cadence=None,
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


def plot_averaged_class_map(class_map, classes=None, continuous=False,
                   xticks=(0, 15, 2), yticks=(0, 15, 2), xscale=0.725 * 0.097, yscale=0.725 * 0.097,
                   output=None, figsize=None, dpi=600, fontfamily=None):
    """Plot an image of the time averaged classifications

    Parameters
    ----------
    class_map : ndarray, ndim=3
        Three-dimensional array of classifications, with the times given in the first dimension.
    classes : ndarray, optional, default = ndarray of [0, 1, 2, 3, 4]
        Array of all the possible classifications in `class_map`.
    continuous : bool, optional, default = False
        Whether to plot the with a continuous color scale or round to the nearest classification.
    xticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the x-axis ticks in Mm.
    yticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the y-axis ticks in Mm.
    xscale : float, optional = 0.725 * 0.097
        Scaling factor between x-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    yscale : float, optional = 0.725 * 0.097
        Scaling factor between y-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    output : str, optional, default = None
        If present, the filename to save the plot as. If omitted, the plot will not be saved.
    figsize : 2-tuple, optional, default = None
        Size of the figure.
    dpi : int, optional, default = 600
        The number of dots per inch. For controlling the quality of the outputted figure.
    fontfamily : str, optional, default = None
        If provided, this family string will be added to the 'font' rc params group.
    """

    if classes is None:
        classes = np.arange(5, dtype=int)

    if class_map.ndim != 3:
        raise ValueError('`class_map` must have 3 dimensions, got %s' % class_map.ndim)

    class_map = np.mean(class_map, axis=0)

    if fontfamily is not None:
        plt.rc('font', family=fontfamily)

    if continuous:
        cmap = cm.get_cmap('binary_r')
        vmin = min(classes)
        vmax = max(classes)
    else:
        cmap_colors = np.array(['#0072b2', '#56b4e9', '#009e73', '#e69f00', '#d55e00'])[:len(classes)]
        cmap = colors.ListedColormap(cmap_colors)
        vmin = min(classes) - 0.5
        vmax = max(classes) + 0.5

    extent = (0, len(class_map[0]), 0, len(class_map))
    cmap.set_bad(color='#999999', alpha=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    im = ax.imshow(class_map, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
    fig.colorbar(im, ax=ax, ticks=classes, orientation='vertical', label='absorption' + ' ' * 47 + 'emission')

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

    plt.show()

    if output is not None and isinstance(output, str):
        fig.savefig(output, bbox_inches='tight', dpi=dpi)
