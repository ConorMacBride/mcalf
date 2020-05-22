import numpy as np
from matplotlib import pyplot as plt


__all__ = ['plot_map']


def plot_map(velmap, mask=None, umbra_mask=None, figsize=None, vmin=None, vmax=None, extent=None,
             xticks=(0, 15, 2), yticks=(0, 15, 2), xscale=0.725 * 0.097, yscale=0.725 * 0.097,
             output=None, dpi=600, fontfamily=None, units="km/s", linewidths=None):
    """Plot the velocity map

    Plots a velocity map for publication.

    Parameters
    ----------
    velmap : ndarray, ndim=2
        Two-dimensional array of velocities.
    mask : ndarray of bool, ndim=2, shape `velmap`, optional, default = None
        Mask showing the region where velocities were found for. True is outside the
        velocity region and False is where valid velocities should be found. Specifying
        a mask allows for errors in the velocity calculation to be black and points
        outside the region to be gray. If omitted, all invalid points will be gray.
    umbra_mask : ndarray of bool, ndim=2, shape `velmap`, optional, default = None
        A mask of the umbra, True outside, False inside. If given, a contour will
        outline the umbra, or other feature the mask represents.
    output : str, optional, default = None
        If present, the filename to save the plot as.
    figsize : 2-tuple
        Size of the figure.
    dpi : int
        The number of dots per inch. For controlling the quality of the outputted figure.
    fontfamily : str, optional, default = None
        If provided, this family string will be added to the 'font' rc params group.
    vmin : float, optional, default = -max(|`velmap`|)
        Minimum velocity to plot. If not given, will be -vmax, for vmax not None.
    vmax : float, optional, default = max(|`velmap`|)
        Maximum velocity to plot. If not given, will be -vmin, for vmin not None.
    extent : 4-tuple, optional, default = (0, n_rows, 0, n_columns)
        Data range of `velmap`. TODO: Remove (assume one-to-one relationship)
    xticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the x-axis ticks in Mm.
    yticks : 3-tuple, optional, default = (0, 15, 2)
        The start, stop and step for the y-axis ticks in Mm.
    xscale : float, optional = 0.725 * 0.097
        Scaling factor between x-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    yscale : float, optional = 0.725 * 0.097
        Scaling factor between y-axis data coordinate steps and 1 Mm. Mm = data / xscale.
    units : str, optional, default = 'km/s'
        The units of `velmap` data. Printed on colorbar.
    linewidths : float or sequence of floats, optional, default = None
        The width of the contours plotted for `umbra_mask`.
    """
    if fontfamily is not None:
        plt.rc('font', family=fontfamily)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = plt.get_cmap('bwr')

    velmap_cropped = velmap.copy()

    if extent is None:
        extent = (0, len(velmap_cropped), 0, len(velmap_cropped[0]))

    if mask is not None:
        # Show invalid pixels outside the mask as black, inside as gray
        if mask.shape != velmap_cropped.shape:
            raise ValueError("`mask` must be the same shape as `velmap`")
        unmasked_section = np.empty_like(mask, dtype=float)
        unmasked_section[mask] = np.nan
        unmasked_section[~mask] = 1
        cmap_mask = plt.get_cmap('gray')
        cmap_mask.set_bad(color='#999999', alpha=1)
        ax.imshow(unmasked_section, cmap=cmap_mask, extent=extent, interpolation='nearest')
        cmap.set_bad(color='#000000', alpha=0)
        velmap_cropped[mask] = np.nan
    else:
        cmap.set_bad(color='#999999', alpha=1)

    # Show the velocities
    if vmin is None and vmax is None:
        vmax = np.nanmax(np.abs(velmap))
        vmin = -vmax
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin
    im = ax.imshow(velmap_cropped, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')

    if umbra_mask is not None:
        if umbra_mask.shape != velmap_cropped.shape:
            raise ValueError("`umbra_mask` must be the same shape as `velmap`")
        plt.contour(umbra_mask[::-1], cmap='binary', extent=extent, linewidths=linewidths)

    plt.colorbar(im, ax=ax, label='Doppler velocity ({})'.format(units))

    xticks_Mm = np.arange(*xticks)
    xticks = (xticks_Mm / xscale) + extent[0]
    plt.xticks(xticks, xticks_Mm)
    plt.xlabel('Distance (Mm)')

    yticks_Mm = np.arange(*yticks)
    yticks = (yticks_Mm / yscale) + extent[2]
    plt.yticks(yticks, yticks_Mm)
    plt.ylabel('Distance (Mm)')

    plt.show()

    if output is not None and isinstance(output, str):
        fig.savefig(output, bbox_inches='tight', dpi=dpi)
