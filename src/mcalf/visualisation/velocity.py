import copy

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import astropy.units

from mcalf.utils.plot import calculate_extent


__all__ = ['plot_map']


def plot_map(arr, mask=None, umbra_mask=None, resolution=None, offset=(0, 0), vmin=None, vmax=None,
             lw=None, show_colorbar=True, unit="km/s", ax=None):
    """Plot a velocity map array.

    Parameters
    ----------
    arr : numpy.ndarray[float] or astropy.units.quantity.Quantity, ndim=2
        Two-dimensional array of velocities.
    mask : numpy.ndarray[bool], ndim=2, shape=arr, optional, default=None
        Mask showing the region where velocities were found for. True is outside the
        velocity region and False is where valid velocities should be found. Specifying
        a mask allows for errors in the velocity calculation to be black and points
        outside the region to be gray. If omitted, all invalid points will be gray.
    umbra_mask : numpy.ndarray[bool], ndim=2, shape=arr, optional, default=None
        A mask of the umbra, True outside, False inside. If given, a contour will
        outline the umbra, or other feature the mask represents.
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
    vmin : float, optional, default= ``-max(|arr|)``
        Minimum velocity to plot. If not given, will be -vmax, for vmax not None.
    vmax : float, optional, default= ``max(|arr|)``
        Maximum velocity to plot. If not given, will be -vmin, for vmin not None.
    lw : float, optional, default=None
        The line width of the contour line plotted for `umbra_mask`.
        Passed as `linewidths` to :func:`matplotlib.axes.Axes.contour`.
    show_colorbar : bool, optional, default=True
        Whether to draw a colorbar.
    unit : str or astropy.units.UnitBase or astropy.units.quantity.Quantity, optional, default='km/s'
        The units of `arr` data. Printed on colorbar.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which the velocity map will be plotted.
        Defaults to the current axis of the current figure.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The object returned by :func:`matplotlib.axes.Axes.imshow` after plotting `arr`.

    See Also
    --------
    mcalf.models.FitResults.velocities : Calculate the Doppler velocities for an array of fits.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.plot_map
    """
    if ax is None:
        ax = plt.gca()

    # Validate `arr`
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise TypeError('`arr` must be a numpy.ndarray with 2 dimensions.')
    arr = arr.copy()  # Edit a copy of `arr`

    # Validate `mask` and `umbra_mask`
    for n, v in (('mask', mask), ('umbra_mask', umbra_mask)):
        if v is not None:
            if not isinstance(v, np.ndarray) or v.ndim != 2:
                raise TypeError(f'`{n}` must be a numpy.ndarray with 2 dimensions.')
            if v.shape != arr.shape:
                raise ValueError(f'`{n}` must be the same shape as `arr`')

    # Update default unit if unit present in `arr`
    if isinstance(arr[0, 0], astropy.units.quantity.Quantity):
        unit = arr.unit.to_string(astropy.units.format.LatexInline)
        arr = arr.value  # Remove unit
    # Convert a `unit` parameter that was provided as an astropy unit
    if isinstance(unit, (astropy.units.UnitBase, astropy.units.quantity.Quantity)):
        unit = unit.to_string(astropy.units.format.LatexInline)

    # Calculate a specific extent if a resolution is specified
    # TODO: Allow the `dimension` to be set by the user.
    extent = calculate_extent(arr.shape, resolution, offset,
                              ax=ax, dimension='distance')

    # Configure default colormap
    cmap = copy.copy(mpl.cm.get_cmap('bwr'))
    cmap.set_bad(color='#999999', alpha=1)

    # Show invalid pixels outside the mask as black, inside as gray
    if mask is not None:

        # Create image from mask
        mask = mask.astype(bool)
        unmasked_section = np.empty_like(mask, dtype=float)
        unmasked_section[mask] = np.nan  # Outside mask
        unmasked_section[~mask] = 1  # Inside mask

        # Configure colormap of mask
        cmap_mask = copy.copy(mpl.cm.get_cmap('gray'))
        cmap_mask.set_bad(color='#999999', alpha=1)

        # Show the masked region
        ax.imshow(unmasked_section, cmap=cmap_mask, origin='lower',
                  extent=extent, interpolation='nearest')

        arr[mask] = np.nan  # Remove values from `arr` that are outside mask
        cmap.set_bad(color='#000000', alpha=0)  # Update default colormap

    # Calculate range for symmetric colormap
    if vmin is None and vmax is None:
        vmax = np.nanmax(np.abs(arr))
        vmin = -vmax
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    # Show the velocities
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   extent=extent, interpolation='nearest')

    # Outline the umbra
    if umbra_mask is not None:
        umbra_mask = umbra_mask.astype(bool)
        ax.contour(umbra_mask, [0.5], colors='k', origin='lower',
                   extent=extent, linewidths=lw)

    if show_colorbar:
        ax.get_figure().colorbar(im, ax=[ax], label=f'Doppler velocity ({unit})')

    return im
