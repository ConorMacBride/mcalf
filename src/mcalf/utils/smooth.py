import numpy as np
import scipy.ndimage


__all__ = ['moving_average', 'gaussian_kern_3d', 'smooth_cube', 'mask_classifications']


def moving_average(array, width):
    """Boxcar moving average.

    Calculate the moving average of an array with a boxcar of defined width. An odd width is recommended.

    Parameters
    ----------
    array : numpy.ndarray, ndim=1
        Array to find the moving average of.
    width : int
        Width of the boxcar. Odd integer recommended. Less than or equal to length of `array`.

    Returns
    -------
    averaged : numpy.ndarray, shape=`array`
        Averaged array.

    Notes
    -----
    The moving average is calculated at each point of the `array` by finding the (unweighted) mean of the subarrays
    of length given by `width`. These subarrays are centred at the point in the `array` that the current average is
    currently being calculated for. If an odd `width` is chosen, the sub array will include the current point plus an
    equal number of points on either side. However, if an even `width` is chosen, the sub array will bias including the
    extra point to the left of the current index. If the subarray spans past the boundaries, the values beyond the
    boundary is ignored and the mean is calculated by dividing by the number of points that are inside the boundaries.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> moving_average(x, 3)
    array([1.5, 2. , 3. , 4. , 4.5])
    >>> moving_average(x, 2)
    array([1. , 1.5, 2.5, 3.5, 4.5])
    """

    if not isinstance(width, (int, np.integer)) or width <= 0 or width > len(array):
        raise ValueError("`width` must be a positive integer less than the length of `array`, got %s." % width)

    kernel = np.ones(width) / width
    averaged = np.convolve(array, kernel, mode='same')

    # Apply corrections at edge
    to_correct_left = int(width/2)
    to_correct_right = to_correct_left
    to_correct_right -= 0 if width % 2 else 1

    # Adjust weighting on left edge
    n_overlap = width - 1
    for i in np.arange(to_correct_left-1, -1, -1):
        averaged[i] *= width / n_overlap
        n_overlap -= 1

    # Adjust weighting on right edge
    n_overlap = width - 1
    for i in range(len(array) - to_correct_right, len(array)):
        averaged[i] *= width / n_overlap
        n_overlap -= 1

    return averaged


def gaussian_kern_3d(width=5, sigma=(1, 1, 1)):
    """3D Gaussian kernel.

    Create a Gaussian kernel of shape `width`*`width`*`width`.

    Parameters
    ----------
    width : int, optional, default=5
        Length of all three dimensions of the Gaussian kernel.
    sigma : array_like, tuple, optional, default=(1, 1, 1)
        Sigma values for the time, horizontal and vertical dimensions.

    Returns
    -------
    kernel : numpy.ndarray, shape=(`width`, `width`, `width`)
        The generated kernel.

    Examples
    --------
    >>> gaussian_kern_3d(width=3, sigma=(2, 1, 1.5))
    array([[[0.42860385, 0.53526143, 0.42860385],
            [0.48567179, 0.60653066, 0.48567179],
            [0.42860385, 0.53526143, 0.42860385]],
    <BLANKLINE>
           [[0.70664828, 0.8824969 , 0.70664828],
            [0.8007374 , 1.        , 0.8007374 ],
            [0.70664828, 0.8824969 , 0.70664828]],
    <BLANKLINE>
           [[0.42860385, 0.53526143, 0.42860385],
            [0.48567179, 0.60653066, 0.48567179],
            [0.42860385, 0.53526143, 0.42860385]]])
    """
    s = np.linspace(-1, 1, width)
    x, y, z = np.meshgrid(s, s, s)
    kernel = np.exp(-x**2/(2*sigma[0]**2) - y**2/(2*sigma[1]**2) - z**2/(2*sigma[2]**2))
    return kernel


def smooth_cube(cube, mask, **kwargs):
    """Apply Gaussian smoothing to velocities.

    Smooth the cube of velocities with a Gaussian kernel, applying weights at boundaries.

    Parameters
    ----------
    cube : numpy.ndarray, ndim=3
        Cube of velocities with dimensions [time, row, column].
    mask : numpy.ndarray, ndim=2
        The mask to apply to the [row, column] at every time. Points that are 0 or false will be removed.
    **kwargs : dict, optional
        Keyword arguments to pass to :func:`gaussian_kern_3d`.

    Returns
    -------
    cube_ : numpy.ndarray, shape=`cube`
        The smoothed cube.
    """
    # Masking
    cube_ = cube.copy()
    mask3d = np.empty_like(cube_)  # Mask for `cube_`
    for timestep in range(len(mask3d)):  # Copy mask throughout time
        mask3d[timestep] = mask.copy()
    cube_[np.isnan(cube_)] = 0  # Remove NaN
    cube_[mask3d == 0] = 0  # Remove masked points

    kernel = gaussian_kern_3d(**kwargs)  # Get the kernel

    # Count the number of neighbours (weighting) at each pixel
    neighbour_count = scipy.ndimage.convolve(mask3d, kernel, mode="constant")
    m = neighbour_count > 0  # Record the points that do have neighbours

    cube_ = scipy.ndimage.convolve(cube_, kernel, mode="constant")  # Apply smoothing
    cube_[m] /= neighbour_count[m]  # Normalise depending on neighbours

    cube_[mask3d == 0] = np.nan  # Restore masked pixels
    cube_[np.isnan(cube)] = np.nan  # Restore NaN values

    return cube_


def mask_classifications(class_map, vmin=None, vmax=None, reduce=True):
    """Mask 2D and 3D arrays of classifications.

    If 3D, also reduces to 2D by selecting the most common classification
    along the first dimension.

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

    Returns
    -------
    class_map : numpy.ndarray[int], ndim=2
        `class_map` with values between `vmin` and `vmax`
        averaged along the first dimension.
    vmin : int
        Updated `vmin` value.
    vmax : int
        Updated `vmax` value.

    See Also
    --------
    mcalf.visualisation.plot_class_map : Plot a map of the classifications.
    """
    # Validate the `class_map` parameter
    if not isinstance(class_map, np.ndarray):
        raise TypeError(f'`class_map` must be a numpy.ndarray, got {type(class_map)}.')
    if class_map.ndim not in (2, 3):
        raise ValueError(f'`class_map` must have either 2 or 3 dimensions, got {class_map.ndim}.')
    if not issubclass(class_map.dtype.type, np.integer):
        raise TypeError(f'`class_map` must be an array of integers, got {class_map.dtype}.')

    # Short-circuit if all classifications negative
    if len(class_map[class_map >= 0]) == 0:
        class_map = class_map.copy()
        if reduce and class_map.ndim == 3:
            class_map = class_map[0]
        class_map[:] = -1  # Set all invalid to -1 for consistency with main code
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 0
        return class_map, vmin, vmax

    # Validate or set `vmin` and `vmax`
    if vmax is None:
        vmax = np.max(class_map[class_map >= 0])
    elif not isinstance(vmax, (int, np.integer)):
        raise TypeError(f'`vmax` must be an integer, got {type(vmax)}.')
    elif vmax < 0:
        raise ValueError(f'`vmax` must not be less than zero.')
    if vmin is None:
        vmin = np.min(class_map[class_map >= 0])
    elif not isinstance(vmin, (int, np.integer)):
        raise TypeError(f'`vmin` must be an integer, got {type(vmin)}.')
    elif vmin < 0:
        raise ValueError(f'`vmin` must not be less than zero.')
    if vmin > vmax:
        raise ValueError(f'`vmin` must be less than `vmax`, got {vmin} to {vmax}.')

    # Ignore classifications outside of range
    class_map = class_map.copy()
    class_map[class_map < vmin] = -1
    class_map[class_map > vmax] = -1

    # If 3D, choose the most common classification along the first dimension
    if reduce and class_map.ndim == 3:

        c = np.arange(vmin, vmax + 1)  # classes
        bins = np.arange(vmin - 0.5, vmax + 1)  # len(bins)=len(c)+1

        ave = np.empty(class_map.shape[1:], dtype=int)  # init

        for i in range(len(class_map[0])):  # spatial rows
            for j in range(len(class_map[0, 0])):  # spatial columns
                counts = np.histogram(class_map[:, i, j], bins)[0]
                if counts.max() == 0:
                    ave[i, j] = -1  # no valid classifications
                else:
                    ave[i, j] = c[counts.argmax()]

        class_map = ave  # Replace 3D array with 2D average

    return class_map, vmin, vmax
