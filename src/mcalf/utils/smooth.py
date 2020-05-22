import numpy as np
import scipy.ndimage


__all__ = ['moving_average', 'gaussian_kern_3d', 'smooth_cube']


def moving_average(array, width):
    """Boxcar moving average

    Calculate the moving average of an array with a boxcar of defined width. An odd width is recommended.

    Parameters
    ----------
    array : ndarray
        Array to find the moving average of.
    width : int
        Width of the boxcar. Odd integer recommended.

    Returns
    -------
    averaged : ndarray of shape `array`
        Averaged array.
    """

    if not isinstance(width, int) or width <= 0 or width > len(array):
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
    """3D Gaussian kernel

    Create a Gaussian kernel of shape `width`*`width`*`width`.

    Parameters
    ----------
    width : int, optional, default = 5
        Length of all three dimensions of the Gaussian kernel.
    sigma : array_like, tuple, optional, default = (1, 1, 1)
        Sigma values for the time, horizontal and vertical dimensions.

    Returns
    -------
    kernel : ndarray, shape (`width`, `width`, `width`)
        The generated kernel.
    """
    s = np.linspace(-1, 1, width)
    x, y, z = np.meshgrid(s, s, s)
    kernel = np.exp(-x**2/(2*sigma[0]**2) - y**2/(2*sigma[1]**2) - z**2/(2*sigma[2]**2))
    return kernel


def smooth_cube(cube, mask, **kwargs):
    """Apply Gaussian smoothing to velocities

    Smooth the cube of velocities with a Gaussian kernel, applying weights at boundaries.

    Parameters
    ----------
    cube : ndarray, ndim=3
        Cube of velocities with dimensions [time, row, column].
    mask : ndarray, ndim=2
        The mask to apply to the [row, column] at every time.
    kwargs : optional
        Keyword arguments to pass to `gaussian_kern_3d`.

    Returns
    -------
    cube_ : ndarray, shape `cube`
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
