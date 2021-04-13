import numpy as np


__all__ = ['genmask', 'radial_distances']


def genmask(width, height, radius=np.inf, right_shift=0, up_shift=0):
    """Generate a circular mask of specified size.

    Parameters
    ----------
    width : int
        Width of mask.
    height : int
        Height of mask.
    radius : int, optional, default=inf
        Radius of mask.
    right_shift : int, optional, default=0
        Indices to shift forward through row.
    up_shift : int, optional, default=0
        Indices to shift forward through columns.

    Returns
    -------
    array : numpy.ndarray, shape=(height, width)
        The generated mask.

    Examples
    --------
    .. minigallery:: mcalf.utils.mask.genmask
    """
    array = radial_distances(width, height) < radius  # Create mask
    array = np.roll(array, [up_shift, right_shift], [0, 1])  # Apply shift
    return array


def radial_distances(n_cols, n_rows):
    """Generates a 2D array of specified shape of radial distances from the centre.

    Parameters
    ----------
    n_cols : int
        Number of columns.
    n_rows : int
        Number of rows.

    Returns
    -------
    array : numpy.ndarray, shape=(n_rows, n_cols)
        Array of radial distances.

    See Also
    --------
    genmask : Generates a circular mask.
    """
    horiz_mid = (n_cols - 1) / 2.0  # Horizontal midpoint
    verti_mid = (n_rows - 1) / 2.0  # Vertical midpoint
    array = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            array[i, j] = np.sqrt((j - horiz_mid)**2.0 + (i - verti_mid)**2.0)
    return array
