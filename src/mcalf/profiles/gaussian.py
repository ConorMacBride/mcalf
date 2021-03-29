import numpy as np


__all__ = ['single_gaussian']


def single_gaussian(x, a, b, c, d):
    """Gaussian function.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelengths to evaluate Gaussian function at.
    a : float
        Amplitude.
    b : float
        Central line core.
    c : float
        Sigma of Gaussian.
    d : float
        Background to add.

    Returns
    -------
    result : numpy.ndarray, shape=`x.shape`
        The value of the Gaussian function here.
    """
    g = a * np.exp(- (x-b)**2.0 / (2.0 * c**2.0))
    return g + d
