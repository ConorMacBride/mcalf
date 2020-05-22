import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad, quad_vec

# Load the C library
import os.path
import ctypes
import glob
# # Commands to manually generate
# gcc -Wall -fPIC -c voigt.c
# gcc -shared -o libvoigt.so voigt.o
dllabspath = "{0}{1}".format(os.path.dirname(os.path.abspath(__file__)), os.path.sep)  # Path to libraries directory
libfile = glob.glob('{}ext_voigtlib.*.so'.format(dllabspath))[0]  # Select first (and only) library in this directory
lib = ctypes.CDLL(libfile)  # Load the library
lib.func.restype = ctypes.c_double  # Specify the expected result type
lib.func.argtypes = (ctypes.c_int, ctypes.c_double)  # Specify the type of the input parameters
cvoigt = lib.func  # Create alias for the specific function used in functions below


# Parameters for `voigt_approx_nobg` and other approx. Voigt functions
params = np.array([[-1.2150, -1.3509, -1.2150, -1.3509],
                   [1.2359, 0.3786, -1.2359, -0.3786],
                   [-0.3085, 0.5906, -0.3085, 0.5906],
                   [0.0210, -1.1858, -0.0210, 1.1858]])
sqrt_ln2 = np.sqrt(np.log(2))
sqrt_pi = np.sqrt(np.pi)
A, B, C, D = params


__all__ = ['voigt_approx_nobg', 'voigt_approx', 'double_voigt_approx_nobg', 'double_voigt_approx',
           'voigt_nobg', 'voigt', 'double_voigt_nobg', 'double_voigt']


def voigt_approx_nobg(x, a, b, s, g):
    """Voigt function (efficient approximation) with no background (Base approx. Voigt function)

    This is the base for all other approximated Voigt functions. Not implemented in any models yet as initial tests
    exhibited slow convergence.

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a : float
        Amplitude of the Lorentzian.
    b : float
        Central line core.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_approx : Approximated Voigt function with background added
    double_voigt_approx_nobg : Two approximated Voigt functions added together
    double_voigt_approx : Two approximated Voigt functions and a background added together
    voigt_nobg : Base Voigt function with no background
    voigt : Voigt function with background added
    double_voigt_nobg : Two Voigt functions added together
    double_voigt : Two Voigt function and a background added together

    Notes
    -----
    This algorithm is taken from A. B. McLean et al.[1]_.

    .. [1] A. B. McLean, C. E. J. Mitchell and D. M. Swanston, "Implementation of an efficient analytical
    approximation to the Voigt function for photoemission lineshape analysis," Journal of Electron Spectroscopy and
    Related Phenomena, vol. 69, pp. 125-132, 1994. https://doi.org/10.1016/0368-2048(94)02189-7
    """
    fwhm_g = 2 * s * np.sqrt(2 * np.log(2))
    fwhm_l = 2 * g
    xx = (x - b) * 2 * sqrt_ln2 / fwhm_g
    xx = xx[..., np.newaxis]
    yy = fwhm_l * sqrt_ln2 / fwhm_g
    yy = yy[..., np.newaxis]
    v = np.sum((C * (yy - A) + D * (xx - B)) / ((yy - A) ** 2 + (xx - B) ** 2), axis=-1)
    return fwhm_l * a * sqrt_pi / fwhm_g * v


def voigt_approx(x, a, b, s, g, d):
    """Voigt function (efficient approximation) with background

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a : float
        Amplitude of the Lorentzian.
    b : float
        Central line core.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).
    d : float
        Background.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_approx_nobg : Base approximated Voigt function with no background
    double_voigt_approx_nobg : Two approximated Voigt functions added together
    double_voigt_approx : Two approximated Voigt functions and a background added together
    voigt_nobg : Base Voigt function with no background
    voigt : Voigt function with background added
    double_voigt_nobg : Two Voigt functions added together
    double_voigt : Two Voigt function and a background added together
    """
    return voigt_approx_nobg(x, a, b, s, g) + d


def double_voigt_approx_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2):
    """Double Voigt function (efficient approximation) with no background

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a1 : float
        Amplitude of the Lorentzian of 1st Voigt function.
    b1 : float
        Central line core of 1st Voigt function.
    s1 : float
        Sigma (for Gaussian) of 1st Voigt function.
    g1 : float
        Gamma (for Lorentzian) of 1st Voigt function.
    a2 : float
        Amplitude of 2st Voigt function.
    b2 : float
        Central line core of 2st Voigt function.
    s2 : float
        Sigma (for Gaussian) of 2st Voigt function.
    g2 : float
        Gamma (for Lorentzian) of 2st Voigt function.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_approx_nobg : Base approximated Voigt function with no background
    voigt_approx : Approximated Voigt function with background added
    double_voigt_approx : Two approximated Voigt functions and a background added together
    voigt_nobg : Base Voigt function with no background
    voigt : Voigt function with background added
    double_voigt_nobg : Two Voigt functions added together
    double_voigt : Two Voigt function and a background added together
    """
    return voigt_approx_nobg(x, a1, b1, s1, g1) + voigt_approx_nobg(x, a2, b2, s2, g2)


def double_voigt_approx(x, a1, b1, s1, g1, a2, b2, s2, g2, d):
    """Double Voigt function (efficient approximation) with background

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a1 : float
        Amplitude of the Lorentzian of 1st Voigt function.
    b1 : float
        Central line core of 1st Voigt function.
    s1 : float
        Sigma (for Gaussian) of 1st Voigt function.
    g1 : float
        Gamma (for Lorentzian) of 1st Voigt function.
    a2 : float
        Amplitude of 2st Voigt function.
    b2 : float
        Central line core of 2st Voigt function.
    s2 : float
        Sigma (for Gaussian) of 2st Voigt function.
    g2 : float
        Gamma (for Lorentzian) of 2st Voigt function.
    d : float
        Background.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_approx_nobg : Base approximated Voigt function with no background
    voigt_approx : Approximated Voigt function with background added
    double_voigt_approx_nobg : Two approximated Voigt functions added together
    voigt_nobg : Base Voigt function with no background
    voigt : Voigt function with background added
    double_voigt_nobg : Two Voigt functions added together
    double_voigt : Two Voigt function and a background added together
    """
    return voigt_approx_nobg(x, a1, b1, s1, g1) + voigt_approx_nobg(x, a2, b2, s2, g2) + d


def voigt_nobg(x, a, b, s, g, clib=True):
    """Voigt function with no background (Base Voigt function)

    This is the base of all the other Voigt functions.

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a : float
        Amplitude.
    b : float
        Central line core.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).
    clib : bool, optional, default = True
        Whether to use the complied C library or a slower Python version. If using the C library, the accuracy
        of the integration is reduced to give the code a significant speed boost. Python version can be used when
        speed is not a priority. Python version will remove deviations that are sometimes present around the wings
        due to the reduced accuracy.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt : Voigt function with background added
    double_voigt_nobg : Two Voigt functions added together
    double_voigt : Two Voigt function and a background added together

    More Info
    ---------
    More information on the Voigt function can be found here: https://en.wikipedia.org/wiki/Voigt_profile
    """
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    u = x - b
    if clib:
        i = [quad(cvoigt, -np.inf, np.inf, args=(v, s, g), epsabs=1.49e-1, epsrel=1.49e-4)[0] for v in u]
    else:
        i = quad_vec(lambda y: np.exp(-y**2 / (2 * s**2)) / (g**2 + (u - y)**2), -np.inf, np.inf)[0]
    const = g / (s * np.sqrt(2 * np.pi**3))
    return a * const * np.array(i)


def voigt(x, a, b, s, g, d, clib=True):
    """Voigt function with background

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a : float
        Amplitude.
    b : float
        Central line core.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).
    d : float
        Background.
    clib : bool, optional, default = True
        Whether to use the complied C library or a slower Python version. If using the C library, the accuracy
        of the integration is reduced to give the code a significant speed boost. Python version can be used when
        speed is not a priority. Python version will remove deviations that are sometimes present around the wings
        due to the reduced accuracy.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_nobg : Base Voigt function with no background
    double_voigt_nobg : Two Voigt functions added together
    double_voigt : Two Voigt function and a background added together

    More Info
    ---------
    More information on the Voigt function can be found here: https://en.wikipedia.org/wiki/Voigt_profile
    """
    return voigt_nobg(x, a, b, s, g, clib) + d


def double_voigt_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2, clib=True):
    """Double Voigt function with no background

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a1 : float
        Amplitude of 1st Voigt function.
    b1 : float
        Central line core of 1st Voigt function.
    s1 : float
        Sigma (for Gaussian) of 1st Voigt function.
    g1 : float
        Gamma (for Lorentzian) of 1st Voigt function.
    a2 : float
        Amplitude of 2st Voigt function.
    b2 : float
        Central line core of 2st Voigt function.
    s2 : float
        Sigma (for Gaussian) of 2st Voigt function.
    g2 : float
        Gamma (for Lorentzian) of 2st Voigt function.
    clib : bool, optional, default = True
        Whether to use the complied C library or a slower Python version. If using the C library, the accuracy
        of the integration is reduced to give the code a significant speed boost. Python version can be used when
        speed is not a priority. Python version will remove deviations that are sometimes present around the wings
        due to the reduced accuracy.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_nobg : Base Voigt function with no background
    voigt : Voigt function with background added
    double_voigt : Two Voigt function and a background added together

    More Info
    ---------
    More information on the Voigt function can be found here: https://en.wikipedia.org/wiki/Voigt_profile
    """
    return voigt_nobg(x, a1, b1, s1, g1, clib) + voigt_nobg(x, a2, b2, s2, g2, clib)


def double_voigt(x, a1, b1, s1, g1, a2, b2, s2, g2, d, clib=True):
    """Double Voigt function with background

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Voigt function at.
    a1 : float
        Amplitude of 1st Voigt function.
    b1 : float
        Central line core of 1st Voigt function.
    s1 : float
        Sigma (for Gaussian) of 1st Voigt function.
    g1 : float
        Gamma (for Lorentzian) of 1st Voigt function.
    a2 : float
        Amplitude of 2st Voigt function.
    b2 : float
        Central line core of 2st Voigt function.
    s2 : float
        Sigma (for Gaussian) of 2st Voigt function.
    g2 : float
        Gamma (for Lorentzian) of 2st Voigt function.
    d : float
        Background.
    clib : bool, optional, default = True
        Whether to use the complied C library or a slower Python version. If using the C library, the accuracy
        of the integration is reduced to give the code a significant speed boost. Python version can be used when
        speed is not a priority. Python version will remove deviations that are sometimes present around the wings
        due to the reduced accuracy.

    Returns
    -------
    result : ndarray of shape `x.shape`
        The value of the Voigt function here.

    See Also
    --------
    voigt_nobg : Base Voigt function with no background
    voigt : Voigt function with background added
    double_voigt_nobg : Two Voigt functions added together

    More Info
    ---------
    More information on the Voigt function can be found here: https://en.wikipedia.org/wiki/Voigt_profile
    """
    return double_voigt_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2, clib) + d
