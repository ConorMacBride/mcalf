import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad, quad_vec
from scipy.special import voigt_profile

# Load the C library
import ctypes
from pathlib import Path
# # Commands to manually generate
# gcc -Wall -fPIC -c voigt.c
# gcc -shared -o libvoigt.so voigt.o
dllabspath = Path(__file__).absolute().parent  # Path to libraries directory
try:
    libfile = [str(i) for i in dllabspath.glob('ext_voigtlib.*.so')][0]  # Select first (and only) library
    lib = ctypes.CDLL(libfile)  # Load the library
    lib.func.restype = ctypes.c_double  # Specify the expected result type
    lib.func.argtypes = (ctypes.c_int, ctypes.c_double)  # Specify the type of the input parameters
    cvoigt = lib.func  # Create alias for the specific function used in functions below
    CLIB_INSTALLED = True
except IndexError:  # File does not exist
    CLIB_INSTALLED = False

# Parameters for `voigt_approx_nobg` and other approx. Voigt functions
params = np.array([[-1.2150, -1.3509, -1.2150, -1.3509],
                   [1.2359, 0.3786, -1.2359, -0.3786],
                   [-0.3085, 0.5906, -0.3085, 0.5906],
                   [0.0210, -1.1858, -0.0210, 1.1858]])
sqrt_ln2 = np.sqrt(np.log(2))
sqrt_pi = np.sqrt(np.pi)
A, B, C, D = params


__all__ = ['voigt_integrate', 'voigt_faddeeva', 'voigt_mclean',
           'voigt_nobg', 'voigt', 'double_voigt_nobg', 'double_voigt']


def voigt_integrate(x, s, g, clib=CLIB_INSTALLED, **kwargs):
    """Voigt function implementation (calculated by integrating).

    The default Voigt implementation.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelengths to evaluate Voigt function at.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).
    clib : bool, optional, default=True
        Whether to use the complied C library or a slower Python version. If using the C library, the accuracy
        of the integration is reduced to give the code a significant speed boost. Python version can be used when
        speed is not a priority. Python version will remove deviations that are sometimes present around the wings
        due to the reduced accuracy. If the C extensions is not installed, will default to false.

    Returns
    -------
    result : numpy.ndarray, shape=`x.shape`
        The value of the Voigt function here.

    Notes
    -----
    More information on the Voigt function can be found here: https://en.wikipedia.org/wiki/Voigt_profile
    """
    # return a * voigt_profile(x - b, s, g)
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    if clib:
        i = [quad(cvoigt, -np.inf, np.inf, args=(v, s, g), epsabs=1.49e-1, epsrel=1.49e-4)[0] for v in x]
    else:
        i = quad_vec(lambda y: np.exp(-y**2 / (2 * s**2)) / (g**2 + (x - y)**2), -np.inf, np.inf)[0]
    const = g / (s * np.sqrt(2 * np.pi**3))
    return const * np.array(i)


def voigt_faddeeva(x, s, g, **kwargs):
    """Voigt function implementation (Faddeeva).

    Parameters
    ----------
    x : numpy.ndarray
        Wavelengths to evaluate Voigt function at.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).

    Returns
    -------
    result : numpy.ndarray, shape=`x.shape`
        The value of the Voigt function here.
    """
    return voigt_profile(x, s, g)


def voigt_mclean(x, s, g, **kwargs):
    """Voigt function implementation (efficient approximation).

    Not implemented in any models yet as initial tests exhibited slow convergence.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelengths to evaluate Voigt function at.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian).

    Returns
    -------
    result : numpy.ndarray, shape=`x.shape`
        The value of the Voigt function here.

    Notes
    -----
    This algorithm is taken from A. B. McLean et al. [1]_.

    References
    ----------
    .. [1] A. B. McLean, C. E. J. Mitchell and D. M. Swanston, "Implementation of an efficient analytical
      approximation to the Voigt function for photoemission lineshape analysis," Journal of Electron Spectroscopy and
      Related Phenomena, vol. 69, pp. 125-132, 1994. https://doi.org/10.1016/0368-2048(94)02189-7
    """
    fwhm_g = 2 * s * np.sqrt(2 * np.log(2))
    fwhm_l = 2 * g
    xx = x * 2 * sqrt_ln2 / fwhm_g
    xx = xx[..., np.newaxis]
    yy = fwhm_l * sqrt_ln2 / fwhm_g
    yy = yy[..., np.newaxis]
    v = np.sum((C * (yy - A) + D * (xx - B)) / ((yy - A) ** 2 + (xx - B) ** 2), axis=-1)
    return fwhm_l * sqrt_pi / fwhm_g * v


def voigt_nobg(x, a, b, s, g, impl=voigt_faddeeva, **kwargs):
    """Voigt function with no background (Base Voigt function).

    This is the base of all the other Voigt functions.

    Parameters
    ----------
    ${SINGLE_VOIGT}

    ${SEE_ALSO}
    """
    return a * impl(x - b, s, g, **kwargs)


def voigt(x, a, b, s, g, d, **kwargs):
    """Voigt function with background.

    Parameters
    ----------
    ${SINGLE_VOIGT}
    ${BACKGROUND}

    ${SEE_ALSO}
    """
    return voigt_nobg(x, a, b, s, g, **kwargs) + d


def double_voigt_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2, **kwargs):
    """Double Voigt function with no background.

    Parameters
    ----------
    ${DOUBLE_VOIGT}

    ${SEE_ALSO}
    """
    return voigt_nobg(x, a1, b1, s1, g1, **kwargs) + voigt_nobg(x, a2, b2, s2, g2, **kwargs)


def double_voigt(x, a1, b1, s1, g1, a2, b2, s2, g2, d, **kwargs):
    """Double Voigt function with background.

    Parameters
    ----------
    ${DOUBLE_VOIGT}
    ${BACKGROUND}

    ${SEE_ALSO}
    """
    return double_voigt_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2, **kwargs) + d


# Define "Parameters" options
__input_x = """
    x : numpy.ndarray
        Wavelengths to evaluate Voigt function at."""
__single_voigt = __input_x + """
    a : float
        Amplitude of the Lorentzian.
    b : float
        Central line core.
    s : float
        Sigma (for Gaussian).
    g : float
        Gamma (for Lorentzian)."""
__double_voigt = __input_x + """
    a1 : float
        Amplitude of 1st Voigt function.
    b1 : float
        Central line core of 1st Voigt function.
    s1 : float
        Sigma (for Gaussian) of 1st Voigt function.
    g1 : float
        Gamma (for Lorentzian) of 1st Voigt function.
    a2 : float
        Amplitude of 2nd Voigt function.
    b2 : float
        Central line core of 2nd Voigt function.
    s2 : float
        Sigma (for Gaussian) of 2nd Voigt function.
    g2 : float
        Gamma (for Lorentzian) of 2nd Voigt function."""
__background = """
    d : float
        Background."""


def __see_also(func):
    """Return the "See Also" section with the current function removed."""
    see_also = filter(lambda x: f" {func.__name__} " not in x, [
        '    voigt_nobg : Base Voigt function with no background.',
        '    voigt : Voigt function with background added.',
        '    double_voigt_nobg : Two Voigt functions added together.',
        '    double_voigt : Two Voigt function and a background added together.',
    ])
    return """
    Returns
    -------
    result : numpy.ndarray, shape=`x.shape`
        The value of the Voigt function here.

    See Also
    --------
    """ + "\n".join(see_also).lstrip()


for f in [voigt_nobg, voigt, double_voigt_nobg, double_voigt]:
    f.__doc__ = f.__doc__.replace('${SINGLE_VOIGT}', __single_voigt.lstrip())
    f.__doc__ = f.__doc__.replace('${DOUBLE_VOIGT}', __double_voigt.lstrip())
    f.__doc__ = f.__doc__.replace('${BACKGROUND}', __background.lstrip())
    f.__doc__ = f.__doc__.replace('${SEE_ALSO}', __see_also(f).lstrip())

del __input_x
del __single_voigt
del __double_voigt
del __background
del __see_also


def voigt_approx_nobg(*args, **kwargs):
    # For backwards compatibility
    return voigt_nobg(*args, impl=voigt_mclean, **kwargs)


def voigt_approx(*args, **kwargs):
    # For backwards compatibility
    return voigt(*args, impl=voigt_mclean, **kwargs)


def double_voigt_approx_nobg(*args, **kwargs):
    # For backwards compatibility
    return double_voigt_nobg(*args, impl=voigt_mclean, **kwargs)


def double_voigt_approx(*args, **kwargs):
    # For backwards compatibility
    return double_voigt(*args, impl=voigt_mclean, **kwargs)
