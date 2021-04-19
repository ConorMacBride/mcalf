import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad, quad_vec

# Load the C library
import os.path
from pathlib import Path
import ctypes
# # Commands to manually generate
# gcc -Wall -fPIC -c voigt.c
# gcc -shared -o libvoigt.so voigt.o
dllabspath = Path(os.path.dirname(os.path.abspath(__file__)))  # Path to libraries directory
try:
    libfile = [str(i) for i in dllabspath.glob('ext_voigtlib.*.so')][0]  # Select first (and only) library
    lib = ctypes.CDLL(libfile)  # Load the library
    lib.func.restype = ctypes.c_double  # Specify the expected result type
    lib.func.argtypes = (ctypes.c_int, ctypes.c_double)  # Specify the type of the input parameters
    cvoigt = lib.func  # Create alias for the specific function used in functions below
except IndexError:  # File does not exist
    warnings.warn("Could not locate the external C library. Further use of `clib` will fail!")

###
# readthedocs.org does not support clib (force clib=False)
import os
not_on_rtd = os.environ.get('READTHEDOCS') != 'True'
rtd = {}
if not not_on_rtd:  # Reduce computation time (and accuracy) of no clib version
    rtd = {'epsabs': 1.49e-1, 'epsrel': 1.49e-4}
###

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
    """Voigt function (efficient approximation) with no background (Base approx. Voigt function).

    This is the base for all other approximated Voigt functions. Not implemented in any models yet as initial tests
    exhibited slow convergence.

    Parameters
    ----------
    ${SINGLE_VOIGT}

    ${EXTRA_APPROX}
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
    """Voigt function (efficient approximation) with background.

    Parameters
    ----------
    ${SINGLE_VOIGT}
    ${BACKGROUND}

    ${EXTRA_APPROX}
    """
    return voigt_approx_nobg(x, a, b, s, g) + d


def double_voigt_approx_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2):
    """Double Voigt function (efficient approximation) with no background.

    Parameters
    ----------
    ${DOUBLE_VOIGT}

    ${EXTRA_APPROX}
    """
    return voigt_approx_nobg(x, a1, b1, s1, g1) + voigt_approx_nobg(x, a2, b2, s2, g2)


def double_voigt_approx(x, a1, b1, s1, g1, a2, b2, s2, g2, d):
    """Double Voigt function (efficient approximation) with background.

    Parameters
    ----------
    ${DOUBLE_VOIGT}
    ${BACKGROUND}

    ${EXTRA_APPROX}
    """
    return voigt_approx_nobg(x, a1, b1, s1, g1) + voigt_approx_nobg(x, a2, b2, s2, g2) + d


def voigt_nobg(x, a, b, s, g, clib=True):
    """Voigt function with no background (Base Voigt function).

    This is the base of all the other Voigt functions.

    Parameters
    ----------
    ${SINGLE_VOIGT}
    ${CLIB}

    ${EXTRA_STD}
    """
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    u = x - b
    if clib and not_on_rtd:
        i = [quad(cvoigt, -np.inf, np.inf, args=(v, s, g), epsabs=1.49e-1, epsrel=1.49e-4)[0] for v in u]
    else:
        i = quad_vec(lambda y: np.exp(-y**2 / (2 * s**2)) / (g**2 + (u - y)**2), -np.inf, np.inf, **rtd)[0]
    const = g / (s * np.sqrt(2 * np.pi**3))
    return a * const * np.array(i)


def voigt(x, a, b, s, g, d, clib=True):
    """Voigt function with background.

    Parameters
    ----------
    ${SINGLE_VOIGT}
    ${BACKGROUND}
    ${CLIB}

    ${EXTRA_STD}
    """
    return voigt_nobg(x, a, b, s, g, clib) + d


def double_voigt_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2, clib=True):
    """Double Voigt function with no background.

    Parameters
    ----------
    ${DOUBLE_VOIGT}
    ${CLIB}

    ${EXTRA_STD}
    """
    return voigt_nobg(x, a1, b1, s1, g1, clib) + voigt_nobg(x, a2, b2, s2, g2, clib)


def double_voigt(x, a1, b1, s1, g1, a2, b2, s2, g2, d, clib=True):
    """Double Voigt function with background.

    Parameters
    ----------
    ${DOUBLE_VOIGT}
    ${BACKGROUND}
    ${CLIB}

    ${EXTRA_STD}
    """
    return double_voigt_nobg(x, a1, b1, s1, g1, a2, b2, s2, g2, clib) + d


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
__clib = """
    clib : bool, optional, default=True
        Whether to use the complied C library or a slower Python version. If using the C library, the accuracy
        of the integration is reduced to give the code a significant speed boost. Python version can be used when
        speed is not a priority. Python version will remove deviations that are sometimes present around the wings
        due to the reduced accuracy."""

# Define "Returns"
__returns = """
    Returns
    -------
    result : numpy.ndarray, shape=`x.shape`
        The value of the Voigt function here.

    """

# Define "Notes" (and "References") section
__notes_approx = """
    Notes
    -----
    This algorithm is taken from A. B. McLean et al. [1]_.

    References
    ----------
    .. [1] A. B. McLean, C. E. J. Mitchell and D. M. Swanston, "Implementation of an efficient analytical
      approximation to the Voigt function for photoemission lineshape analysis," Journal of Electron Spectroscopy and
      Related Phenomena, vol. 69, pp. 125-132, 1994. https://doi.org/10.1016/0368-2048(94)02189-7"""
__notes_std = """
    Notes
    -----
    More information on the Voigt function can be found here: https://en.wikipedia.org/wiki/Voigt_profile"""


# Define special "See Also" options
__see_also_approx = [
    '    voigt_approx_nobg : Base approximated Voigt function with no background.',
    '    voigt_approx : Approximated Voigt function with background added.',
    '    double_voigt_approx_nobg : Two approximated Voigt functions added together.',
    '    double_voigt_approx : Two approximated Voigt functions and a background added together.',
]
__see_also_std = [
    '    voigt_nobg : Base Voigt function with no background.',
    '    voigt : Voigt function with background added.',
    '    double_voigt_nobg : Two Voigt functions added together.',
    '    double_voigt : Two Voigt function and a background added together.',
]
# Extract the function name for easy exclusion of item
__see_also_approx = [(i.split(':')[0].strip(), i) for i in __see_also_approx]
__see_also_std = [(i.split(':')[0].strip(), i) for i in __see_also_std]
__see_also = __see_also_approx + __see_also_std


def __rm_self(func, items):
    """Return the "See Also" section with the current function removed."""
    ret = [i[1] for i in items if i[0] != func]
    return 'See Also\n    --------\n    ' + '\n'.join(ret).lstrip() + '\n'


def __extra_approx(func):
    """Merge common approx functions sections."""
    return __returns + __rm_self(func.__name__, __see_also) + __notes_approx


def __extra_std(func):
    """Merge common standard functions sections."""
    return __returns + __rm_self(func.__name__, __see_also_std) + __notes_std


for f in [voigt_approx_nobg, voigt_approx, double_voigt_approx_nobg, double_voigt_approx,
          voigt_nobg, voigt, double_voigt_nobg, double_voigt]:
    f.__doc__ = f.__doc__.replace('${SINGLE_VOIGT}', __single_voigt.lstrip())
    f.__doc__ = f.__doc__.replace('${DOUBLE_VOIGT}', __double_voigt.lstrip())
    f.__doc__ = f.__doc__.replace('${BACKGROUND}', __background.lstrip())
    f.__doc__ = f.__doc__.replace('${CLIB}', __clib.lstrip())
    f.__doc__ = f.__doc__.replace('${EXTRA_APPROX}', __extra_approx(f).lstrip())
    f.__doc__ = f.__doc__.replace('${EXTRA_STD}', __extra_std(f).lstrip())

del __input_x
del __single_voigt
del __double_voigt
del __background
del __clib
del __returns
del __notes_approx
del __notes_std
del __see_also_approx
del __see_also_std
del __see_also
del __rm_self
del __extra_approx
del __extra_std
