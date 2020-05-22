import numpy as np
from scipy.special import erf


__all__ = ['single_gaussian', 'skew_normal', 'skew_normal_with_gaussian']


def single_gaussian(x, a, b, c, d):
    """Gaussian function

    Parameters
    ----------
    x : ndarray
        Wavelengths to evaluate Gaussian function at.
    a : float
        Amplitude.
    b : float
        Central line core.
    c : float
        Sigma of Gaussian.
    d : float
        Background to add.
    """
    g = a * np.exp(- (x-b)**2.0 / (2.0 * c**2.0))
    return g + d


def _skew_normal_with_gaussian(x, a_a=0, alpha=1, xi=0, omega=1, a_e=0, b=0, c=1, d=0):
    x1 = (x - xi) / omega
    pdf = np.exp(- x1 ** 2.0 / 2.0) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + erf(alpha * x1 / np.sqrt(2.0)))
    skewed_normal = 2.0 * pdf * cdf / omega

    absorption = a_a * skewed_normal
    emission = a_e * np.exp(- (x - b) ** 2.0 / (2.0 * c ** 2.0))

    return absorption + emission + d


def skew_normal(x, a_a, alpha, xi, omega, d):
    return _skew_normal_with_gaussian(x, a_a=a_a, alpha=alpha, xi=xi, omega=omega, d=d)


def skew_normal_with_gaussian(x, a_a, alpha, xi, omega, a_e, b, c, d):  # This can be optimised later
    return _skew_normal_with_gaussian(x, a_a=a_a, alpha=alpha, xi=xi, omega=omega, a_e=a_e, b=b, c=c, d=d)
