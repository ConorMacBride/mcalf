import numpy as np
import scipy.interpolate
from mcalf.profiles.gaussian import single_gaussian


__all__ = ['reinterpolate_spectrum', 'normalise_spectrum', 'generate_sigma']


def reinterpolate_spectrum(spectrum, original_wavelengths, constant_wavelengths):
    """Reinterpolate the spectrum.

    Reinterpolates the spectrum such that intensities at `original_wavelengths` are transformed into
    intensities at `constant_wavelengths`. Uses :class:`scipy.interpolate.InterpolatedUnivariateSpline` to
    interpolate.

    Parameters
    ----------
    spectrum : numpy.ndarray, ndim=1
        Spectrum to reinterpolate.
    original_wavelengths : numpy.ndarray, ndim=1, length=length of `spectrum`
        Wavelengths of `spectrum`.
    constant_wavelengths : numpy.ndarray, ndim=1
        Wavelengths to cast `spectrum` into.

    Returns
    -------
    spectrum : numpy.ndarray, length=length of `constant_wavelengths`
        Reinterpolated spectrum.
    """
    s = scipy.interpolate.InterpolatedUnivariateSpline(original_wavelengths, spectrum)  # Fit spline
    return s(constant_wavelengths)  # Evaluate spline at new wavelengths


def normalise_spectrum(spectrum, original_wavelengths=None, constant_wavelengths=None, prefilter_response=None,
                       model=None):
    """Normalise an individual spectrum to have intensities in range [0, 1].

    .. warning::
        Not recommended for normalising many spectra in a loop.

    Parameters
    ----------
    spectrum : numpy.ndarray, ndim=1
        Spectrum to reinterpolate and normalise.
    original_wavelengths : numpy.ndarray, ndim=1, length=length of `spectrum`, optional
        Wavelengths of `spectrum`.
    constant_wavelengths : numpy.ndarray, ndim=1, optional
        Wavelengths to cast `spectrum` into.
    prefilter_response : numpy.ndarray, ndim=1, length=length of `constant_wavelengths`, optional
        Prefilter response to divide spectrum by.
    model : child class of mcalf.models.ModelBase, optional
        Model to extract the above parameters from.

    Returns
    -------
    spectrum : numpy.ndarray, ndim-1, length=length of `constant_wavelengths`
        The normalised spectrum.
    """
    from mcalf.models import ModelBase
    if issubclass(model.__class__, ModelBase):
        if original_wavelengths is None:
            original_wavelengths = model.original_wavelengths
        if constant_wavelengths is None:
            constant_wavelengths = model.constant_wavelengths
        if prefilter_response is None:
            prefilter_response = model.prefilter_response
    if original_wavelengths is not None or constant_wavelengths is not None:
        if original_wavelengths is None or constant_wavelengths is None:
            raise ValueError("original_wavelengths and constant_wavelengths must be provided")
        spectrum = reinterpolate_spectrum(spectrum, original_wavelengths, constant_wavelengths)
    if prefilter_response is not None:
        spectrum /= prefilter_response
    spectrum -= min(spectrum)
    spectrum /= max(spectrum)
    return spectrum


def generate_sigma(sigma_type, wavelengths, line_core, a=-0.95, c=0.04, d=1, centre_rad=7, a_peak=0.4):
    """Generate the default sigma profiles.

    Parameters
    ----------
    sigma_type : int
        Type of profile to generate. Should be either `1` or `2`.
    wavelengths : array_like
        Wavelengths to use for sigma profile.
    line_core : float
        Line core to use as centre of Gaussian sigma profile.
    a : float, optional, default=-0.95
        Amplitude of Gaussian sigma profile.
    c : float, optional, default=0.04
        Sigma of Gaussian sigma profile.
    d : float, optional, default=1
        Background of Gaussian sigma profile.
    centre_rad : int, optional, default=7
        Width of central flattened region.
    a_peak : float, optional, default=0.4
        Amplitude of central 7 pixel section, if `sigma_type` is 2.

    Returns
    -------
    sigma : numpy.ndarray, length=n_wavelengths
        The generated sigma profile.
    """
    sigma = single_gaussian(wavelengths, a, line_core, c, d)  # Define initial sigma profile
    centre_i = np.argmin(sigma)  # Get central index of sigma
    sigma[:centre_i - centre_rad] = sigma[centre_rad:centre_i]  # Shift left half left
    sigma[centre_i + centre_rad:] = sigma[centre_i:-centre_rad]  # Shift right half right
    sigma[centre_i - centre_rad:centre_i + centre_rad] = sigma[centre_i]  # Connect the two halves with a line
    if sigma_type == 2:
        sigma[centre_i - 3:centre_i + 3 + 1] = a_peak  # Lessen importance of peak detail
    return sigma
