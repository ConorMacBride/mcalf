import pytest
import numpy as np

from mcalf.visualisation import plot_spectrum
from ..helpers import figure_test


@pytest.fixture(scope='function')
def wavelengths_spectrum():
    wavelengths = np.linspace(-1, 1, 30)
    wavelengths = np.delete(wavelengths, [1, 3, 5, 7, -8, -6, -4, -2])
    spectrum = -4.523 * np.exp(-wavelengths ** 2 / 0.05) + 11.354
    wavelengths += 2432.249
    return wavelengths, spectrum


@figure_test
def test_plot_spectrum(pytestconfig, wavelengths_spectrum):
    wavelengths, spectrum = wavelengths_spectrum
    plot_spectrum(wavelengths, spectrum)


@figure_test
def test_plot_spectrum_no_smooth(pytestconfig, wavelengths_spectrum):
    wavelengths, spectrum = wavelengths_spectrum
    plot_spectrum(wavelengths, spectrum, smooth=False)


@figure_test
def test_plot_spectrum_no_norm(pytestconfig, wavelengths_spectrum):
    wavelengths, spectrum = wavelengths_spectrum
    plot_spectrum(wavelengths, spectrum, normalised=False)


@figure_test
def test_plot_spectrum_no_smooth_norm(pytestconfig, wavelengths_spectrum):
    wavelengths, spectrum = wavelengths_spectrum
    plot_spectrum(wavelengths, spectrum, smooth=False, normalised=False)
