import pytest
import numpy as np

from mcalf.models.ibis import IBIS8542Model
from mcalf.profiles.voigt import voigt


def test_ibis8542model_basic():
    # Will break if default parameters are changes in mcalf.models.ibis.IBIS8542Model
    wl = 1000.97
    x_orig = np.linspace(999.81, 1002.13, num=25)
    prefilter_main = 1 - np.abs(x_orig - wl) * 0.1
    prefilter_wvscl = x_orig - wl
    m = IBIS8542Model(stationary_line_core=wl, original_wavelengths=x_orig,
                      prefilter_ref_main=prefilter_main, prefilter_ref_wvscl=prefilter_wvscl)

    bg = 1327.243
    arr = voigt(x_orig, -231.42, wl+0.05, 0.2, 0.21, bg)

    m.load_array(np.array([arr, arr]), names=['row', 'wavelength'])
    m.load_background(np.array([bg, bg]), names=['row'])

    # Fit with assumed neural network classification
    fit1, fit2 = m.fit(row=[0, 1], classifications=np.array([0, 0]))
    fit3 = m.fit_spectrum(arr, classifications=0, background=bg)  # Make sure this is equiv
    truth = [-215.08275199, 1001.01035476, 0.00477063, 0.28869302]

    assert np.array_equal(fit1.parameters, fit2.parameters)
    assert np.array_equal(fit1.parameters, fit3.parameters)
    assert list(fit1.parameters) == pytest.approx(truth)
