import pytest
import numpy as np

from mcalf.models import ModelBase as DummyModel
from mcalf.utils.spec import reinterpolate_spectrum, normalise_spectrum, generate_sigma


def test_reinterpolate_spectrum():
    np.random.seed(0)  # Produce identical results
    x_orig = np.cumsum((np.random.rand(30) * 2) + 1) - 30
    y = np.random.rand(len(x_orig))
    x_const = np.linspace(-20, 20, 15)
    res = reinterpolate_spectrum(y, x_orig, x_const)
    truth = [0.17333763, 0.59316857, 0.55491608, 0.75624398, 0.84520129,
             0.40598020, 0.47206980, 0.40619915, 0.55777817, 0.08496342,
             0.19650506, 0.32022676, 0.47101510, 0.45437529, 0.85481038]
    assert res == pytest.approx(truth)


def test_normalise_spectrum():
    x_orig = np.linspace(-20, 20, 15)
    x_const = x_orig * 0.9
    np.random.seed(0)  # Produce identical results
    y = (np.random.rand(len(x_orig)) * 100) - 50
    prefilter = 1 - (np.random.rand(len(x_orig)) * 0.3)
    res = normalise_spectrum(y, original_wavelengths=x_orig, constant_wavelengths=x_const,
                             prefilter_response=prefilter)
    truth = [0.40058047, 0.29610272, 0.24341921, 0.03779960, 0.07715727,
             0.32041000, 0.00441682, 0.71697103, 1.00000000, 0.00000000,
             0.38235223, 0.35785920, 0.02479681, 0.48465846, 0.68644184]
    assert res == pytest.approx(truth)

    with pytest.raises(ValueError):
        normalise_spectrum(y, original_wavelengths=x_orig)
    with pytest.raises(ValueError):
        normalise_spectrum(y, constant_wavelengths=x_const)


def test_normalise_spectrum_model():
    m = DummyModel(original_wavelengths=[0.0, 0.1])
    m.original_wavelengths = np.linspace(-20, 20, 15)
    m.constant_wavelengths = m.original_wavelengths * 0.9
    np.random.seed(0)  # Produce identical results
    y = (np.random.rand(len(m.original_wavelengths)) * 100) - 50
    m.prefilter_response = 1 - (np.random.rand(len(m.original_wavelengths)) * 0.3)
    res = normalise_spectrum(y, model=m)
    truth = [0.40058047, 0.29610272, 0.24341921, 0.03779960, 0.07715727,
             0.32041000, 0.00441682, 0.71697103, 1.00000000, 0.00000000,
             0.38235223, 0.35785920, 0.02479681, 0.48465846, 0.68644184]
    assert res == pytest.approx(truth)


def test_generate_sigma():
    # Will fail if default parameters of mcalf.utils.spec.generate_sigma change

    res = generate_sigma(1, np.linspace(6.21, 7.24, num=25), 6.743)[::3]
    truth = [0.99999996, 0.96730289, 0.14147828, 0.14147828, 0.14147828,
             0.14147828, 0.14147828, 0.77445118, 0.99999459]
    assert res == pytest.approx(truth)

    res = generate_sigma(2, np.linspace(6.21, 7.24, num=25), 6.743)[::3]
    truth = [0.99999996, 0.96730289, 0.14147828, 0.40000000, 0.40000000,
             0.40000000, 0.14147828, 0.77445118, 0.99999459]
    assert res == pytest.approx(truth)
