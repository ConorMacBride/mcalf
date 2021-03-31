import pytest
import numpy as np
import astropy.units as u

from mcalf.visualisation import plot_map
from ..helpers import figure_test


@pytest.fixture
def arr():
    np.random.seed(1234)
    a = np.random.rand(8, 6)
    a[3, 3] = np.nan
    a[5, 1] = np.nan
    return (a * 16) - 8


@pytest.fixture
def mask(arr):
    mask = np.full_like(arr, True, dtype=bool)
    mask[1:7, 2:] = False
    return mask


@pytest.fixture
def umbra_mask(arr):
    mask = np.full_like(arr, True, dtype=bool)
    mask[2:6, 2:5] = False
    mask[4, 2] = True
    mask[2, 4] = True
    return mask


@figure_test
def test_plot_map_basic(pytestconfig, arr):
    plot_map(arr)


@figure_test
def test_plot_map_mask(pytestconfig, arr, mask):
    plot_map(arr, mask)


@figure_test
def test_plot_map_umbra_mask(pytestconfig, arr, umbra_mask):
    plot_map(arr, umbra_mask=umbra_mask)


@figure_test
def test_plot_map_both_masks(pytestconfig, arr, mask, umbra_mask):
    plot_map(arr, mask, umbra_mask)


@figure_test
def test_plot_map_resolution(pytestconfig, arr):
    plot_map(arr, resolution=(2.5, 3 * u.kg / u.Hz))


@figure_test
def test_plot_map_resolution_offset(pytestconfig, arr):
    plot_map(arr, resolution=(2.5, 3 * u.kg / u.Hz), offset=(-3, -4))


@figure_test
def test_plot_map_resolution_offset_masks(pytestconfig, arr, mask, umbra_mask):
    plot_map(arr, mask, umbra_mask, resolution=(2.5, 3 * u.kg / u.Hz), offset=(-3, -4))


@figure_test
def test_plot_map_vmin(pytestconfig, arr):
    plot_map(arr, vmin=-4.5)


@figure_test
def test_plot_map_vmax(pytestconfig, arr):
    plot_map(arr, vmax=4.5)


@figure_test
def test_plot_map_unit(pytestconfig, arr):
    plot_map(arr, unit='test/unit')


@figure_test
def test_plot_map_unit_astropy(pytestconfig, arr):
    plot_map(arr, unit=(u.m / u.s))


@figure_test
def test_plot_map_unit_arr_astropy(pytestconfig, arr):
    plot_map(arr * u.m / u.s, unit='test/unit')


@figure_test
def test_plot_map_lw(pytestconfig, arr, umbra_mask):
    plot_map(arr, umbra_mask=umbra_mask, lw=5.)


@figure_test
def test_plot_map_colorbar(pytestconfig, arr):
    plot_map(arr, show_colorbar=False)


def test_plot_map_validate_arr(arr):
    for a in ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.array([arr, arr])):
        with pytest.raises(TypeError) as e:
            plot_map(a)
        assert '`arr` must be a numpy.ndarray with 2 dimensions' in str(e.value)


def test_plot_map_validate_masks(arr, mask, umbra_mask):

    # Mask is 2D array
    for a in ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.array([mask, mask])):
        with pytest.raises(TypeError) as e:
            plot_map(arr, a, umbra_mask)
        assert '`mask` must be a numpy.ndarray with 2 dimensions' in str(e.value)

    # Umbra mask is 2D array
    for a in ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.array([umbra_mask, umbra_mask])):
        with pytest.raises(TypeError) as e:
            plot_map(arr, mask, a)
        assert '`umbra_mask` must be a numpy.ndarray with 2 dimensions' in str(e.value)

    # Mask wrong shape
    with pytest.raises(ValueError) as e:
        plot_map(arr, np.vstack([mask, mask]), umbra_mask)
    assert '`mask` must be the same shape as `arr`' in str(e.value)

    # Umbra mask wrong shape
    with pytest.raises(ValueError) as e:
        plot_map(arr, mask, np.vstack([umbra_mask, umbra_mask]))
    assert '`umbra_mask` must be the same shape as `arr`' in str(e.value)
