import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from mcalf.visualisation import plot_classifications, bar, plot_class_map, init_class_data
from ..helpers import figure_test, class_map


def spectra(n=20, w=10, low=0, high=6):
    """Generate an array y of n spectra each with w wavelength points
    along the second dimension. Array l of length n has random integers
    from [low, high)."""
    np.random.seed(4321)
    a, b, s = np.random.rand(3, n, 1)

    a = a - 0.5

    width_deviation = w / 5
    b = width_deviation * b - (width_deviation / 2) + w // 2

    s = 2 * s + 2

    x = np.array([np.arange(w)])

    y = a * np.exp(- (x - b) ** 2 / (2 * s ** 2)) + 0.5

    l = np.random.randint(low, high, len(y))

    return y, l


def test_plot_classifications():

    with pytest.raises(TypeError) as e:
        plot_classifications([[0, 1], [1, 2]], np.array([3, 4]))
    assert '`spectra` must be a numpy.ndarray' in str(e.value)

    with pytest.raises(TypeError) as e:
        plot_classifications(np.array([[0, 1], [1, 2]]), [3, 4])
    assert '`labels` must be a numpy.ndarray' in str(e.value)

    with pytest.raises(TypeError) as e:
        plot_classifications(np.array([0, 1]), np.array([3, 4]))
    assert '`spectra` must be a 2D array' in str(e.value)

    for l in (np.array([[3, 4]]), np.array([3.5, 4])):
        with pytest.raises(TypeError) as e:
            plot_classifications(np.array([[0, 1], [1, 2]]), l)
        assert '`labels` must be a 1D array of integers' in str(e.value)

    with pytest.raises(ValueError) as e:
        plot_classifications(np.array([[0, 1], [1, 2]]), np.array([3, 4, 6]))
    assert '`spectra` and `labels` must be the same length' in str(e.value)

    s, l = spectra(10, 5, 0, 6)

    with pytest.raises(ValueError) as e:
        plot_classifications(s, l, nrows=3, ncols=2)
    assert 'Both `nrows` and `ncols` cannot be given together' in str(e.value)

    with pytest.raises(TypeError) as e:
        plot_classifications(s, l, nrows=2.5)
    assert '`nrows` must be an integer' in str(e.value)

    with pytest.raises(TypeError) as e:
        plot_classifications(s, l, ncols=3.5)
    assert '`ncols` must be an integer' in str(e.value)

    for nlines in (3.5, -3, 0):
        with pytest.raises(TypeError) as e:
            plot_classifications(s, l, nlines=nlines)
        assert '`nlines` must be a positive integer' in str(e.value)

    s, l = spectra(100, 5, 2, 7)  # 5 plots

    with pytest.raises(ValueError) as e:
        plot_classifications(s, l, nrows=6)  # 6x1, not 5x1
    assert '`nrows` is larger than it needs to be' in str(e.value)

    with pytest.raises(ValueError) as e:
        plot_classifications(s, l, ncols=4)  # 2x4, not 2x3
    assert '`ncols` is larger than it needs to be' in str(e.value)


@figure_test
def test_plot_classifications_5plots(pytestconfig):
    s, l = spectra(40, 30, 0, 5)  # 5 plots
    plot_classifications(s, l)


@figure_test
def test_plot_classifications_4plots(pytestconfig):
    s, l = spectra(40, 30, 1, 5)  # 4 plots
    plot_classifications(s, l)


@figure_test
def test_plot_classifications_3plots(pytestconfig):
    s, l = spectra(40, 30, 2, 5)  # 3 plots
    plot_classifications(s, l, show_labels=False)


@figure_test
def test_plot_classifications_1plot(pytestconfig):
    s, l = spectra(40, 30, 3, 4)  # 1 plot
    plot_classifications(s, l)


@figure_test
def test_plot_classifications_3x2(pytestconfig):
    s, l = spectra(40, 30, 0, 5)  # 5 plots
    plot_classifications(s, l, nrows=3)


@figure_test
def test_plot_classifications_4x1(pytestconfig):
    s, l = spectra(40, 30, 1, 5)  # 4 plots
    plot_classifications(s, l, ncols=1)


@figure_test
def test_plot_classifications_3x2(pytestconfig):
    s, l = spectra(40, 30, 2, 8)  # 6 plots
    plot_classifications(s, l, nrows=3)


@figure_test
def test_plot_classifications_not01(pytestconfig):
    s, l = spectra(40, 30, 3, 5)  # 2 plots
    s[l == 4] += 0.5  # classification 4 not in [0, 1]
    plot_classifications(s, l)


def test_no_args():
    # Custom errors as not required if data keyword argument given.

    with pytest.raises(TypeError) as e:
        bar()
    assert "bar() missing 1 required positional argument: 'class_map'" == str(e.value)

    with pytest.raises(TypeError) as e:
        plot_class_map()
    assert "plot_class_map() missing 1 required positional argument: 'class_map'" == str(e.value)


@figure_test
def test_bar(pytestconfig):
    m = class_map(30, 10, 20, 8)
    bar(m)


@figure_test
def test_bar_range(pytestconfig):
    m = class_map(30, 10, 20, 8)
    bar(m, vmin=2, vmax=4)


@figure_test
def test_bar_reduce(pytestconfig):
    m = class_map(30, 10, 20, 8)
    bar(m, reduce=False)


@figure_test
def test_bar_allbad(pytestconfig):
    m = class_map(30, 10, 20, 8)
    m[:] = -1
    bar(m)


@figure_test
def test_plot_class_map(pytestconfig):
    m = class_map(30, 10, 20, 8)
    plot_class_map(m)


@figure_test
def test_plot_class_map_nocolorbar(pytestconfig):
    m = class_map(30, 10, 20, 8)
    plot_class_map(m, show_colorbar=False)


@figure_test
def test_plot_class_map_range(pytestconfig):
    m = class_map(30, 10, 20, 8)
    plot_class_map(m, vmin=2, vmax=4)


@figure_test
def test_plot_class_map_units(pytestconfig):
    m = class_map(30, 10, 20, 8)
    plot_class_map(m, resolution=(2.5 * u.m, 3.5 * u.km), offset=(5, 10),
                   dimension=('dim x', 'dim y'))


@figure_test
def test_plot_class_map_allbad(pytestconfig):
    m = class_map(30, 10, 20, 8)
    m[:] = -1
    plot_class_map(m)


@figure_test
def test_plot_bar_and_class_map(pytestconfig):
    m = class_map(30, 10, 20, 8)
    fig, ax = plt.subplots(1, 2)
    data = init_class_data(m, ax=ax[0], colorbar_settings={'ax': ax, 'location': 'bottom'})
    plot_class_map(data=data, ax=ax[0])
    bar(data=data, ax=ax[1])
