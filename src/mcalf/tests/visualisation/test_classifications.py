import pytest
import numpy as np

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
