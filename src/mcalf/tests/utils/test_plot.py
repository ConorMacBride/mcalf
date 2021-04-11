import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units

from mcalf.utils.plot import hide_existing_labels, calculate_axis_extent, calculate_extent, class_cmap
from ..helpers import figure_test


def test_hide_existing_labels():
    plot_settings = {
        'LineA': {'color': 'r', 'label': 'A'},
        'LineB': {'color': 'g', 'label': 'B'},
        'LineC': {'color': 'b', 'label': 'C'},
    }
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], **plot_settings['LineA'])
    ax.plot([0, 1], [1, 0], **plot_settings['LineB'])
    hide_existing_labels(plot_settings)
    plt.close()
    ret = [x['label'] for x in plot_settings.values()]
    assert ret == ['_A', '_B', 'C']


def test_calculate_axis_extent():
    # Invalid `res`
    with pytest.raises(TypeError) as e:
        calculate_axis_extent('1 m', 1000)
    assert '`resolution` values must be' in str(e.value)

    # Invalid `px`
    with pytest.raises(TypeError) as e:
        calculate_axis_extent(250.0, 1000.5)
    assert '`px` must be an integer' in str(e.value)

    # Invalid `offset`
    with pytest.raises(TypeError) as e:
        calculate_axis_extent(250.0, 1000, [10])
    assert '`offset` must be an float or integer' in str(e.value)

    # Default `unit` used
    _, _, un = calculate_axis_extent(250.0, 1000, 10)
    assert un == 'Mm'  # Will fail if default value is changed
    _, _, un = calculate_axis_extent(250.0, 1000, 10, 'testunit')
    assert un == 'testunit'  # Will fail if default value is changed

    # Unit extracted
    _, _, un = calculate_axis_extent(250.0 * astropy.units.kg, 1000, 10, 'testunit')
    assert un == '$\\mathrm{kg}$'

    # Test good values
    f, l, _ = calculate_axis_extent(3., 7, -3.5)
    assert -10.5 == pytest.approx(f)
    assert 10.5 == pytest.approx(l)
    f, l, _ = calculate_axis_extent(2., 2, 1)
    assert 2. == pytest.approx(f)
    assert 6. == pytest.approx(l)


def test_calculate_extent():

    # Resolution of None should return None
    assert calculate_extent((100, 200), None) is None

    # Shape is 2-tuple
    for a in ((1, 1, 1), np.array([1, 1])):
        with pytest.raises(TypeError) as e:
            calculate_extent(a, resolution=(1, 1), offset=(1, 1))
        assert '`shape` must be a tuple of length 2' in str(e.value)

    # Resolution is 2-tuple
    for a in ((1, 1, 1), np.array([1, 1])):
        with pytest.raises(TypeError) as e:
            calculate_extent((100, 200), resolution=a, offset=(1, 1))
        assert '`resolution` must be a tuple of length 2' in str(e.value)

    # Offset is 2-tuple
    for a in ((1, 1, 1), np.array([1, 1])):
        with pytest.raises(TypeError) as e:
            calculate_extent((100, 200), resolution=(1, 1), offset=a)
        assert '`offset` must be a tuple of length 2' in str(e.value)

    # Test good value
    l, r, b, t = calculate_extent((7, 2), (2., 3.), (1, -3.5))
    assert 2. == pytest.approx(l)
    assert 6. == pytest.approx(r)
    assert -10.5 == pytest.approx(b)
    assert 10.5 == pytest.approx(t)


def plot_helper_calculate_extent(*args, dimension=None):
    fig, ax = plt.subplots()
    calculate_extent(*args, ax=ax, dimension=dimension)
    x, y = ax.get_xlabel(), ax.get_ylabel()
    plt.close(fig)
    return x, y


def test_calculate_extent_ax():

    # Common args
    args = ((7, 2), (2., 3.), (1, -3.5))

    # Dimension is 2-tuple or 2-list
    for d in ((1, 1, 1), np.array([1, 1]), [1, 1, 1]):
        with pytest.raises(TypeError) as e:
            plot_helper_calculate_extent(*args, dimension=d)
        assert '`dimension` must be a tuple or list of length 2' in str(e.value)

    # Different for both
    x, y = plot_helper_calculate_extent(*args, dimension=('test / one', 'test / two'))
    assert 'test / one (Mm)' == x
    assert 'test / two (Mm)' == y

    # Default values
    x, y = plot_helper_calculate_extent(*args)
    assert 'x-axis (Mm)' == x
    assert 'y-axis (Mm)' == y

    # Same for both
    x, y = plot_helper_calculate_extent(*args, dimension='test / same')
    assert 'test / same (Mm)' == x
    assert 'test / same (Mm)' == y


def test_class_cmap():

    # Incorrect `n` type
    with pytest.raises(TypeError) as e:
        class_cmap('original', 2.5)
    assert '`n` must be an integer' in str(e.value)

    # Correct `n` type
    class_cmap('original', 4)


@figure_test
def test_class_cmap_plot(pytestconfig):

    params = (
        ('original', 3),
        ('original', 5),
        ('original', 6),
        ('jet', 4),
    )

    fig, axes = plt.subplots(len(params))

    for ax, (style, n) in zip(axes, params):
        gradient = np.arange(-1, n + 1)
        gradient = np.vstack((gradient, gradient))
        cmap = class_cmap(style, n)
        ax.imshow(gradient, vmin=-0.5, vmax=n - 0.5, cmap=cmap)
        ax.set(xticks=[], yticks=[])
