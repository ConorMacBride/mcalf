import pytest
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units

from mcalf.utils.misc import make_iter, load_parameter, merge_results, hide_existing_labels, \
    calculate_axis_extent, calculate_extent

from ..helpers import data_path_function
data_path = data_path_function('utils')


def test_make_iter():
    i = [(1, 2, 3), (4,), [9, "a"], [23], "abc"]
    assert make_iter(*i) == i
    assert make_iter(1, [2], 3) == [[1], [2], [3]]


def test_load_parameter_string():
    # Need to test loading files
    i = "  [  42 ,  9.2, 6 +wl - 2.3, 12.5 + wl,wl, inf, -inf, inf+   4., 5.2-inf] "
    inf = float('inf')
    truth = [42.0, 9.2, 6.9, 15.7, 3.2, inf, -inf, inf, -inf]
    assert load_parameter(i, wl=3.2) == pytest.approx(truth)

    # Everything should be converted to floats
    assert isinstance(load_parameter("wl", wl=int(5)), float)


def test_load_parameter_file():

    truth = np.array([243.23, 62, 2523.43, 0, -62.2, np.inf, np.nan], dtype=np.float64)

    # Unsupported extension given
    with pytest.raises(ValueError):
        load_parameter(data_path('test_load_parameter_file.unsupported'))

    # Test each file type
    for ext in ['npy', 'csv', 'fits', 'sav']:
        res = load_parameter(data_path(f'test_load_parameter_file.{ext}'))
        assert res[:-1] == pytest.approx(truth[:-1])
        assert np.isnan(res[-1])

    # Test CSV with defined wl
    res = load_parameter(data_path('test_load_parameter_file_wl.csv'), wl=2523.43)
    assert res[:-1] == pytest.approx(truth[:-1])
    assert np.isnan(res[-1])

    # Test CSV with expected exceptions
    for e in ['typeerror', 'syntaxerror']:
        with pytest.raises(SyntaxError):  # Note: the TypeError is converted into a SyntaxError
            load_parameter(data_path(f'test_load_parameter_file_{e}.csv'), wl=2523.43)


def test_merge_results(tmp_path):

    # Compatible files to test merging
    compatible_files = [
        data_path('test_merge_results_1.fits'),
        data_path('test_merge_results_2.fits'),
        data_path('test_merge_results_3.fits'),
    ]

    # Merge and save
    output_file = tmp_path / "test_merge_results_output.fits"
    merge_results(compatible_files, output_file)

    # Compare merged files to expected merge
    test = fits.open(output_file, mode='readonly')
    verify = fits.open(data_path('test_merge_results_all.fits'), mode='readonly')
    # Diff ignoring checksums as too strict (compare values instead)
    diff = fits.FITSDiff(test, verify, ignore_keywords=['CHECKSUM', 'DATASUM'])
    assert diff.identical  # If this fails tolerances *may* need to be adjusted

    # Incompatible files to test merging
    incompatible_files = [
        data_path('test_merge_results_1.fits'),
        data_path('test_merge_results_2.fits'),
        data_path('test_merge_results_3.fits'),
        data_path('test_merge_results_2.fits'),  # Duplicate (overlapping) file should fail
    ]

    # Merge (should fail before saving)
    with pytest.raises(ValueError):
        merge_results(incompatible_files, output_file)

    # Compatible files but wrong time for one (should fail)
    compatible_files_wrongtime = [
        data_path('test_merge_results_1.fits'),
        data_path('test_merge_results_2_wrongtime.fits'),
        data_path('test_merge_results_3.fits'),
    ]

    # Merge (should fail before saving)
    with pytest.raises(ValueError):
        merge_results(compatible_files_wrongtime, output_file)

    # Must provide a list of multiple files
    with pytest.raises(TypeError):  # single string
        merge_results(data_path('test_merge_results_1.fits'), output_file)
    with pytest.raises(TypeError):  # list of length 1
        merge_results([data_path('test_merge_results_1.fits')], output_file)

    # The extra HDU should cause an error
    with pytest.raises(ValueError) as excinfo:
        merge_results([
            data_path('test_merge_results_2.fits'),
            data_path('test_merge_results_1_extrahdu.fits'),
        ], output_file)
    assert 'nexpected' in str(excinfo.value)  # "Unexpected"
    # reverse (now extra is in first file)
    with pytest.raises(ValueError) as excinfo:
        merge_results([
            data_path('test_merge_results_1_extrahdu.fits'),
            data_path('test_merge_results_2.fits'),
        ], output_file)
    assert 'nexpected' in str(excinfo.value)  # "Unexpected"


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
