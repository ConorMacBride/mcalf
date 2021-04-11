import pytest
import numpy as np
from astropy.io import fits

from mcalf.utils.misc import make_iter, load_parameter, merge_results

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
