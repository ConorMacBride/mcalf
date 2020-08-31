import pytest
import os
import numpy as np

from mcalf.utils.misc import make_iter, load_parameter


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

    abspath = f"{os.path.dirname(os.path.abspath(__file__))}{os.path.sep}data{os.path.sep}"
    truth = np.array([243.23, 62, 2523.43, 0, -62.2, np.inf, np.nan], dtype=np.float64)

    # Unsupported extension given
    with pytest.raises(ValueError):
        load_parameter(f"{abspath}test_load_parameter_file.unsupported")

    # Test each file type
    for ext in ['npy', 'csv', 'fits', 'sav']:
        res = load_parameter(f"{abspath}test_load_parameter_file.{ext}")
        assert res[:-1] == pytest.approx(truth[:-1])
        assert np.isnan(res[-1])

    # Test CSV with defined wl
    res = load_parameter(f"{abspath}test_load_parameter_file_wl.csv", wl=2523.43)
    assert res[:-1] == pytest.approx(truth[:-1])
    assert np.isnan(res[-1])

    # Test CSV with expected exceptions
    for e in ['typeerror', 'syntaxerror']:
        with pytest.raises(SyntaxError):  # Note: the TypeError is converted into a SyntaxError
            res = load_parameter(f"{abspath}test_load_parameter_file_{e}.csv", wl=2523.43)
