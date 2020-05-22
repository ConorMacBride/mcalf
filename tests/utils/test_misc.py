import pytest

from mcalf.utils.misc import make_iter, load_parameter


def test_make_iter():
    i = [(1, 2, 3), (4,), [9, "a"], [23], "abc"]
    assert make_iter(*i) == i
    assert make_iter(1, [2], 3) == [[1], [2], [3]]


def test_load_parameter():
    # Need to test loading files
    i = "  [  42 ,  9.2, 6 +wl - 2.3, 12.5 + wl,wl, inf, -inf, inf+   4., 5.2-inf] "
    inf = float('inf')
    truth = [42.0, 9.2, 6.9, 15.7, 3.2, inf, -inf, inf, -inf]
    assert load_parameter(i, wl=3.2) == pytest.approx(truth)
