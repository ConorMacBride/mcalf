import pytest
import numpy as np

from mcalf.utils.mask import genmask, radial_distances


def test_radial_distances():
    res = radial_distances(4, 9)
    truth = np.array([[4.27200187, 4.03112887, 4.03112887, 4.27200187],
                      [3.35410197, 3.04138127, 3.04138127, 3.35410197],
                      [2.5, 2.06155281, 2.06155281, 2.5],
                      [1.80277564, 1.11803399, 1.11803399, 1.80277564],
                      [1.5, 0.5, 0.5, 1.5],
                      [1.80277564, 1.11803399, 1.11803399, 1.80277564],
                      [2.5, 2.06155281, 2.06155281, 2.5],
                      [3.35410197, 3.04138127, 3.04138127, 3.35410197],
                      [4.27200187, 4.03112887, 4.03112887, 4.27200187]])
    assert res == pytest.approx(truth)


def test_genmask():
    res = genmask(4, 9, radius=2.2)
    truth = np.array([[False, False, False, False],
                      [False, False, False, False],
                      [False, True, True, False],
                      [True, True, True, True],
                      [True, True, True, True],
                      [True, True, True, True],
                      [False, True, True, False],
                      [False, False, False, False],
                      [False, False, False, False]])
    assert np.sum(res == truth) == 4 * 9

    res = genmask(4, 9, radius=2.2, right_shift=1, up_shift=-2)
    truth = np.array([[False, False, True, True],
                      [True, True, True, True],
                      [True, True, True, True],
                      [True, True, True, True],
                      [False, False, True, True],
                      [False, False, False, False],
                      [False, False, False, False],
                      [False, False, False, False],
                      [False, False, False, False]])
    assert np.sum(res == truth) == 4 * 9
