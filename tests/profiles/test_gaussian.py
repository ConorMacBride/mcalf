import pytest
import numpy as np

from mcalf.profiles.gaussian import single_gaussian


def test_gaussian():

    # # Gaussian function test 1

    pts = np.array([-4.245, -0.324, 0.243, 1.163, 1.739, 99.999])
    params = [-6.342, 0.62, 0.541, 3.21]
    res = single_gaussian(pts, *params)
    truth = [3.21000000, 1.82620902, -1.76481841, -0.62239711, 2.46317790, 3.21000000]
    assert res == pytest.approx(truth)

    # # Gaussian function test 2

    pts = np.array([-4.153, -0.323, 0.243, 0.682, 1.739, 99.999])
    params = [7.142, 0.14, 0.321, 7.25]
    res = single_gaussian(pts, *params)
    truth = [7.25000000, 9.77383785, 14.03363668, 8.96690188, 7.25002922, 7.25000000]
    assert res == pytest.approx(truth)
