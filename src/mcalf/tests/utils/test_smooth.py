import pytest
import numpy as np

from mcalf.utils.smooth import moving_average, gaussian_kern_3d, smooth_cube, mask_classifications
from ..helpers import class_map


def test_moving_average():

    x = np.array([0.4, 1.2, 5.4, 8, 1.47532, 23.42, 63, 21, 14.75, 6, 2.64, 0.142])

    res = moving_average(x, 2)
    assert res == pytest.approx(np.array([0.4, 0.8, 3.3, 6.7, 4.73766, 12.44766, 43.21, 42., 17.875, 10.375, 4.32,
                                          1.391]))

    res = moving_average(x, 3)
    assert res == pytest.approx(np.array([0.8, 2.33333333, 4.86666667, 4.95844, 10.96510667, 29.29844, 35.80666667,
                                          32.91666667, 13.91666667, 7.79666667, 2.92733333, 1.391]))

    res = moving_average(x, 5)
    assert res == pytest.approx(np.array([2.33333333, 3.75, 3.295064, 7.899064, 20.259064, 23.379064, 24.729064,
                                          25.634, 21.478, 8.9064, 5.883, 2.92733333]))

    res = moving_average(x, 12)
    assert res == pytest.approx(np.array([6.64922, 14.69933143, 15.486915, 15.40503556, 14.464532, 13.38957455,
                                          12.28561, 13.36612, 14.582732, 15.60303556, 16.553415, 18.70742857]))

    for w in [3.5, 0, -3, 13]:  # Test invalid widths
        with pytest.raises(ValueError):
            moving_average(x, w)


def test_gaussian_kern_3d():

    # With default parameters of width=5 and sigma=(1, 1, 1)
    res = gaussian_kern_3d()
    truth = np.array([[[0.22313016, 0.32465247, 0.36787944, 0.32465247, 0.22313016],
                       [0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247],
                       [0.36787944, 0.53526143, 0.60653066, 0.53526143, 0.36787944],
                       [0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247],
                       [0.22313016, 0.32465247, 0.36787944, 0.32465247, 0.22313016]],

                      [[0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247],
                       [0.47236655, 0.68728928, 0.77880078, 0.68728928, 0.47236655],
                       [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
                       [0.47236655, 0.68728928, 0.77880078, 0.68728928, 0.47236655],
                       [0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247]],

                      [[0.36787944, 0.53526143, 0.60653066, 0.53526143, 0.36787944],
                       [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
                       [0.60653066, 0.8824969, 1., 0.8824969, 0.60653066],
                       [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
                       [0.36787944, 0.53526143, 0.60653066, 0.53526143, 0.36787944]],

                      [[0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247],
                       [0.47236655, 0.68728928, 0.77880078, 0.68728928, 0.47236655],
                       [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
                       [0.47236655, 0.68728928, 0.77880078, 0.68728928, 0.47236655],
                       [0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247]],

                      [[0.22313016, 0.32465247, 0.36787944, 0.32465247, 0.22313016],
                       [0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247],
                       [0.36787944, 0.53526143, 0.60653066, 0.53526143, 0.36787944],
                       [0.32465247, 0.47236655, 0.53526143, 0.47236655, 0.32465247],
                       [0.22313016, 0.32465247, 0.36787944, 0.32465247, 0.22313016]]])
    assert res == pytest.approx(truth)

    res = gaussian_kern_3d(width=3, sigma=(1.5, 0.7, 0.9))
    truth = np.array([[[0.15568597, 0.28862403, 0.15568597],
                       [0.19442824, 0.36044779, 0.19442824],
                       [0.15568597, 0.28862403, 0.15568597]],

                      [[0.43192377, 0.8007374, 0.43192377],
                       [0.53940751, 1., 0.53940751],
                       [0.43192377, 0.8007374, 0.43192377]],

                      [[0.15568597, 0.28862403, 0.15568597],
                       [0.19442824, 0.36044779, 0.19442824],
                       [0.15568597, 0.28862403, 0.15568597]]])
    assert res == pytest.approx(truth)


def test_smooth_cube():
    np.random.seed(0)  # Produce identical results
    cube = np.random.rand(5, 5, 5) * 100 - 50
    mask = np.array([[1, 1, 1, 1, 0],
                    [0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 0],
                    [1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 1]], dtype=int)
    res = smooth_cube(cube, mask, width=2, sigma=(1.2, 0.6, 1.4))
    truth = np.array([[[4.31830133e+00, 1.02965975e+01, 1.91043418e+01, 1.76588896e-02, np.nan],
                       [np.nan, 9.91998647e+00, np.nan, 1.33945458e+01, -2.98884340e+01],
                       [np.nan, np.nan, 2.64095719e+01, 3.51530895e+01, np.nan],
                       [-1.81095221e+01, -5.15689778e+00, 3.61023714e+00, np.nan, 4.69101513e-01],
                       [np.nan, -1.00013304e+01, -7.84092032e+00, -1.05514319e+01, -2.59007402e+01]],

                      [[3.65691013e+00, 1.57056595e+01, 9.86134349e+00, -1.79691126e+01, np.nan],
                       [np.nan, 1.75151400e+01, np.nan, -1.03351476e+01, -3.68392304e+01],
                       [np.nan, np.nan, 2.49878480e+00, -3.88009617e+00, np.nan],
                       [-5.57846637e+00, -2.03151495e+00, -2.98843786e+00, np.nan, -3.35401316e+00],
                       [np.nan, -5.38197129e+00, 6.49031413e-01, 5.81205525e-01, 5.14871752e+00]],

                      [[-1.00305940e+01, -1.71083008e+00, -5.57436167e+00, -1.05334176e+01, np.nan],
                       [np.nan, -4.55896449e+00, np.nan, -5.26767691e+00, -9.44864769e+00],
                       [np.nan, np.nan, -2.17783552e+01, -2.25091513e+01, np.nan],
                       [3.84769782e+00, -2.88330601e+00, -5.67411131e+00, np.nan, -2.17634111e+01],
                       [np.nan, 1.30081927e+00, 1.07663546e+01, 4.44361511e+00, -1.28020472e+01]],

                      [[-1.20645968e+01, -9.75815925e+00, 4.87884633e-01, 1.15538827e+01, np.nan],
                       [np.nan, -5.00688220e+00, np.nan, 5.13812774e+00, 2.59675233e+01],
                       [np.nan, np.nan, -1.03354339e+01, -3.61697176e+00, np.nan],
                       [5.85709312e+00, 4.07016012e+00, 2.70320241e+00, np.nan, -1.47377948e+01],
                       [np.nan, 1.60071244e+00, 1.12280352e+01, -2.46298117e+00, -2.85724738e+01]],

                      [[1.32888138e+00, 3.24146422e+00, 1.40154733e+01, 2.12673063e+01, np.nan],
                       [np.nan, 1.45760603e+01, np.nan, -8.91080166e-01, 4.52749012e+01],
                       [np.nan, np.nan, 2.60630329e+00, -5.01572953e-01, np.nan],
                       [9.29777733e+00, 2.29946022e+01, 2.27115569e+01, np.nan, 5.81933193e+00],
                       [np.nan, 2.28704008e+01, 3.00036917e+01, 3.39226239e+00, -7.61449514e+00]]])
    assert res == pytest.approx(truth, nan_ok=True)


def test_mask_classifications():

    with pytest.raises(TypeError) as e:
        mask_classifications([[0, 1], [1, 2]])
    assert '`class_map` must be a numpy.ndarray' in str(e.value)

    c = class_map(4, 5, 3, 5)  # t y x n

    with pytest.raises(ValueError) as e:
        mask_classifications(c[0, 0])
    assert '`class_map` must have either 2 or 3 dimensions, got 1' in str(e.value)

    with pytest.raises(TypeError) as e:
        mask_classifications(c.astype(float))
    assert '`class_map` must be an array of integers' in str(e.value)

    with pytest.raises(TypeError) as e:
        mask_classifications(c, vmax=3.5)
    assert '`vmax` must be an integer' in str(e.value)

    with pytest.raises(ValueError) as e:
        mask_classifications(c, vmax=-2)
    assert '`vmax` must not be less than zero' in str(e.value)

    with pytest.raises(TypeError) as e:
        mask_classifications(c, vmin=3.5)
    assert '`vmin` must be an integer' in str(e.value)

    with pytest.raises(ValueError) as e:
        mask_classifications(c, vmin=-2)
    assert '`vmin` must not be less than zero' in str(e.value)

    # vmin above vmax
    with pytest.raises(ValueError) as e:
        mask_classifications(c, vmin=3, vmax=1)
    assert '`vmin` must be less than `vmax`' in str(e.value)

    # no processing needed: 2D with original range
    assert np.array_equal(mask_classifications(class_map(1, 5, 3, 5)[0])[0], class_map(1, 5, 3, 5)[0])

    # no processing requested: 3D with original range
    assert np.array_equal(mask_classifications(class_map(4, 5, 3, 5), reduce=False)[0], class_map(4, 5, 3, 5))

    # test vmin and vmax calculated correctly
    c = class_map(4, 5, 3, 6)  # t y x n
    c[c == 0] = -1  # move all classification 0 -> -1
    assert mask_classifications(c)[1:3] == (1, 5)

    # test vmin and vmax used correctly
    truth = np.array([[1, -1, 1, -1],
                      [1, 2, -1, 2],
                      [2, -1, -1, -1]], dtype=int)
    res = mask_classifications(class_map(1, 4, 3, 4)[0], vmin=1, vmax=2)[0]
    assert np.array_equal(res, truth)

    # test average calculated correctly
    truth = np.array([[1, 2, 0, 3],
                      [1, 0, 0, 2],
                      [2, 0, -1, 1]], dtype=int)
    res = mask_classifications(class_map(3, 4, 3, 4))[0]
    assert np.array_equal(res, truth)

    # test all negative
    c = np.full((4, 6), -1, dtype=int)
    assert np.array_equal(c, mask_classifications(c)[0])
    c = np.full((3, 4, 6), -1, dtype=int)
    assert np.array_equal(c[0], mask_classifications(c)[0])
    c = np.full((3, 4, 6), -1, dtype=int)
    assert np.array_equal(c, mask_classifications(c, reduce=False)[0])
