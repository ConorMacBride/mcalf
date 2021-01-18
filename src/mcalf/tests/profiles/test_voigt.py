import pytest
import numpy as np

from mcalf.profiles.voigt import voigt_approx_nobg, voigt_approx, double_voigt_approx_nobg, double_voigt_approx, \
    voigt_nobg, voigt, double_voigt_nobg, double_voigt

pts1 = np.array([-4.245, -0.324, 0.243, 1.163, 1.739, 99.999])
params1 = [8.242, 0.20, 0.241, 0.129, 5.348, 1.228, 0.152, 0.213, 6.82]

pts2 = np.array([-4.153, -0.323, 0.243, 0.682, 1.739, 99.999])
params2 = [7.142, 0.14, 0.321, 0.274, -4.281, 0.842, 0.181, 0.127, 7.25]


def test_voigt_clib():

    # Test 1
    res = double_voigt(pts1, *params1, clib=True)
    truth = [6.84938173, 9.35261435, 16.43320347, 13.11756603, 8.41536644, 6.82007115]
    assert res == pytest.approx(truth)

    # Test 2
    res = double_voigt(pts2, *params2, clib=True)
    truth = [7.27727702, 10.12250135, 11.49218384, 5.12373520, 7.27801656, 7.25004487]
    assert res == pytest.approx(truth)


def test_voigt_noclib():

    # Test 1
    res = double_voigt(pts1, *params1, clib=False)
    truth = [6.84938217, 9.35261435, 16.43320410, 13.11756590, 8.41536450, 6.82007115]
    assert res == pytest.approx(truth)

    # Test 2
    res = double_voigt(pts2, *params2, clib=False)
    truth = [7.27727751, 10.12250135, 11.49217969, 5.12373519, 7.27801656, 7.25004487]
    assert res == pytest.approx(truth)


def test_voigt_approx():

    # Test 1
    res = double_voigt_approx(pts1, *params1)
    truth = [6.83815735, 8.10237525, 11.61976236, 11.73676887, 8.05422184, 6.82004642]
    assert res == pytest.approx(truth)

    # Test 2
    res = double_voigt_approx(pts2, *params2)
    truth = [7.28207577, 10.29607126, 11.99513939, 7.63595484, 7.41246856, 7.25005616]
    assert res == pytest.approx(truth)


def test_voigt_wrappers():

    for pts, params in ([pts1, params1], [pts2, params2]):

        params_absorption = params[:4]
        params_emission = params[4:-1]
        background = params[-1]

        base_absorption = voigt_nobg(pts, *params_absorption)
        base_emission = voigt_nobg(pts, *params_emission)

        assert all([a == b for a, b in zip(
            voigt(pts, *params_absorption, background),
            base_absorption + background
        )])
        assert all([a == b for a, b in zip(
            double_voigt_nobg(pts, *params_absorption, *params_emission),
            base_absorption + base_emission
        )])
        assert all([a == b for a, b in zip(
            double_voigt(pts, *params_absorption, *params_emission, background),
            base_absorption + base_emission + background
        )])

        base_absorption_approx = voigt_approx_nobg(pts, *params_absorption)
        base_emission_approx = voigt_approx_nobg(pts, *params_emission)

        assert all([a == b for a, b in zip(
            voigt_approx(pts, *params_absorption, background),
            base_absorption_approx + background
        )])
        assert all([a == b for a, b in zip(
            double_voigt_approx_nobg(pts, *params_absorption, *params_emission),
            base_absorption_approx + base_emission_approx
        )])
        assert all([a == b for a, b in zip(
            double_voigt_approx(pts, *params_absorption, *params_emission, background),
            base_absorption_approx + base_emission_approx + background
        )])
