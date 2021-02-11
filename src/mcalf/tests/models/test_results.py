import pytest
import numpy as np

from mcalf.models import ModelBase as DummyModel, FitResult, FitResults

fitted_parameters = [1, 2, 1000.2, 1001.8, 5]
fit_info = {'chi2': 1.4, 'classification': 2, 'profile': 'abc',
            'success': True, 'index': [123, 456, 789]}


def test_fitresult_passthrough():
    fit = FitResult(fitted_parameters, fit_info)
    assert fit.parameters == [1, 2, 1000.2, 1001.8, 5]
    assert len(fit) == 5
    assert fit.chi2 == 1.4
    assert fit.classification == 2
    assert fit.profile == 'abc'
    assert isinstance(fit.success, bool) and fit.success
    assert fit.index == [123, 456, 789]

    # Test that the string representation can be formed without error
    repr(fit)
    fit.index = [None]*3
    repr(fit)


def test_fitresult_velocity():
    m = DummyModel(original_wavelengths=[1000.4, 1000.6])
    m.stationary_line_core = 1000.5
    m.quiescent_wavelength = 2
    m.active_wavelength = 3
    fit = FitResult(fitted_parameters, fit_info)
    assert fit.velocity(m, vtype='quiescent') == pytest.approx(-89.95502249)
    assert fit.velocity(m, vtype='active') == pytest.approx(389.80509745)

    # Ensure nan is returned if no active component fitted
    fitted_parameters_trim = fitted_parameters[:3]
    fit = FitResult(fitted_parameters_trim, fit_info)
    vel = fit.velocity(m, vtype='active')
    assert vel != vel  # assert is nan

    # Ensure an invalid velocity type is detected
    with pytest.raises(ValueError):
        vel = fit.velocity(m, vtype='unknown-vtype')


def test_fitresults_init():
    fits = FitResults((49, 52), 4, time=12)
    assert fits.parameters.shape == (49, 52, 4)
    assert fits.chi2.shape == (49, 52)
    assert fits.classifications.shape == (49, 52)
    assert fits.profile.shape == (49, 52)
    assert fits.success.shape == (49, 52)
    assert fits.time == 12

    with pytest.raises(TypeError):  # Should be a tuple
        fits = FitResults(10, 3)

    with pytest.raises(TypeError):  # Should be a tuple of length 2
        fits = FitResults((10, 32, 53), 8)

    with pytest.raises(ValueError):  # Should be an integer >= 1
        fits = FitResults((10, 5), 5.5)
    with pytest.raises(ValueError):  # Should be an integer >= 1
        fits = FitResults((10, 5), 0)


def test_fitresults_append():

    # Create dummy fit results
    fit1 = FitResult(
        [2, 6, 254.6, 963.4],
        {'chi2': 7.43, 'classification': 4, 'profile': 'absorption',
            'success': True, 'index': [12, 34, 81]}
    )
    fit2 = FitResult(
        [9, 2, 724.32, 134.8],
        {'chi2': 1.34, 'classification': 2, 'profile': 'emission',
            'success': True, 'index': [12, 0, 99]}
    )
    fit3 = FitResult(
        [1, 8, 932.1, 327.5, 3.7, 9, 2, 0.2],
        {'chi2': 0.79, 'classification': 1, 'profile': 'both',
            'success': False, 'index': [12, 99, 0]}
    )
    fit4 = FitResult(  # With incorrect time index
        [6, 4, 356.2, 738.5],
        {'chi2': 8.2, 'classification': 3, 'profile': 'absorption',
            'success': True, 'index': [3, 0, 25]}
    )
    fit5 = FitResult(  # With unknown profile name
        [5, 3, 256.2, 628.5],
        {'chi2': 8.1, 'classification': 3, 'profile': 'continuum',
            'success': True, 'index': [12, 10, 40]}
    )

    # Initialise empty FitResults object
    fits = FitResults((100, 100), 8, time=12)

    # Append dummy fits
    fits.append(fit1)
    fits.append(fit2)
    fits.append(fit3)

    with pytest.raises(ValueError):  # Time index does not match
        fits.append(fit4)

    with pytest.raises(ValueError):  # Unknown profile
        fits.append(fit5)

    assert all([a == b for a, b in zip(fits.parameters[34, 81][:4], fit1.parameters)])
    assert fits.chi2[34, 81] == fit1.chi2
    assert fits.classifications[34, 81] == fit1.classification
    assert fits.profile[34, 81] == fit1.profile
    assert fits.success[34, 81] == fit1.success

    assert all([a == b for a, b in zip(fits.parameters[0, 99][4:], fit2.parameters)])
    assert fits.chi2[0, 99] == fit2.chi2
    assert fits.classifications[0, 99] == fit2.classification
    assert fits.profile[0, 99] == fit2.profile
    assert fits.success[0, 99] == fit2.success

    assert all([a == b for a, b in zip(fits.parameters[99, 0], fit3.parameters)])
    assert fits.chi2[99, 0] == fit3.chi2
    assert fits.classifications[99, 0] == fit3.classification
    assert fits.profile[99, 0] == fit3.profile
    assert fits.success[99, 0] == fit3.success


def test_fitresults_velocities():
    m = DummyModel(original_wavelengths=[1000.4, 1000.6])
    m.stationary_line_core = 1000.5
    m.quiescent_wavelength = 0
    m.active_wavelength = 1
    fits = FitResults((4, 4), 2)
    fits.parameters = np.array([
        [[1000.2, 192.4], [826.5, 534.23], [8365.86, 1252.32], [1532.3, 2152.3]],
        [[978.73, 753.52], [1253.5, 1329.3], [6423.4, 2355.45], [12.53, 2523.3]],
        [[825.8, 862.5], [1759.5, 1000.9], [2633.4, 234.43], [2535.353, 152.34]],
        [[896.53, 153.2], [1224.3, 1111.11], [634.54, 2353.97], [242.35, 763.4]]
    ])

    truth_quiescent = np.array([[-8.99550225e+01, -5.21739130e+04, 2.20850375e+06, 1.59460270e+05],
                                [-6.52773613e+03, 7.58620690e+04, 1.62605697e+06, -2.96242879e+05],
                                [-5.23838081e+04, 2.27586207e+05, 4.89625187e+05, 4.60225787e+05],
                                [-3.11754123e+04, 6.71064468e+04, -1.09733133e+05, -2.27331334e+05]])

    truth_active = np.array([[-2.42308846e+05, -1.39811094e+05, 7.55082459e+04, 3.45367316e+05],
                             [-7.40569715e+04, 9.85907046e+04, 4.06281859e+05, 4.56611694e+05],
                             [-4.13793103e+04, 1.19940030e+02, -2.29706147e+05, -2.54320840e+05],
                             [-2.54062969e+05, 3.31664168e+04, 4.05838081e+05, -7.10944528e+04]])

    result_quiescent = fits.velocities(m, vtype='quiescent')
    result_active = fits.velocities(m, vtype='active')

    with pytest.raises(ValueError):
        fits.velocities(m, vtype='unknown-vtype')

    assert result_quiescent == pytest.approx(truth_quiescent)
    assert result_active == pytest.approx(truth_active)
