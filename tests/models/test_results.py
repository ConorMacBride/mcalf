import pytest

from mcalf.models.base import ModelBase as DummyModel
from mcalf.models.results import FitResult, FitResults

fitted_parameters = [1, 2, 1000.2, 1001.8, 5]
fit_info = {'chi2': 1.4, 'classification': 2, 'profile': 'abc',
            'success': True, 'index': [123, 456, 789]}


def test_fitresult_passthrough():
    fit = FitResult(fitted_parameters, fit_info)
    assert fit.parameters == [1, 2, 1000.2, 1001.8, 5]
    assert fit.chi2 == 1.4
    assert fit.classification == 2
    assert fit.profile == 'abc'
    assert isinstance(fit.success, bool) and fit.success
    assert fit.index == [123, 456, 789]


def test_fitresult_velocity():
    m = DummyModel()
    m.stationary_line_core = 1000.5
    m.quiescent_wavelength = 2
    m.active_wavelength = 3
    fit = FitResult(fitted_parameters, fit_info)
    assert fit.velocity(m, vtype='quiescent') == pytest.approx(-89.95502249)
    assert fit.velocity(m, vtype='active') == pytest.approx(389.80509745)

    # Ensure nan is returned if no active component fitted
    fitted_parameters[3] = float('nan')
    fit = FitResult(fitted_parameters, fit_info)
    vel = fit.velocity(m, vtype='active')
    assert vel != vel  # assert is nan
