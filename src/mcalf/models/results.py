import numpy as np

from mcalf.utils.misc import make_iter


__all__ = ['FitResult', 'FitResults']


class FitResult:
    """Class that holds the result of a fit

    Parameters
    ----------
    fitted_parameters : ndarray
        The parameters fitted.
    fit_info : dict
        Additional information on the fit including at least 'classification', 'profile', 'success' and 'index'.

    Attributes
    ----------
    parameters : ndarray
        The parameters fitted.
    classification : int
        Classification of the fitted spectrum.
    profile : str
        Profile of the fitted spectrum.
    success : bool
        Whether the fit was completed successfully.
    index : list
        Index ([<time>, <row>, <column>]) of the spectrum in the spectral array.
    __dict__
        Other attributes may be present depending on the `fit_info` used.
    """
    def __init__(self, fitted_parameters, fit_info):
        self.__dict__ = fit_info  # Load first
        self.parameters = fitted_parameters

    def __len__(self):  # Calling the python `len` function on this object will return the number of fitted parameters
        return len(self.parameters)

    def __repr__(self):  # Useful string output of the object
        success = 'Successful ' if self.__dict__['success'] else 'Unsuccessful '
        if self.__dict__['index'] is not None and len(self.__dict__['index']) == 3:
            index = 'at ({}, {}, {}) '.format(*self.__dict__['index'])
        else:
            index = ''
        return success + 'FitResult ' + index + 'with ' + self.__dict__['profile'] \
            + ' profile of classification ' + str(self.__dict__['classification'])

    def plot(self, model, **kwargs):
        """Plot the data and fitted parameters

        This calls the `plot` method on `model` but will plot for this FitResult object. See the model's `plot` method
        for more details.

        Parameters
        ----------
        model : child class of ModelBase
            The model object to plot with.
        **kwargs
            See the `model.plot` method for more details.
        """
        model.plot(self, **kwargs)

    def velocity(self, model, vtype='quiescent'):
        """Calculate the Doppler velocity of the fit using `model` parameters

        Parameters
        ----------
        model : child class of ModelBase
            The model object to take parameters from.
        vtype : {'quiescent', 'active'}, default='quiescent'
            The velocity type to find.

        Returns
        -------
        velocity : float
            The calculated velocity.
        """
        stationary_line_core = model.stationary_line_core

        if vtype == 'quiescent':
            index = model.quiescent_wavelength
        elif vtype == 'active':
            index = model.active_wavelength
        else:
            raise ValueError("unknown velocity type '%s'" % vtype)

        try:
            wavelength = self.parameters[index]  # Choose the shifted wavelength from the fitted parameters
        except IndexError:  # Fit not compatible with this velocity type
            wavelength = np.nan  # No emission fitted

        return (wavelength - stationary_line_core) / stationary_line_core * 300000  # km/s


class FitResults:
    """Class that holds multiple fit results in a way that can be easily processed

    Attributes
    ----------
    parameters : ndarray, shape
    """
    def __init__(self, shape, n_parameters, time=None):

        if not isinstance(shape, tuple):
            raise TypeError("`shape` must be a tuple, got %s" % type(shape))

        if not isinstance(n_parameters, int) and n_parameters < 1:
            raise ValueError("`n_parameters` must be an integer greater than zero, got %s" % n_parameters)
        parameters_shape = tuple(list(shape) + [n_parameters])
        self.parameters = np.full(parameters_shape, np.nan, dtype=float)

        self.classifications = np.full(shape, -1, dtype=int)

        self.profile = np.full(shape, '', dtype=str)

        self.success = np.full(shape, False, dtype=bool)

        self.chi2 = np.full(shape, np.nan, dtype=float)

        self.time = time
        self.n_parameters = n_parameters

    def append(self, result):
        time, row, column = result.index
        if self.time is not None and self.time != time:
            raise ValueError("The time index of `result` does not match the time index being filled.")

        p = result.profile
        if p == 'absorption':
            self.parameters[row, column, :4] = result.parameters
        elif p == 'emission':
            self.parameters[row, column, 4:] = result.parameters
        elif p == 'both':
            self.parameters[row, column] = result.parameters
        else:
            raise ValueError("Unknown profile '%s'" % p)

        self.classifications[row, column] = result.classification
        self.profile[row, column] = result.profile
        self.success[row, column] = result.success
        self.chi2[row, column] = result.chi2

    def velocities(self, model, row=None, column=None, vtype='quiescent'):
        """Calculate the Doppler velocities of the fit results using `model` parameters

        Parameters
        ----------
        model : child class of ModelBase
            The model object to take parameters from.
        row : int, list, array_like, iterable, optional, default = None
            The row indices to find velocities for. All if omitted.
        column : int, list, array_like, iterable, optional, default = None
            The column indices to find velocities for. All if omitted.
        vtype : {'quiescent', 'active'}, default='quiescent'
            The velocity type to find.

        Returns
        -------
        velocities : ndarray of shape (`row`, `column`)
            The calculated velocities for the specified `row` and `column` positions.
        """
        if row is None:
            row = range(len(self.parameters))
        if column is None:
            column = range(len(self.parameters[0]))

        if vtype == 'quiescent':
            index = model.quiescent_wavelength
        elif vtype == 'active':
            index = model.active_wavelength
        else:
            raise ValueError("unknown velocity type '%s'" % vtype)

        index, row, column = make_iter(index, row, column)
        wavelengths = self.parameters[row][:, column][:, :, index]
        stationary_line_core = model.stationary_line_core
        return np.squeeze((wavelengths - stationary_line_core) / stationary_line_core * 300000, axis=2)  # km/s
