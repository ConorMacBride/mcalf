import numpy as np
from astropy.io import fits

from mcalf.utils.misc import make_iter


__all__ = ['FitResult', 'FitResults']


class FitResult:
    """Class that holds the result of a fit.

    Parameters
    ----------
    fitted_parameters : numpy.ndarray
        The parameters fitted.
    fit_info : dict
        Additional information on the fit including at least 'classification', 'profile', 'success', 'chi2' and 'index'.

    Attributes
    ----------
    parameters : numpy.ndarray
        The parameters fitted.
    classification : int
        Classification of the fitted spectrum.
    profile : str
        Profile of the fitted spectrum.
    success : bool
        Whether the fit was completed successfully.
    chi2 : float
        Chi-squared value for the fit.
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
        index = ''
        if 'index' in self.__dict__:
            i = self.__dict__['index']
            if (isinstance(i, list) or isinstance(i, tuple)) and len(i) == 3 and all([j is not None for j in i]):
                index = 'at ({}, {}, {}) '.format(*i)
        return success + 'FitResult ' + index + 'with ' + self.__dict__['profile'] \
            + ' profile of classification ' + str(self.__dict__['classification'])

    def plot(self, model, **kwargs):
        """Plot the data and fitted parameters.

        This calls the `plot` method on `model` but will plot for this FitResult object. See the model's `plot` method
        for more details.

        Parameters
        ----------
        model : child class of :class:`~mcalf.models.ModelBase`
            The model object to plot with.
        **kwargs : dict
            See the `model.plot` method for more details.
        """
        model.plot(self, **kwargs)

    def velocity(self, model, vtype='quiescent'):
        """Calculate the Doppler velocity of the fit using `model` parameters.

        Parameters
        ----------
        model : child class of :class:`~mcalf.models.ModelBase`
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
    """Class that holds multiple fit results in a way that can be easily processed.

    Parameters
    ----------
    shape : tuple of int
        The number of rows and columns to hold data for, e.g. (n_rows, n_columns).
    n_parameters : int
        The number of fitted parameters per spectrum that need to be stored.
    time : int, optional, default=None
        The time the `FitResults` object will store data for. Optional, but if it is set, only
        :class:`~mcalf.models.FitResult` objects with a matching time can be appended.

    Attributes
    ----------
    parameters : numpy.ndarray, shape=(row, column, parameter)
        Array of fitted parameters.
    classifications : numpy.ndarray of int, shape=(row, column)
        Array of classifications.
    profile : numpy.ndarray of str, shape=(row, column)
        Array of profiles.
    success : numpy.ndarray of bool, shape=(row, column)
        Array of success statuses.
    chi2 : numpy.ndarray, shape=(row, column)
        Array of chi-squared values.
    time : int, default=None
        Time index that the :class:`~mcalf.models.FitResult` object refers to (if provided).
    n_parameters : int
        Number of parameters in the last dimension of `parameters`.
    """
    def __init__(self, shape, n_parameters, time=None):
        # TODO Allow multiple time indices to be imported
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError("`shape` must be a tuple of length 2, got %s" % type(shape))

        if not isinstance(n_parameters, (int, np.integer)) or n_parameters < 1:
            raise ValueError("`n_parameters` must be an integer greater than zero, got %s" % n_parameters)
        parameters_shape = tuple(list(shape) + [n_parameters])
        self.parameters = np.full(parameters_shape, np.nan, dtype=float)

        self.classifications = np.full(shape, -1, dtype=int)

        self.profile = np.full(shape, '', dtype=object)

        self.success = np.full(shape, False, dtype=bool)

        self.chi2 = np.full(shape, np.nan, dtype=float)

        self.time = time
        self.n_parameters = n_parameters

    def append(self, result):
        """Append a :class:`~mcalf.models.FitResult` object to the `FitResults` object.

        Parameters
        ----------
        result : ~mcalf.models.FitResult
            :class:`~mcalf.models.FitResult` object to append.
        """
        time, row, column = result.index
        if self.time is not None and self.time != time:
            raise ValueError("The time index of `result` does not match the time index being filled.")

        # TODO Make the number of parameters and types of profiles general.
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
        """Calculate the Doppler velocities of the fit results using `model` parameters.

        Parameters
        ----------
        model : child class of mcalf.models.ModelBase
            The model object to take parameters from.
        row : int, list, array_like, iterable, optional, default=None
            The row indices to find velocities for. All if omitted.
        column : int, list, array_like, iterable, optional, default=None
            The column indices to find velocities for. All if omitted.
        vtype : {'quiescent', 'active'}, default='quiescent'
            The velocity type to find.

        Returns
        -------
        velocities : numpy.ndarray, shape=(row, column)
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

    def save(self, filename, model=None):
        """Saves the FitResults object to a FITS file.

        Parameters
        ----------
        filename : file path, file object or file-like object
            FITS file to write to. If a file object, must be opened in a writeable mode.
        model : child class of mcalf.models.ModelBase, optional, default=None
            If provided, use this model to calculate and include both quiescent and active Doppler velocities.

        Notes
        -----
        Saves a FITS file to the location specified by `filename`. All the parameters are stored in a separate,
        named, HDU.
        """
        # Compress profile array to integers
        p_uniq = np.unique(self.profile)
        p_legend = np.array_str(p_uniq)
        p = np.full_like(self.profile, -1, dtype=np.int16)
        for i in range(len(p_uniq)):
            p[self.profile == p_uniq[i]] = i

        header = fits.Header({
            'NTIME': 1,
            'NROWS': self.classifications.shape[-2],
            'NCOLS': self.classifications.shape[-1],
            'TIME': self.time,
        })
        primary_hdu = fits.PrimaryHDU([], header)

        header = fits.Header({'NPARAMS': self.n_parameters})
        parameters_hdu = fits.ImageHDU(self.parameters, header, 'PARAMETERS')

        classifications_hdu = fits.ImageHDU(np.asarray(self.classifications, dtype=np.int16), name='CLASSIFICATIONS')

        header = fits.Header({'PROFILES': p_legend})
        profile_hdu = fits.ImageHDU(p, header, 'PROFILE')

        success_hdu = fits.ImageHDU(np.asarray(self.success, dtype=np.int16), name='SUCCESS')

        chi2_hdu = fits.ImageHDU(self.chi2, name='CHI2')

        hdul = fits.HDUList([primary_hdu, parameters_hdu, classifications_hdu, profile_hdu, success_hdu, chi2_hdu])

        if model is not None:
            for head, vtype, name in [('ACTIVE', 'active', 'VLOSA'),
                                      ('QUIESCENT', 'quiescent', 'VLOSQ')]:
                header = fits.Header({'VTYPE': head, 'UNIT': 'KM/S'})
                v = self.velocities(model, vtype=vtype)
                v_hdu = fits.ImageHDU(v, header, name)
                hdul.append(v_hdu)

        hdul.writeto(filename, checksum=True)
