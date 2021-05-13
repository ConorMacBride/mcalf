import copy

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from mcalf.models.results import FitResult
from mcalf.models.base import *
from mcalf.profiles.voigt import voigt_nobg, double_voigt_nobg
from mcalf.utils.spec import generate_sigma
from mcalf.utils.misc import load_parameter
from mcalf.visualisation import plot_ibis8542


__all__ = ['IBIS8542Model']


class IBIS8542Model(ModelBase):
    """Class for working with IBIS 8542 Å calcium II spectral imaging observations.
    
    Parameters
    ----------
    ${PARAMETERS}

    Attributes
    ----------
    ${ATTRIBUTES}
    quiescent_wavelength : int, default=1
        The index within the fitted parameters of the absorption Voigt line core wavelength.
    active_wavelength : int, default=5
        The index within the fitted parameters of the emission Voigt line core wavelength.
    ${ATTRIBUTES_EXTRA}
    """
    def __init__(self, **kwargs):

        # STAGE 0A: Initialise the parent class `ModelBase`
        class_keys = [  # Keys that should not be passed to parent class's kwargs
            'absorption_guess',
            'emission_guess',
            'absorption_min_bound',
            'emission_min_bound',
            'absorption_max_bound',
            'emission_max_bound',
            'absorption_x_scale',
            'emission_x_scale',
            'random_state',
        ]  # These must match dictionary in STAGE 1 (defined there as stationary_line_core needs to be set)
        base_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in class_keys}
        super().__init__(**base_kwargs)

        # STAGE 0B: Load child default values for parent class attributes
        # stationary_line_core
        if self.stationary_line_core is None:
            self.stationary_line_core = 8542.099145376844
        # prefilter_response
        self._set_prefilter()  # Update the prefilter using stationary_line_core
        # sigma
        if self.sigma is None or (isinstance(self.sigma, bool) and self.sigma):
            self.sigma = [generate_sigma(i, self.constant_wavelengths, self.stationary_line_core) for i in [1, 2]]
        elif isinstance(self.sigma, bool) and not self.sigma:
            self.sigma = [np.ones(len(self.constant_wavelengths)) for i in [1, 2]]

        # STAGE 1: Define dictionary of default attribute values
        defaults = {
            'absorption_guess': [-1000, self.stationary_line_core, 0.2, 0.1],
            'emission_guess': [1000, self.stationary_line_core, 0.2, 0.1],
            'absorption_min_bound': [-np.inf, self.stationary_line_core - 0.15, 1e-6, 1e-6],
            'emission_min_bound': [0, -np.inf, 1e-6, 1e-6],
            'absorption_max_bound': [0, self.stationary_line_core + 0.15, 1, 1],
            'emission_max_bound': [np.inf, np.inf, 1, 1],
            'absorption_x_scale': [1500, 0.2, 0.3, 0.5],
            'emission_x_scale': [1500, 0.2, 0.3, 0.5],
            'random_state': None,
        }
        assert defaults.keys() == {k: None for k in class_keys}.keys()  # keys of `defaults` must match `class_keys`

        # STAGE 2: Update defaults with any values specified in a config file
        class_defaults = {k: self.config[k] for k in self.config.keys() if k in defaults.keys()}
        for k in class_defaults.keys():
            if k in ['absorption_x_scale', 'emission_x_scale', 'random_state']:
                # These should not need the stationary line core
                class_defaults[k] = load_parameter(class_defaults[k])
            else:
                class_defaults[k] = load_parameter(class_defaults[k], wl=self.stationary_line_core)
            self.config.pop(k)  # Remove copied parameter
        defaults.update(class_defaults)  # Update the defaults with the config file

        # STAGE 3: Update defaults with the keyword arguments passed into the class initialisation
        class_kwargs = {k: kwargs[k] for k in defaults.keys() if k in kwargs.keys()}
        defaults.update(class_kwargs)  # Update the defaults

        # STAGE 4: Set the object attributes (with some type enforcing)
        # values in the defaults dict
        self.absorption_guess = list(defaults['absorption_guess'])
        self.emission_guess = list(defaults['emission_guess'])
        self.absorption_min_bound = list(defaults['absorption_min_bound'])
        self.emission_min_bound = list(defaults['emission_min_bound'])
        self.absorption_max_bound = list(defaults['absorption_max_bound'])
        self.emission_max_bound = list(defaults['emission_max_bound'])
        self.absorption_x_scale = list(defaults['absorption_x_scale'])
        self.emission_x_scale = list(defaults['emission_x_scale'])
        # attributes whose default value cannot be changed during initialisation
        self.quiescent_wavelength = 1  # Index of quiescent wavelength in the fitted_parameters
        self.active_wavelength = 5  # Index of active wavelength in the fitted_parameters
        # neural_network
        if self.neural_network is None:
            mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,), max_iter=1000,
                                random_state=defaults['random_state'])
            parameter_space = {'alpha': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]}  # Search region
            # Set GridSearchCV to find best alpha
            self.neural_network = GridSearchCV(mlp, parameter_space, cv=5, n_jobs=-1)

        # STAGE 5: Validate the loaded attributes
        self._validate_attributes()

    def _validate_attributes(self):
        """Validate some of the object's attributes.

        Raises
        ------
        ValueError
            To signal that an attribute is not valid.
        """
        # Arrays must be of length 4
        if len(self.absorption_guess) != 4:
            raise ValueError("absorption_guess should be an array of length 4, got %s" %
                             len(self.absorption_guess))
        if len(self.emission_guess) != 4:
            raise ValueError("emission_guess should be an array of length 4, got %s" %
                             len(self.emission_guess))
        if len(self.absorption_min_bound) != 4:
            raise ValueError("absorption_min_bound should be an array of length 4, got %s" %
                             len(self.absorption_min_bound))
        if len(self.emission_min_bound) != 4:
            raise ValueError("emission_min_bound should be an array of length 4, got %s" %
                             len(self.emission_min_bound))
        if len(self.absorption_max_bound) != 4:
            raise ValueError("absorption_max_bound should be an array of length 4, got %s" %
                             len(self.absorption_max_bound))
        if len(self.emission_max_bound) != 4:
            raise ValueError("emission_max_bound should be an array of length 4, got %s" %
                             len(self.emission_max_bound))
        if len(self.absorption_x_scale) != 4:
            raise ValueError("absorption_x_scale should be an array of length 4, got %s" %
                             len(self.absorption_x_scale))
        if len(self.emission_x_scale) != 4:
            raise ValueError("emission_x_scale should be an array of length 4, got %s" %
                             len(self.emission_x_scale))

        # Absorption and emission fits must be constrained with the correct amplitude sign
        if self.absorption_guess[0] > 0:
            raise ValueError("absorption_guess amplitude should be negative, got %s" %
                             self.absorption_guess[0])
        if self.emission_guess[0] < 0:
            raise ValueError("emission_guess amplitude should be positive, got %s" %
                             self.emission_guess[0])
        if self.emission_min_bound[0] < 0:
            raise ValueError("emission_min_bound amplitude should be positive, got %s" %
                             self.emission_min_bound[0])
        if self.absorption_max_bound[0] > 0:
            raise ValueError("absorption_max_bound amplitude should be negative, got %s" %
                             self.absorption_max_bound[0])

        # Minimum bounds must be less that corresponding maximum bounds
        if np.sum(np.asarray(self.absorption_max_bound) - np.asarray(self.absorption_min_bound) > 0) != \
                len(self.absorption_max_bound):
            raise ValueError("values of absorption_max_bound must be greater than their "
                             "corresponding values in absorption_min_bound")
        if np.sum(np.asarray(self.emission_max_bound) - np.asarray(self.emission_min_bound) > 0) != \
                len(self.emission_max_bound):
            raise ValueError("values of emission_max_bound must be greater than their "
                             "corresponding values in emission_min_bound")

    def _get_sigma(self, classification=None, sigma=None):
        """Infer a sigma profile from the parameters provided.

        If no sigma is provided, use classification to take from the model object's sigma attribute.
        If sigma is provided and is an integer, take from the model object's sigma attribute at that index,
        otherwise, return sigma as a :class:`numpy.ndarray`.

        Parameters
        ----------
        classification : int, optional, default=None
            Classification sigma profile needed to fit.
        sigma : int or array_like, optional, default=None
            Explicit sigma index or profile.

        Returns
        -------
        sigma : numpy.ndarray, length=n_constant_wavelengths
             The sigma profile.

        See Also
        --------
        mcalf.utils.spec.generate_sigma : Generate a specified sigma profile.

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> import numpy as np
        >>> wavelengths = np.linspace(8541.3, 8542.7, 30)
        >>> model = mcalf.models.IBIS8542Model(original_wavelengths=wavelengths)

        Choose a sigma profile for the specified classification:

        >>> model._get_sigma(classification=3)
        array([1.        , 1.        , 1.        , 1.        , 1.        ,
               0.99999606, 0.99909053, 0.95597984, 0.55338777, 0.05021681,
               0.05021681, 0.05021681, 0.05021681, 0.4       , 0.4       ,
               0.4       , 0.4       , 0.4       , 0.4       , 0.4       ,
               0.05021681, 0.05021681, 0.05021681, 0.05021681, 0.57661719,
               0.96043995, 0.99922519, 0.99999682, 1.        , 1.        ])

        Get a specific sigma profile "index":

        >>> (model._get_sigma(sigma=1) == model.sigma[1]).all()
        True

        Convert an array like object into a suitable datatype for a sigma profile:

        >>> sigma = [1, 2, 3, 4, 5]
        >>> model._get_sigma(sigma=sigma)
        array([1., 2., 3., 4., 5.])
        """
        if sigma is None:
            # Decide how to weight the wavelengths
            if classification == 0:
                return self.sigma[0]
            else:
                return self.sigma[1]
        else:
            if isinstance(sigma, (int, np.integer)):
                return self.sigma[sigma]
            else:
                return np.asarray(sigma, dtype=np.float64)

    def _fit(self, spectrum, classification=None, spectrum_index=None, profile=None, sigma=None):
        """Fit a single spectrum for the given profile or classification.

        .. warning::
            Using this method directly will skip the corrections that are applied to spectra by the
            :meth:`~mcalf.models.ModelBase.get_spectra` method. Use :meth:`.fit_spectrum` instead.

        Parameters
        ----------
        spectrum : numpy.ndarray, ndim=1, length=n_constant_wavelengths
            The spectrum to be fitted.
        classification : int, optional, default=None
            Classification to determine the fitted profile to use (if profile not explicitly given).
        spectrum_index : array_like or list or tuple, length=3, optional, default=None
            The [time, row, column] index of the `spectrum` provided. Only used for error reporting.
        profile : str, optional, default=None
            The profile to fit. (Will infer profile from classification if omitted.)
        sigma : int or array_like, optional, default=None
            Explicit sigma index or profile. See :meth:`~mcalf.models.IBIS8542Model._get_sigma` for details.

        Returns
        -------
        result : mcalf.models.FitResult
            Outcome of the fit returned in a :class:`~mcalf.models.FitResult` object.

        See Also
        --------
        .fit_spectrum : The recommended method for fitting a single spectrum.
        .fit : The recommended method for fitting multiple spectra.
        mcalf.models.FitResult : The object that the fit method returns.
        """
        if profile is None:  # If profile hasn't been specified, find it

            # Decide which profiles to fit
            if classification in [2, 3, 4]:
                profile = 'both'
            elif classification in [0, 1]:
                profile = 'absorption'
            elif classification is None:
                raise ValueError("classification must be specified if profile is not specified")
            elif not isinstance(classification, (int, np.integer)):
                raise TypeError("classification must be an integer")
            else:
                raise ValueError("unexpected classification, got %s" % classification)

        sigma = self._get_sigma(classification, sigma)

        if profile == 'absorption':
            model = voigt_nobg
            guess = self.absorption_guess
            min_bound = self.absorption_min_bound
            max_bound = self.absorption_max_bound
            x_scale = self.absorption_x_scale
        elif profile == 'emission':
            model = voigt_nobg
            guess = self.emission_guess
            min_bound = self.emission_min_bound
            max_bound = self.emission_max_bound
            x_scale = self.emission_x_scale
        elif profile == 'both':
            model = double_voigt_nobg
            guess = self.absorption_guess + self.emission_guess
            min_bound = self.absorption_min_bound + self.emission_min_bound
            max_bound = self.absorption_max_bound + self.emission_max_bound
            x_scale = self.absorption_x_scale + self.absorption_x_scale
        else:
            raise ValueError("fit profile must be either None, 'absorption', 'emission' or 'both', got %s" % profile)

        time, row, column = spectrum_index if spectrum_index is not None else [None]*3
        fitted_parameters, success = self._curve_fit(model, spectrum, guess, sigma, (min_bound, max_bound), x_scale,
                                                     time=time, row=row, column=column)

        chi2 = np.sum(((spectrum - model(self.constant_wavelengths, *fitted_parameters)) / sigma) ** 2)

        fit_info = {'chi2': chi2, 'classification': classification, 'profile': profile,
                    'success': success, 'index': spectrum_index}

        return FitResult(fitted_parameters, fit_info)

    def plot(self, fit=None, time=None, row=None, column=None, spectrum=None, classification=None, background=None,
             sigma=None, stationary_line_core=None, **kwargs):
        """Plots the data and fitted parameters.

        Parameters
        ----------
        fit : mcalf.models.FitResult or list or array_like, optional, default=None
            The fitted parameters to plot with the data. Can extract the necessary plot metadata from the fit object.
            Otherwise, `fit` should be the parameters to be fitted to either a Voigt or double Voigt profile depending
            on the number of parameters fitted.
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list,
            :class:`numpy.ndarray`, a Python range, etc. can be used. If not provided, will be taken from
            `fit` if it is a :class:`~mcalf.models.FitResult` object, unless a `spectrum` is provided.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectrum : numpy.ndarray, length=`original_wavelengths`, ndim=1, optional, default=None
            The explicit spectrum to plot along with a fit (if specified).
        classification : int, optional, default=None
            Used to determine which sigma profile to use. See :meth:`~mcalf.models.IBIS8542Model._get_sigma`
            for more details. If not provided, will be taken from `fit` if it is a
            :class:`~mcalf.models.FitResult` object, unless a `spectrum` is provided.
        background : float or array_like, length=n_constant_wavelengths, optional, default= see note
            Background to added to the fitted profiles. If a `spectrum` is given, this will default to zero, otherwise
            the value loaded by :meth:`~mcalf.models.ModelBase.load_background` will be used.
        sigma : int or array_like, optional, default=None
            Explicit sigma index or profile. See :meth:`~mcalf.models.IBIS8542Model._get_sigma` for details.
        stationary_line_core : float, optional, default=`stationary_line_core`
            The stationary line core wavelength to mark on the plot.
        **kwargs : dict
            Other parameters used to adjust the plotting.
            See :func:`mcalf.visualisation.plot_ibis8542` for full details.

            * `separate` -- See :meth:`plot_separate`.
            * `subtraction` -- See :meth:`plot_subtraction`.
            * `sigma_scale` -- A factor to multiply the error bars to change their prominence.

        See Also
        --------
        plot_separate : Plot the fit parameters separately.
        plot_subtraction : Plot the spectrum with the emission fit subtracted from it.
        mcalf.models.FitResult.plot : Plotting method on the fit result.

        Examples
        --------
        .. minigallery:: mcalf.visualisation.plot_ibis8542
        """
        if fit.__class__ == FitResult:  # If fit is a `FitResult`

            # Extract classification (so it can choose the correct sigma to plot)
            if classification is None:
                classification = fit.classification

            # Extract spectrum index (so it can load the raw data)
            if (time, row, column) == (None, None, None) and fit.index is not None:
                time, row, column = fit.index

            # Replace fit with just the fitted parameters
            fit = fit.parameters

        # Has a spectrum been given? (used for tidying plot later)
        explicit_spectrum = False if spectrum is None else True

        # Get the spectrum and reduce it to 1D
        spectrum = self.get_spectra(time=time, row=row, column=column, spectrum=spectrum, background=True)
        spectrum = np.squeeze(spectrum)

        # If no background to be added to the fit is given...
        if background is None and fit is not None:
            if not explicit_spectrum:  # Take from loaded background if using indices
                time, row, column = self._get_time_row_column(time=time, row=row, column=column)
                background = self.background[time, row, column]
            else:  # Otherwise assume to be zero
                background = 0

        if classification is not None or sigma is not None:  # Skip if neither classification or sigma are given
            sigma = self._get_sigma(classification, sigma)

        if stationary_line_core is None:
            stationary_line_core = self.stationary_line_core

        return plot_ibis8542(self.constant_wavelengths, spectrum, fit=fit, background=background, sigma=sigma,
                             stationary_line_core=stationary_line_core, **kwargs)

    def plot_separate(self, *args, **kwargs):
        """Plot the fitted profiles separately.

        If multiple profiles exist, fit them separately.
        Arguments are the same as the :meth:`plot` method.

        See Also
        --------
        plot : General plotting method.
        plot_subtraction : Plot the spectrum with the emission fit subtracted from it.
        mcalf.models.FitResult.plot : Plotting method on the fit result.
        """
        self.plot(*args, separate=True, **kwargs)

    def plot_subtraction(self, *args, **kwargs):
        """Plot the spectrum with the emission fit subtracted from it.

        If multiple profiles exist, subtract the fitted emission from the raw data.
        Arguments are the same as the :meth:`plot` method.

        See Also
        --------
        plot : General plotting method.
        plot_separate : Plot the fit parameters separately.
        mcalf.models.FitResult.plot : Plotting method on the fit result.
        """
        self.plot(*args, subtraction=True, **kwargs)


# Copy documentation from base class
IBIS8542_PARAMETERS = copy.deepcopy(BASE_PARAMETERS)
IBIS8542_ATTRIBUTES = copy.deepcopy(BASE_ATTRIBUTES)

# Update documentation from base class (include new defaults)
for d in [IBIS8542_PARAMETERS, IBIS8542_ATTRIBUTES]:
    d['stationary_line_core'] = """
    stationary_line_core : float, optional, default=8542.099145376844
        Wavelength of the stationary line core."""
    d['sigma'] = """
    sigma : list of array_like or bool, length=(2, n_wavelengths), optional, default=[type1, type2]
        A list of different sigma that are used to weight particular wavelengths along the spectra when fitting. The
        fitting method will expect to be able to choose a sigma array from this list at a specific index. It's default
        value is `[generate_sigma(i, constant_wavelengths, stationary_line_core) for i in [1, 2]]`.
        See :func:`mcalf.utils.spec.generate_sigma` for more information. 
        If bool, True will generate the default sigma value regardless of the value specified in `config`,
        and False will set `sigma` to be all ones, effectively disabling it."""
IBIS8542_ATTRIBUTES['neural_network'] = """
    neural_network : sklearn.neural_network.MLPClassifier, optional, default= see description
        The :class:`sklearn.neural_network.MLPClassifier` object (or similar) that will be used to
        classify the spectra. Defaults to a :class:`sklearn.model_selection.GridSearchCV`
        with :class:`~sklearn.neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,), max_iter=1000)`
        for best `alpha` selected from `[1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]`."""

# Add documentation for new parameters and attributes
IBIS8542_DOCS = """        
    absorption_guess : array_like, length=4, optional, default=[-1000, stationary_line_core, 0.2, 0.1]
        Initial guess to take when fitting the absorption Voigt profile.
    emission_guess : array_like, length=4, optional, default=[1000, stationary_line_core, 0.2, 0.1]
        Initial guess to take when fitting the emission Voigt profile.
    absorption_min_bound : array_like, length=4, optional, default=[-np.inf, stationary_line_core-0.15, 1e-6, 1e-6]
        Minimum bounds for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_min_bound : array_like, length=4, optional, default=[0, -np.inf, 1e-6, 1e-6]
        Minimum bounds for all the emission Voigt profile parameters in order of the function's arguments.
    absorption_max_bound : array_like, length=4, optional, default=[0, stationary_line_core+0.15, 1, 1]
        Maximum bounds for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_max_bound : array_like, length=4, optional, default=[np.inf, np.inf, 1, 1]
        Maximum bounds for all the emission Voigt profile parameters in order of the function's arguments.
    absorption_x_scale : array_like, length=4, optional, default=[1500, 0.2, 0.3, 0.5]
        Characteristic scale for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_x_scale : array_like, length=4, optional, default=[1500, 0.2, 0.3, 0.5]
        Characteristic scale for all the emission Voigt profile parameters in order of the function's arguments.
    random_state : int, numpy.random.RandomState, optional, default=None
        Determines random number generation for weights and bias initialisation of the default `neural_network`.
        Pass an int for reproducible results across multiple function calls."""

# Form the docstring and do the replacements
IBIS8542_PARAMETERS_STR = ''.join(IBIS8542_PARAMETERS[i] for i in IBIS8542_PARAMETERS)
IBIS8542_ATTRIBUTES_STR = ''.join(IBIS8542_ATTRIBUTES[i] for i in IBIS8542_ATTRIBUTES)
IBIS8542Model.__doc__ = IBIS8542Model.__doc__.replace(
    '${PARAMETERS}',
    (IBIS8542_DOCS + IBIS8542_PARAMETERS_STR).lstrip()
)
IBIS8542Model.__doc__ = IBIS8542Model.__doc__.replace(
    '${ATTRIBUTES}',
    IBIS8542_DOCS.lstrip()
)
IBIS8542Model.__doc__ = IBIS8542Model.__doc__.replace(
    '${ATTRIBUTES_EXTRA}',
    IBIS8542_ATTRIBUTES_STR.lstrip()
)
