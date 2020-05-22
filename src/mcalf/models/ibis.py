import warnings

import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from mcalf.models.results import FitResult
from mcalf.models.base import ModelBase
from mcalf.profiles.voigt import voigt_nobg, double_voigt_nobg
from mcalf.utils.spec import reinterpolate_spectrum, generate_sigma
from mcalf.utils.misc import load_parameter, make_iter
from mcalf.visualisation.spec import plot_ibis8542


__all__ = ['IBIS8542Model']


class IBIS8542Model(ModelBase):
    """Class for working with IBIS 8542 Å calcium II spectral imaging observations

    Parameters
    ----------
    original_wavelengths : array_like
        One-dimensional array of wavelengths that correspond to the uncorrected spectral data.
    stationary_line_core : float, optional, default = 8542.104320687517
        Wavelength of the stationary line core.
    absorption_guess : array_like, length=4, optional, default = [-1000, stationary_line_core, 0.2, 0.1]
        Initial guess to take when fitting the absorption Voigt profile.
    emission_guess : array_like, length=4, optional, default = [1000, stationary_line_core, 0.2, 0.1]
        Initial guess to take when fitting the emission Voigt profile.
    absorption_min_bound : array_like, length=4, optional, default = [-np.inf, stationary_line_core-0.15, 1e-6, 1e-6]
        Minimum bounds for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_min_bound : array_like, length=4, optional, default = [0, stationary_line_core-0.15, 1e-6, 1e-6]
        Minimum bounds for all the emission Voigt profile parameters in order of the function's arguments.
    absorption_max_bound : array_like, length=4, optional, default = [0, stationary_line_core+0.15, 1, 1]
        Maximum bounds for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_max_bound : array_like, length=4, optional, default = [np.inf, stationary_line_core+0.15, 1, 1]
        Maximum bounds for all the emission Voigt profile parameters in order of the function's arguments.
    absorption_x_scale : array_like, length=4, optional, default = [1500, 0.2, 0.3, 0.5]
        Characteristic scale for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_x_scale : array_like, length=4, optional, default = [1500, 0.2, 0.3, 0.5]
        Characteristic scale for all the emission Voigt profile parameters in order of the function's arguments.
    neural_network : sklearn.neural_network.MLPClassifier, optional, default = see description
        The MLPClassifier object that will be used to classify the spectra. Its default value is
        `MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(10, 4), random_state=1)`.
    constant_wavelengths : array_like, length same as `original_wavelengths`, optional, default = see description
        The desired set of wavelengths that the spectral data should be rescaled to represent. It is assumed
        that these have constant spacing, but that may not be a requirement if you specify your own array.
        The default value is an array from the minimum to the maximum wavelength of `original_wavelengths` in
        constant steps of `delta_lambda`, overshooting the upper bound if the maximum wavelength has not been reached.
    delta_lambda : float, optional, default = 0.05
        The step used between each value of `constant_wavelengths` when its default value has to be calculated.
    sigma : list of array_like or bool, length=(2, n_wavelengths), optional, default = [type1, type2]
        A list of different sigma that are used to weight particular wavelengths along the spectra when fitting. The
        fitting method will expect to be able to choose a sigma array from this list at a specific index. It's default
        value is `[generate_sigma(i, constant_wavelengths, stationary_line_core) for i in [1, 2]]`.
        See `utils.generate_sigma()` for more information. If bool, True will generate the default sigma value
        regardless of the value specified in `config`, and False will set `sigma` to be all ones, effectively disabling
        it.
    prefilter_response : array_like, length=n_wavelengths, optional, default = see note
        Each constant wavelength scaled spectrum will be corrected by dividing it by this array. If `prefilter_response`
        is not give, and `prefilter_ref_main` and `prefilter_ref_wvscl` are not given, `prefilter_response` will have a
        default value of `None`.
    prefilter_ref_main : array_like, optional, default = None
        If `prefilter_response` is not specified, this will be used along with `prefilter_ref_wvscl` to generate the
        default value of `prefilter_response`.
    prefilter_ref_wvscl : array_like, optional, default = None
        If `prefilter_response` is not specified, this will be used along with `prefilter_ref_main` to generate the
        default value of `prefilter_response`.
    config : str, optional, default = None
        Filename of a `.yml` file (relative to current directory) containing the initialising parameters for this
        object. Parameters provided explicitly to the object upon initialisation will override any provided in this
        file. All (or some) parameters that this object accepts can be specified in this file, except `neural_network`
        and `config`. Each line of the file should specify a different parameter and be formatted like
        `emission_guess: '[-inf, wl-0.15, 1e-6, 1e-6]'` or `original_wavelengths: 'original.fits'` for example.
        When specifying a string, use 'inf' to represent `np.inf` and 'wl' to represent `stationary_line_core` as shown.
        If the string matches a file, `utils.load_parameter()` is used to load the contents of the file.
    output : str, optional, default = None
        If the program wants to output data, it will place it relative to the location specified by this parameter.
        Some methods will only save data to a file if this parameter is not `None`. Such cases will be documented
        where relevant.

    Attributes
    ----------
    original_wavelengths : array_like
        One-dimensional array of wavelengths that correspond to the uncorrected spectral data.
    stationary_line_core : float, optional, default = 8542.099145376844
        Wavelength of the stationary line core.
    absorption_guess : array_like, length=4, optional, default = [-1000, stationary_line_core, 0.2, 0.1]
        Initial guess to take when fitting the absorption Voigt profile.
    emission_guess : array_like, length=4, optional, default = [1000, stationary_line_core, 0.2, 0.1]
        Initial guess to take when fitting the emission Voigt profile.
    absorption_min_bound : array_like, length=4, optional, default = [-np.inf, stationary_line_core-0.15, 1e-6, 1e-6]
        Minimum bounds for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_min_bound : array_like, length=4, optional, default = [0, -np.inf, 1e-6, 1e-6]
        Minimum bounds for all the emission Voigt profile parameters in order of the function's arguments.
    absorption_max_bound : array_like, length=4, optional, default = [0, stationary_line_core+0.15, 1, 1]
        Maximum bounds for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_max_bound : array_like, length=4, optional, default = [np.inf, np.inf, 1, 1]
        Maximum bounds for all the emission Voigt profile parameters in order of the function's arguments.
    absorption_x_scale : array_like, length=4, optional, default = [1500, 0.2, 0.3, 0.5]
        Characteristic scale for all the absorption Voigt profile parameters in order of the function's arguments.
    emission_x_scale : array_like, length=4, optional, default = [1500, 0.2, 0.3, 0.5]
        Characteristic scale for all the emission Voigt profile parameters in order of the function's arguments.
    neural_network : sklearn.neural_network.MLPClassifier, optional, default = see description
        The MLPClassifier object (or similar) that will be used to classify the spectra. Defaults to a `GridSearchCV`
        with `MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,), max_iter=1000)`
        for best `alpha` selected from `[1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]`.
    constant_wavelengths : array_like, length same as `original_wavelengths`, optional, default = see description
        The desired set of wavelengths that the spectral data should be rescaled to represent. It is assumed
        that these have constant spacing, but that may not be a requirement if you specify your own array.
        The default value is an array from the minimum to the maximum wavelength of `original_wavelengths` in
        constant steps of `delta_lambda`, overshooting the upper bound if the maximum wavelength has not been reached.
    sigma : list of array_like, length=(2, n_wavelengths), optional, default = [type1, type2]
        A list of different sigma that are used to weight particular wavelengths along the spectra when fitting. The
        fitting method will expect to be able to choose a sigma array from this list at a specific index. It's default
        value is `[generate_sigma(i, constant_wavelengths, stationary_line_core) for i in [1, 2]]`.
        See `utils.generate_sigma()` for more information.
    prefilter_response : array_like, length=n_wavelengths, optional, default = see note
        Each constant wavelength scaled spectrum will be corrected by dividing it by this array. If `prefilter_response`
        is not give, and `prefilter_ref_main` and `prefilter_ref_wvscl` are not given, `prefilter_response` will have a
        default value of `None`.
    output : str, optional, default = None
        If the program wants to output data, it will place it relative to the location specified by this parameter.
        Some methods will only save data to a file if this parameter is not `None`. Such cases will be documented
        where relevant.
    quiescent_wavelength : int, default = 1
        The index within the fitted parameters of the absorption Voigt line core wavelength.
    active_wavelength : int, default = 5
        The index within the fitted parameters of the emission Voigt line core wavelength.
    """
    def __init__(self, stationary_line_core=None,
                 absorption_guess=None, emission_guess=None,
                 absorption_min_bound=None, emission_min_bound=None,
                 absorption_max_bound=None, emission_max_bound=None,
                 absorption_x_scale=None, emission_x_scale=None,
                 neural_network=None,
                 original_wavelengths=None, constant_wavelengths=None,
                 delta_lambda=None, sigma=None, prefilter_response=None,
                 prefilter_ref_main=None, prefilter_ref_wvscl=None,
                 config=None, output=None):

        super().__init__()  # Initialise the parent class `ModelBase` also

        if config is not None:  # Process config file if one is specified

            with open(config, 'r') as stream:  # Load YAML file
                parameters = load(stream, Loader=Loader)

            # Load each parameter if it exists in the file and is not already given
            if 'stationary_line_core' in parameters and stationary_line_core is None:
                stationary_line_core = load_parameter(parameters['stationary_line_core'])
            if 'absorption_guess' in parameters and absorption_guess is None:
                absorption_guess = load_parameter(parameters['absorption_guess'], wl=stationary_line_core)
            if 'emission_guess' in parameters and emission_guess is None:
                emission_guess = load_parameter(parameters['emission_guess'], wl=stationary_line_core)
            if 'absorption_min_bound' in parameters and absorption_min_bound is None:
                absorption_min_bound = load_parameter(parameters['absorption_min_bound'], wl=stationary_line_core)
            if 'emission_min_bound' in parameters and emission_min_bound is None:
                emission_min_bound = load_parameter(parameters['emission_min_bound'], wl=stationary_line_core)
            if 'absorption_max_bound' in parameters and absorption_max_bound is None:
                absorption_max_bound = load_parameter(parameters['absorption_max_bound'], wl=stationary_line_core)
            if 'emission_max_bound' in parameters and emission_max_bound is None:
                emission_max_bound = load_parameter(parameters['emission_max_bound'], wl=stationary_line_core)
            if 'absorption_x_scale' in parameters and absorption_x_scale is None:
                absorption_x_scale = load_parameter(parameters['absorption_x_scale'])
            if 'emission_x_scale' in parameters and emission_x_scale is None:
                emission_x_scale = load_parameter(parameters['emission_x_scale'])
            if 'original_wavelengths' in parameters and original_wavelengths is None:
                original_wavelengths = load_parameter(parameters['original_wavelengths'])
            if 'constant_wavelengths' in parameters and constant_wavelengths is None:
                constant_wavelengths = load_parameter(parameters['constant_wavelengths'])
            if 'delta_lambda' in parameters and delta_lambda is None:
                delta_lambda = load_parameter(parameters['delta_lambda'])
            if 'sigma' in parameters and sigma is None:
                sigma = load_parameter(parameters['sigma'])
            if 'prefilter_response' in parameters and prefilter_response is None:
                prefilter_response = load_parameter(parameters['prefilter_response'])
            if 'prefilter_ref_main' in parameters and prefilter_ref_main is None:
                prefilter_ref_main = load_parameter(parameters['prefilter_ref_main'])
            if 'prefilter_ref_wvscl' in parameters and prefilter_ref_wvscl is None:
                prefilter_ref_wvscl = load_parameter(parameters['prefilter_ref_wvscl'])
            if 'output' in parameters and output is None:
                output = parameters['output']

        # Load default values of any parameters that haven't been given yet
        if stationary_line_core is None:
            stationary_line_core = 8542.099145376844
        if absorption_guess is None:
            absorption_guess = [-1000, stationary_line_core, 0.2, 0.1]
        if emission_guess is None:
            emission_guess = [1000, stationary_line_core, 0.2, 0.1]
        if absorption_min_bound is None:
            absorption_min_bound = [-np.inf, stationary_line_core-0.15, 1e-6, 1e-6]
        if emission_min_bound is None:
            emission_min_bound = [0, -np.inf, 1e-6, 1e-6]
        if absorption_max_bound is None:
            absorption_max_bound = [0, stationary_line_core+0.15, 1, 1]
        if emission_max_bound is None:
            emission_max_bound = [np.inf, np.inf, 1, 1]
        if absorption_x_scale is None:
            absorption_x_scale = [1500, 0.2, 0.3, 0.5]
        if emission_x_scale is None:
            emission_x_scale = [1500, 0.2, 0.3, 0.5]
        if neural_network is None:
            mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,), max_iter=1000)
            parameter_space = {'alpha': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]}  # Search region
            neural_network = GridSearchCV(mlp, parameter_space, cv=5, n_jobs=-1)  # Set GridSearchCV to find best alpha
        if original_wavelengths is None:
            raise ValueError("original_wavelengths must be specified")
        if delta_lambda is None:
            delta_lambda = 0.05
        if constant_wavelengths is None:
            constant_wavelengths = np.arange(min(original_wavelengths), max(original_wavelengths)+delta_lambda,
                                             delta_lambda)
        if prefilter_response is None:
            if prefilter_ref_main is not None and prefilter_ref_wvscl is not None:
                prefilter_response = reinterpolate_spectrum(prefilter_ref_main,
                                                            prefilter_ref_wvscl + stationary_line_core,
                                                            constant_wavelengths)
            else:
                warnings.warn("prefilter_response will not be applied to spectra")
        else:  # Make sure it is a numpy array so that division works as expected when doing array operations
            prefilter_response = np.asarray(prefilter_response, dtype=np.float64)

        # Set the object attributes (with some type enforcing)
        self.stationary_line_core = stationary_line_core
        self.absorption_guess = list(absorption_guess)
        self.emission_guess = list(emission_guess)
        self.absorption_min_bound = list(absorption_min_bound)
        self.emission_min_bound = list(emission_min_bound)
        self.absorption_max_bound = list(absorption_max_bound)
        self.emission_max_bound = list(emission_max_bound)
        self.absorption_x_scale = list(absorption_x_scale)
        self.emission_x_scale = list(emission_x_scale)
        self.neural_network = neural_network
        self.original_wavelengths = np.asarray(original_wavelengths, dtype=np.float64)
        self.constant_wavelengths = np.asarray(constant_wavelengths, dtype=np.float64)
        self.prefilter_response = prefilter_response
        self.output = output

        self.__delta_lambda = delta_lambda

        # Index of wavelength in the fitted_parameters
        self.quiescent_wavelength = 1
        self.active_wavelength = 5

        # Run some checks to make sure the specified parameters are valid
        self._validate_parameters()

        # Generate default sigma profiles for weighting fit (validate dependent parameters first)
        if sigma is None or (isinstance(sigma, bool) and sigma):
            sigma = [generate_sigma(i, self.constant_wavelengths, self.stationary_line_core) for i in [1, 2]]
        elif isinstance(sigma, bool) and not sigma:
            sigma = [np.ones(len(constant_wavelengths)) for i in [1, 2]]

        self.sigma = sigma  # Set the sigma parameter

    def _validate_parameters(self):
        """Validate some of the object's parameters

        Raises
        ------
        ValueError
            To signal that a parameter is not valid.
        """
        # Stationary line core must be a float
        if not isinstance(self.stationary_line_core, float):
            raise ValueError("stationary_line_core must be a float, got %s" % type(self.stationary_line_core))

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

        # Wavelength arrays must be sorted ascending
        if np.sum(np.diff(self.original_wavelengths) > 0) < len(self.original_wavelengths) - 1:
            raise ValueError("original_wavelength array must be sorted ascending")
        if np.sum(np.diff(self.constant_wavelengths) > 0) < len(self.constant_wavelengths) - 1:
            raise ValueError("constant_wavelength array must be sorted ascending")

        # Warn if the constant wavelengths extrapolate the original wavelengths
        if min(self.constant_wavelengths) - min(self.original_wavelengths) < -1e-6:
            # If lower-bound of constant wavelengths is more than 1e-6 outside of the original wavelengths
            warnings.warn("Lower bound of `constant_wavelengths` is outside of `original_wavelengths` range.")
        if max(self.constant_wavelengths) - max(self.original_wavelengths) - self.__delta_lambda > 1e-6:
            # If upper-bound of constant wavelengths is more than 1e-6 ouside the original wavelengths
            warnings.warn("Upper bound of `constant_wavelengths` is outside of `original_wavelengths` range.")

        # Stationary wavelength must be within wavelength range
        original_diff = self.original_wavelengths - self.stationary_line_core
        constant_diff = self.constant_wavelengths - self.stationary_line_core
        for n, i in [['original_wavelengths', original_diff], ['constant_wavelengths', constant_diff]]:
            if min(i) > 1e-6 or max(i) < -1e-6:
                raise ValueError("`stationary_line_core` is not within `{}`".format(n))

        # If a prefilter response is given it must be a compatible length
        if self.prefilter_response is not None:
            if len(self.prefilter_response) != len(self.constant_wavelengths):
                raise ValueError("prefilter_response array must be the same length as constant_wavelengths array")

    def _get_sigma(self, classification=None, sigma=None):
        """Infer a sigma profile from the parameters provided

        If no sigma is provided, use classification to take from the model object's sigma attribute.
        If sigma is provided and is an integer, take from the model object's sigma attribute at that index,
        otherwise, return sigma as a NumPy array.

        Parameters
        ----------
        classification : int, optional, default = None
            Classification sigma profile needed to fit.
        sigma : int or array_like, optional, default = None
            Explicit sigma index or profile.

        Returns
        -------
        sigma : ndarray, length = n_constant_wavelengths
             The sigma profile.

        See Also
        --------
        utils.generate_sigma : Generate a specified sigma profile
        """
        if sigma is None:
            # Decide how to weight the wavelengths
            if classification == 0:
                return self.sigma[0]
            else:
                return self.sigma[1]
        else:
            if isinstance(sigma, int):
                return self.sigma[sigma]
            else:
                return np.asarray(sigma, dtype=np.float64)

    def classify_spectra(self, time=None, row=None, column=None, spectra=None, only_normalise=False):
        """Classify the specified spectra

        Will also normalise each spectrum such that its intensity will range from zero to one.

        Parameters
        ----------
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list, a NumPy
            array, a Python range, etc. can be used.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectra : ndarray, optional, default=None
            The explicit spectra to classify. If `only_normalise` is False, this must be 1D.
        only_normalise : bool, optional, default = False
            Whether the single spectrum given  in `spectra` should not be interpolated and corrected.

        Returns
        -------
        classifications : ndarray
            Array of classifications with the same time, row and column indices as `spectra`.

        See Also
        --------
        train : Train the neural network
        test : Test the accuracy of the neural network
        get_spectra : Get processed spectra from the objects `array` attribute
        """
        if not only_normalise:  # Get the spectrum, otherwise use the provided one directly
            spectra = self.get_spectra(time=time, row=row, column=column, spectrum=spectra)

        # Vectorised normalisation for all spectra such that intensities for each spectrum is in range [0, 1]
        spectra_list = spectra.reshape(-1, spectra.shape[-1]).T  # Reshape & transpose to (n_wavelengths, n_spectra)
        spectra_list = spectra_list - spectra_list.min(axis=0)  # Subtract each spectrum's min value from itself (min 0)
        spectra_list = spectra_list / spectra_list.max(axis=0)  # Divide each spectrum by its max value (min 0, max 1)
        spectra_list = spectra_list.T  # Transpose to (n_spectra, n_wavelengths)

        # Create the empty classifications array
        classifications = np.full(len(spectra_list), -1, dtype=int)  # -1 reserved for invalid
        valid_spectra_i = np.where(~np.isnan(spectra_list[:, 0]))  # Only predict the valid spectra
        # Note: Spectra that have indices in the data but were masked or outside the field of view should be np.nan
        try:
            classifications[valid_spectra_i] = self.neural_network.predict(spectra_list[valid_spectra_i])
        except NotFittedError:
            raise NotFittedError("Neural network has not been trained yet. Call 'train' on the model class first.")
        try:  # Try to give the classifications array the same indices as the spectral array
            classifications = classifications.reshape(*spectra.shape[:-1])
        except TypeError:  # 1D spectra array given
            assert classifications.size == 1  # Verify

        return classifications

    def _fit(self, spectrum, profile=None, sigma=None, classification=None, spectrum_index=None):
        """Fit a single spectrum for the given profile or classification

        Warning: Using this method directly will skip the corrections that are applied to spectra by the `get_spectra`
        method.

        Parameters
        ----------
        spectrum : ndarray, ndim=1, length=n_constant_wavelengths
            The spectrum to be fitted.
        profile : str, optional, default = None
            The profile to fit. (Will infer profile from classification if omitted.)
        sigma : ndarray, ndim=1, length=n_constant_wavelengths
            The sigma array with weights for each spectral wavelength.
        classification : int, optional, default = None
            Classification to determine the fitted profile to use (if profile not explicitly given).
        spectrum_index : array_like or list or tuple, length=3, optional, default = None
            The [time, row, column] index of the `spectrum` provided. Only used for error reporting.

        Returns
        -------
        result : FitResult
            Outcome of the fit returned in a FitResult object

        See Also
        --------
        fit : The recommended method for fitting spectra
        FitResult : The object that the fit method returns
        """
        if profile is None:  # If profile hasn't been specified, find it

            # Decide which profiles to fit
            if classification in [2, 3, 4]:
                profile = 'both'
            elif classification in [0, 1]:
                profile = 'absorption'
            elif classification is None:
                raise ValueError("classification must be specified if profile is not specified")
            elif not isinstance(classification, int):
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

    def fit(self, time=None, row=None, column=None, spectrum=None, profile=None, sigma=None, classifications=None,
            background=None, n_pools=None):
        """Fits the model to specified spectra

        Fits the model to an array of spectra using multiprocessing if requested.

        Parameters
        ----------
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list, a NumPy
            array, a Python range, etc. can be used.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectrum : ndarray, ndim=1, optional, default=None
            The explicit spectrum to fit the model to.
        profile : str, optional, default = None
            The profile to fit. (Will infer profile from `classifications` if omitted.)
        sigma : int or array_like, optional, default = None
            Explicit sigma index or profile. See `_get_sigma` for details.
        classifications : int, optional, default = None
            Classifications to determine the fitted profile to use (if profile not explicitly given). Will use
            neural network to classify them if not.
        background : float, optional, default = None
            If provided, this value will be subtracted from the explicit spectrum provided in `spectrum`. Will
            not be applied to spectra found from the indices, use the `load_background` method instead.
        n_pools : int, optional, default = None
            The number of processing pools to calculate the fitting over. This allocates the fitting of different
            spectra to `n_pools` separate worker processes. When processing a large number of spectra this will make
            the fitting process take less time overall. It also distributes such that each worker process has the
            same ratio of classifications to process. This should balance out the workload between workers.
            If few spectra are being fitted, performance may decrease due to the overhead associated with splitting
            the evaluation over separate processes. If `n_pools` is not an integer greater than zero, it will fit
            the spectrum with a for loop.

        Returns
        -------
        result : list of FitResult, length=n_spectra
            Outcome of the fits returned as a list of FitResult objects
        """
        # Specific fitting algorithm for IBIS Ca II 8542 Å

        # Only include the background is using an explicit spectrum; remove for indices
        include_background = False if spectrum is None else True
        spectra = self.get_spectra(time=time, row=row, column=column, spectrum=spectrum, background=include_background)

        # At least one valid spectrum must be given
        n_valid = np.sum(~np.isnan(spectra[..., 0]))
        if np.sum(~np.isnan(spectra[..., 0])) == 0:
            raise ValueError("no valid spectra given")

        if spectrum is not None and background is not None:  # remove explicit background of explicit spectrum
            spectra -= background

        if classifications is None:  # Classify if not already specified
            classifications = self.classify_spectra(spectra=spectra, only_normalise=True)

        if spectrum is None:  # No explicit spectrum given

            # Create the array of indices that the spectra represent
            time, row, column = make_iter(*self._get_time_row_column(time=time, row=row, column=column))
            indices = np.transpose(np.array(np.meshgrid(time, row, column, indexing='ij')), axes=[1, 2, 3, 0])

            # Make shape (n_spectra, n_features) so can process in a list
            spectra = spectra.reshape(-1, spectra.shape[-1])
            indices = indices.reshape(-1, indices.shape[-1])
            classifications = classifications.reshape(-1)

            # Remove spectra that are invalid (this allows for masking of the loaded data to constrain a region to fit)
            valid_spectra_i = np.where(~np.isnan(spectra[:, 0]))  # Where the first item of the spectrum is not NaN
            spectra = spectra[valid_spectra_i]
            indices = indices[valid_spectra_i]
            classifications = classifications[valid_spectra_i]

            if len(spectra) != len(indices) != len(classifications):  # Postprocessing sanity check
                raise ValueError("number of spectra, number of recorded indices and number of classifications"
                                 "are not the same (impossible error)")

            if n_pools is None or (isinstance(n_pools, int) and n_pools <= 0):  # Multiprocessing not required

                print("Processing {} spectra".format(n_valid))
                results = [self._fit(spectra[i], profile=profile, sigma=sigma, classification=classifications[i],
                                     spectrum_index=indices[i]) for i in range(len(spectra))]

            elif isinstance(n_pools, int) and n_pools >= 1:  # Use multiprocessing

                # Define single argument function that can be evaluated in the pools
                def func(data, profile=profile, sigma=sigma):
                    spectrum, index, classification = data  # Extract data and pass to `_fit` method
                    return self._fit(spectrum, profile=profile, sigma=sigma, classification=classification,
                                     spectrum_index=list(index))

                # Sort the arrays in descending classification order
                s = np.argsort(classifications)[::-1]  # Classifications indices sorted in descending order
                spectra = spectra[s]
                indices = indices[s]
                classifications = classifications[s]

                # Distribute the sorted spectra to each of `n_pools` in turn to ensure even workload
                data = []
                for pool in range(n_pools):
                    data += [[spectra[i], indices[i], classifications[i]] for i in range(pool, len(spectra), n_pools)]

                print("Processing {} spectra over {} pools".format(n_valid, n_pools))

                with Pool(n_pools) as p:  # Send the job to the `n_pools` pools
                    results = p.map(func, data)  # Map each element of `data` to the first argument of `func`
                    p.close()  # Finished, so no more jobs to come
                    p.join()  # Clean up the closed pools
                    p.clear()  # Remove the server so it can be created again

            else:
                raise TypeError("n_pools must be an integer, got %s" % type(n_pools))

        else:  # Explicit spectrum must be 1D so no loop needed
            results = self._fit(spectra, profile=profile, sigma=sigma, classification=classifications,
                                spectrum_index=None)

        return results

    def plot(self, fit=None, time=None, row=None, column=None, spectrum=None, classification=None, background=None,
             sigma=None, stationary_line_core=None, output=False, **kwargs):
        """Plots the data and fitted parameters

        Parameters
        ----------
        fit : FitResult or list or array_like, optional, default = None
            The fitted parameters to plot with the data. Can extract the necessary plot metadata from the fit object.
            Otherwise, `fit` should be the parameters to be fitted to either a Voigt or double Voigt profile depending
            on the number of parameters fitted.
        time : int or iterable, optional, default = None
            The time index. The index can be either a single integer index or an iterable. E.g. a list, a NumPy
            array, a Python range, etc. can be used.
        row : int or iterable, optional, default = None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default = None
            The column index. See comment for `time` parameter.
        spectrum : ndarray of length `original_wavelengths`, ndim=1, optional, default = None
            The explicit spectrum to plot along with a fit (if specified).
        classification : int, optional, default = None
            Used to determine which sigma profile to use. See `_get_sigma` for more details.
        background : float or array_like of length `constant_wavelengths`, optional, default = see note
            Background to added to the fitted profiles. If a `spectrum` is given, this will default to zero, otherwise
            the value loaded by `load_background` will be used.
        sigma : int or array_like, optional, default = None
            Explicit sigma index or profile. See `_get_sigma` for details.
        stationary_line_core : float, optional, default = `self.stationary_line_core`
            The stationary line core wavelength to mark on the plot.
        output : bool or str, optional, default = False
            Whether to save the plot to a file. If true, a file of format `plot_<time>_<row>_<column>.eps` will be
            created in the current directory. If a string, that will be used as the filename. (Can change filetype
            like this.) If false, no file will be created.
        **kwargs
            Parameters used by matplotlib and `separate` (see `plot_separate`) and `subtraction`
            (see `plot_subtraction`).
                - `figsize` passed to `matplotlib.pyplot.figure`
                - `legend_position` passed to `matplotlib.pyplot.legend`
                - `dpi` passed to `matplotlib.pyplot.figure` and `matplotlib.pyplot.savefig`
                - `fontfamily` passed to `matplotlib.pyplot.rc('font', family=`fontfamily`)` if given

        See Also
        --------
        plot_separate : Plot the fit parameters separately
        plot_subtraction : Plot the spectrum with the emission fit subtracted from it
        FitResult.plot : Plotting method on the fit result
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
        if spectrum.ndim != 1:
            raise ValueError("spectrum must be 1D")

        # If no background to be added to the fit is given...
        if background is None and fit is not None:
            if not explicit_spectrum:  # Take from loaded background if using indices
                time, row, column = self._get_time_row_column(time=time, row=row, column=column)
                if sum([isinstance(i, (int, np.int64, np.int32, np.int16, np.int8)) for i in [time, row, column]]) != 3:
                    raise TypeError("plot only accepts integer values for time, row and column")
                background = self.background[time, row, column]
            else:  # Otherwise assume to be zero
                background = 0

        if classification is not None or sigma is not None:  # Skip if neither classification or sigma are given
            sigma = self._get_sigma(classification, sigma)

        if stationary_line_core is None:
            stationary_line_core = self.stationary_line_core

        if isinstance(output, bool):
            if output:
                # Create a name
                output = "plot_{}_{}_{}.eps".format(time, row, column)
            else:
                output = None  # Do not output a file
        elif not isinstance(output, str):  # A valid string can pass through as the filename
            raise TypeError("output must be either boolean or a string, got %s" % type(output))

        plot_ibis8542(self.constant_wavelengths, spectrum, fit=fit, background=background, sigma=sigma,
                      stationary_line_core=stationary_line_core, output=output, **kwargs)

    def plot_separate(self, *args, **kwargs):
        """Plot the fitted profiles separately

        If multiple profiles exist, fit them separately. See `plot` for more details.

        See Also
        --------
        plot : General plotting method
        plot_subtraction : Plot the spectrum with the emission fit subtracted from it
        FitResult.plot : Plotting method on the fit result
        """
        self.plot(*args, separate=True, **kwargs)

    def plot_subtraction(self, *args, **kwargs):
        """Plot the spectrum with the emission fit subtracted from it

        If multiple profiles exist, subtract the fitted emission from the raw data. See `plot` for more details.

        See Also
        --------
        plot : General plotting method
        plot_separate : Plot the fit parameters separately
        FitResult.plot : Plotting method on the fit result
        """
        self.plot(*args, subtraction=True, **kwargs)
