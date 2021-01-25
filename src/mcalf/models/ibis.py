import copy

import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from mcalf.models.results import FitResult
from mcalf.models.base import *
from mcalf.profiles.voigt import voigt_nobg, double_voigt_nobg
from mcalf.utils.spec import generate_sigma
from mcalf.utils.misc import load_parameter, make_iter
from mcalf.visualisation.spec import plot_ibis8542


__all__ = ['IBIS8542Model']


class IBIS8542Model(ModelBase):
    """Class for working with IBIS 8542 Å calcium II spectral imaging observations
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
        ]  # These must match dictionary in STAGE 1 (defined there as stationary_line_core needs to be set)
        base_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in class_keys}
        super().__init__(**base_kwargs)

        # STAGE 0B: Load child default values for parent class attributes
        # stationary_line_core
        if self.stationary_line_core is None:
            self.stationary_line_core = 8542.099145376844
        # prefilter_response
        self._set_prefilter()  # Update the prefilter using stationary_line_core
        # neural_network
        if self.neural_network is None:
            mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,), max_iter=1000)
            parameter_space = {'alpha': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]}  # Search region
            # Set GridSearchCV to find best alpha
            self.neural_network = GridSearchCV(mlp, parameter_space, cv=5, n_jobs=-1)
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
        }
        assert defaults.keys() == {k: None for k in class_keys}.keys()  # keys of `defaults` must match `class_keys`

        # STAGE 2: Update defaults with any values specified in a config file
        class_defaults = {k: self.config[k] for k in self.config.keys() if k in defaults.keys()}
        for k in class_defaults.keys():
            if k in ['absorption_x_scale', 'emission_x_scale']:  # These should not need the stationary line core
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

        # STAGE 5: Validate the loaded attributes
        self._validate_attributes()

    def _validate_attributes(self):
        """Validate some of the object's attributes

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
            if isinstance(sigma, (int, np.integer)):
                return self.sigma[sigma]
            else:
                return np.asarray(sigma, dtype=np.float64)

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
        classifications : int or array_like, optional, default = None
            Classifications to determine the fitted profile to use (if profile not explicitly given). Will use
            neural network to classify them if not. If a multidimensional array, must have the same shape as
            [`time`, `row`, `column`]. Dimensions that would have length of 1 can be excluded.
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

            # Ensure that length of arrays match
            spectra_indices_shape_mismatch = spectra.shape[:-1] != indices.shape[:-1]
            spectra_class_size_mismatch = np.size(spectra[..., 0]) != np.size(classifications)
            spectra_class_shape_mismatch = False  # Only test this if appropriate
            if isinstance(classifications, np.ndarray) and classifications.ndim != 1:
                # If a multidimensional array of classifications are given, make sure it matches the indices layout
                # Allow for dimensions of length 1 to be excluded
                if np.squeeze(spectra[..., 0]).shape != np.squeeze(classifications).shape:
                    spectra_class_shape_mismatch = True
            if spectra_indices_shape_mismatch or spectra_class_size_mismatch or spectra_class_shape_mismatch:
                raise ValueError("number classifications do not match number of spectra and associated indices")

            # Make shape (n_spectra, n_features) so can process in a list
            spectra = spectra.reshape(-1, spectra.shape[-1])
            indices = indices.reshape(-1, indices.shape[-1])
            classifications = np.asarray(classifications)  # Make sure ndarray methods are available
            classifications = classifications.reshape(-1)

            # Remove spectra that are invalid (this allows for masking of the loaded data to constrain a region to fit)
            valid_spectra_i = np.where(~np.isnan(spectra[:, 0]))  # Where the first item of the spectrum is not NaN
            spectra = spectra[valid_spectra_i]
            indices = indices[valid_spectra_i]
            classifications = classifications[valid_spectra_i]

            if len(spectra) != len(indices) != len(classifications):  # Postprocessing sanity check
                raise ValueError("number of spectra, number of recorded indices and number of classifications"
                                 "are not the same (impossible error)")

            # Multiprocessing not required
            if n_pools is None or (isinstance(n_pools, (int, np.integer)) and n_pools <= 0):

                print("Processing {} spectra".format(n_valid))
                results = [self._fit(spectra[i], profile=profile, sigma=sigma, classification=classifications[i],
                                     spectrum_index=indices[i]) for i in range(len(spectra))]

            elif isinstance(n_pools, (int, np.integer)) and n_pools >= 1:  # Use multiprocessing

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
                if sum([isinstance(i, (int, np.integer)) for i in [time, row, column]]) != 3:
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


IBIS8542_PARAMETERS = copy.deepcopy(BASE_PARAMETERS)
IBIS8542_ATTRIBUTES = copy.deepcopy(BASE_ATTRIBUTES)

for d in [IBIS8542_PARAMETERS, IBIS8542_ATTRIBUTES]:
    d['stationary_line_core'] = """
    stationary_line_core : float, optional, default = 8542.099145376844
        Wavelength of the stationary line core."""
    d['sigma'] = """
    sigma : list of array_like or bool, length=(2, n_wavelengths), optional, default = [type1, type2]
        A list of different sigma that are used to weight particular wavelengths along the spectra when fitting. The
        fitting method will expect to be able to choose a sigma array from this list at a specific index. It's default
        value is `[generate_sigma(i, constant_wavelengths, stationary_line_core) for i in [1, 2]]`.
        See `utils.generate_sigma()` for more information. If bool, True will generate the default sigma value
        regardless of the value specified in `config`, and False will set `sigma` to be all ones, effectively disabling
        it."""

IBIS8542_ATTRIBUTES['neural_network'] = """
    neural_network : sklearn.neural_network.MLPClassifier, optional, default = see description
        The MLPClassifier object (or similar) that will be used to classify the spectra. Defaults to a `GridSearchCV`
        with `MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,), max_iter=1000)`
        for best `alpha` selected from `[1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]`."""

IBIS8542_DOCS = """        
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
        Characteristic scale for all the emission Voigt profile parameters in order of the function's arguments."""

IBIS8542_PARAMETERS_STR = ''.join(IBIS8542_PARAMETERS[i] for i in IBIS8542_PARAMETERS)
IBIS8542_ATTRIBUTES_STR = ''.join(IBIS8542_ATTRIBUTES[i] for i in IBIS8542_ATTRIBUTES)

IBIS8542Model.__doc__ += """
    Parameters
    ----------""" + IBIS8542_DOCS + IBIS8542_PARAMETERS_STR + """

    Attributes
    ----------""" + IBIS8542_DOCS + """
    quiescent_wavelength : int, default = 1
        The index within the fitted parameters of the absorption Voigt line core wavelength.
    active_wavelength : int, default = 5
        The index within the fitted parameters of the emission Voigt line core wavelength.""" + IBIS8542_ATTRIBUTES_STR
