import os
import warnings
import collections
import copy

import numpy as np
from scipy.optimize import curve_fit
from pathos.multiprocessing import ProcessPool as Pool
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from mcalf.utils.spec import reinterpolate_spectrum
from mcalf.utils.misc import load_parameter, make_iter


__all__ = ['ModelBase', 'BASE_PARAMETERS', 'BASE_ATTRIBUTES']


class ModelBase:
    """Base class for spectral line model fitting.

    .. warning::
        This class should not be used directly.
        Use derived classes instead.

    Parameters
    ----------
    ${PARAMETERS}

    Attributes
    ----------
    ${ATTRIBUTES}
    """
    def __init__(self, config=None, **kwargs):

        # STAGE 1: Define dictionary of default attribute values
        defaults = {
            'stationary_line_core': None,
            'original_wavelengths': None,
            'constant_wavelengths': None,
            'delta_lambda': 0.05,
            'sigma': None,
            'prefilter_response': None,
            'prefilter_ref_main': None,
            'prefilter_ref_wvscl': None,
            'output': None,
        }

        # STAGE 2: Update defaults with any values specified in a config file
        self.config = {}  # Dictionary of parameters from the config file
        if config is not None:  # Process config file if one is specified

            with open(config, 'r') as stream:  # Load YAML file
                self.config = load(stream, Loader=Loader)

            # Extract the BaseModel attributes from the config file
            class_defaults = {k: self.config[k] for k in self.config.keys() if k in defaults.keys()}
            for k in class_defaults.keys():
                if k not in ['output']:  # Keep string as is (don't pass through `load_parameter`)
                    class_defaults[k] = load_parameter(class_defaults[k])
                self.config.pop(k)  # Remove copied parameter

            # Update the defaults with the config file
            defaults.update(class_defaults)

        # STAGE 3: Update defaults with the keyword arguments passed into the class initialisation
        # Verify that only expected kwargs are passed
        for k in kwargs.keys():
            if k not in defaults.keys():
                raise TypeError(f"ModelBase() got an unexpected keyword argument '{k}'")
        defaults.update(kwargs)  # Update the defaults

        # Load default values of any parameters that haven't been given yet
        if defaults['original_wavelengths'] is None:
            raise ValueError("original_wavelengths must be specified")
        if defaults['constant_wavelengths'] is None:
            defaults['constant_wavelengths'] = np.arange(min(defaults['original_wavelengths']),
                                                         (max(defaults['original_wavelengths'])
                                                         + defaults['delta_lambda']),
                                                         defaults['delta_lambda'])

        # STAGE 4: Set the object attributes (with some type enforcing)
        # values in the defaults dict
        self.original_wavelengths = np.asarray(defaults['original_wavelengths'], dtype=np.float64)
        self.constant_wavelengths = np.asarray(defaults['constant_wavelengths'], dtype=np.float64)
        if defaults['stationary_line_core'] is None:  # Allow to be None by default
            self.__stationary_line_core = None  # Override setter
        else:
            self.stationary_line_core = defaults['stationary_line_core']
        self.__delta_lambda = defaults['delta_lambda']
        self.sigma = defaults['sigma']
        self.prefilter_response = defaults['prefilter_response']
        self.__prefilter_ref_main = defaults['prefilter_ref_main']
        self.__prefilter_ref_wvscl = defaults['prefilter_ref_wvscl']
        self.output = defaults['output']
        # attributes whose default value cannot be changed during initialisation
        self.neural_network = None  # Must be set in a child class initialisation or after initialisation
        self.array = None  # Array that will hold any loaded spectra
        self.background = None  # Array holding constant background values for `self.array`

        # STAGE 5: Validate the loaded attributes
        self._validate_base_attributes()

    @property
    def stationary_line_core(self):
        return self.__stationary_line_core

    @stationary_line_core.setter
    def stationary_line_core(self, wavelength):

        # Verify that stationary_line_core is a float
        if not isinstance(wavelength, float):
            raise ValueError("stationary_line_core must be a float, got %s" % type(wavelength))

        # Stationary wavelength must be within wavelength range
        original_diff = self.original_wavelengths - wavelength
        constant_diff = self.constant_wavelengths - wavelength
        for n, i in [['original_wavelengths', original_diff], ['constant_wavelengths', constant_diff]]:
            if min(i) > 1e-6 or max(i) < -1e-6:
                raise ValueError("`stationary_line_core` is not within `{}`".format(n))

        # Verification passed; set value
        self.__stationary_line_core = wavelength

    def _set_prefilter(self):
        """Set the `prefilter_response` parameter.

        .. deprecated:: 0.2
             Prefilter response correction code, and `prefilter_response`, `prefilter_ref_main`
             and `prefilter_ref_wvscl`, may be removed in a later release of MCALF.
             Spectra should be fully processed before loading into MCALF.

        This method should be called in a child class once `stationary_line_core` has been set.
        """
        if self.prefilter_response is None:
            if self.__prefilter_ref_main is not None and self.__prefilter_ref_wvscl is not None:
                self.prefilter_response = reinterpolate_spectrum(self.__prefilter_ref_main,
                                                                 self.__prefilter_ref_wvscl + self.stationary_line_core,
                                                                 self.constant_wavelengths)
            else:
                return  # None of the prefilter attributes are set, so no prefilter to apply.
        else:  # Make sure it is a numpy array so that division works as expected when doing array operations
            self.prefilter_response = np.asarray(self.prefilter_response, dtype=np.float64)

        # TODO: Remove this warning and possibly all prefilter code
        warnings.warn("Spectra should be fully processed before loading into MCALF. "
                      "Prefilter response correction code may be removed in a later "
                      "release.", PendingDeprecationWarning)

        # If a prefilter response is given it must be a compatible length
        if self.prefilter_response is not None:
            if len(self.prefilter_response) != len(self.constant_wavelengths):
                raise ValueError("prefilter_response array must be the same length as constant_wavelengths array")

    def _validate_base_attributes(self):
        """Validate some of the object's attributes.

        Raises
        ------
        ValueError
            To signal that an attribute is not valid.
        """
        # Wavelength arrays must be sorted ascending
        if np.sum(np.diff(self.original_wavelengths) > 0) < len(self.original_wavelengths) - 1:
            raise ValueError("original_wavelengths array must be sorted ascending")
        if np.sum(np.diff(self.constant_wavelengths) > 0) < len(self.constant_wavelengths) - 1:
            raise ValueError("constant_wavelengths array must be sorted ascending")

        # Warn if the constant wavelengths extrapolate the original wavelengths
        if min(self.constant_wavelengths) - min(self.original_wavelengths) < -1e-6:
            # If lower-bound of constant wavelengths is more than 1e-6 outside of the original wavelengths
            warnings.warn("Lower bound of `constant_wavelengths` is outside of `original_wavelengths` range.")
        if max(self.constant_wavelengths) - max(self.original_wavelengths) - self.__delta_lambda > 1e-6:
            # If upper-bound of constant wavelengths is more than 1e-6 ouside the original wavelengths
            warnings.warn("Upper bound of `constant_wavelengths` is outside of `original_wavelengths` range.")

    def _load_data(self, array, names=None, target=None):
        """Load a specified array into the model object.

        Load `array` with dimension names `names` into the attribute specified by `target`.

        Parameters
        ----------
        array : numpy.ndarray
            The array to load.

        names : list of str, length=`array.ndim`
            List of dimension names for `array`. Valid dimension names depend on `target`.

        target : {'array', 'background'}
            The attribute to load the `array` into.

        See Also
        --------
        load_array : Load and array of spectra.
        load_background : Load an array of spectral backgrounds.
        """

        # Validate specified `target`
        if target not in ['array', 'background']:
            raise ValueError("array target must be either 'array' or 'background', got '%s'" % target)

        # Verify `array` has data on more than one spectrum
        if (target == 'array' and np.ndim(array) == 1) or (target == 'background' and np.ndim(array) == 0):
            raise ValueError("cannot load an array containing one spectrum, use the spectrum directly instead")

        # Validate the dimension names given
        if names is None:
            raise ValueError("dimension names must be specified for each dimension")
        if np.ndim(array) != len(np.atleast_1d(names)):  # All dimensions must be labelled
            raise ValueError("number of dimension names do not match number of columns")
        if len(np.atleast_1d(names)) != len(np.unique(np.atleast_1d(names))):
            raise ValueError("duplicate dimension names found")
        if target == 'array' and 'wavelength' not in names:
            raise ValueError("array must contain a wavelength dimension")

        # # Pad input array to include all possible input dimensions and
        # #     then transpose it such that order of dimensions is consistent
        #   e.g. ['time', 'wavelength']           -> ['time', 'row', 'column', 'wavelength']
        #        ['wavelength', 'column', 'row']  -> ['time', 'row', 'column', 'wavelength']

        # Define the transposition on the input array to get the dimensions in order
        #     ['time', 'row', 'column'(, 'wavelength')]
        n_dims = 4 if target == 'array' else 3  # ndim of (padded) array to load data into
        transposition = [None] * n_dims
        for i in range(len(names)):
            if names[i] == 'time':
                transposition[0] = i
            elif names[i] == 'row':
                transposition[1] = i
            elif names[i] == 'column':
                transposition[2] = i
            elif names[i] == 'wavelength' and target == 'array':  # Only `target` 'array' should contain wavelengths
                transposition[3] = i
            else:  # Name given is not recognised
                raise ValueError("name '{}' is not a valid dimension name".format(names[i]))

        # Define where the new padded dimensions will be placed within the transposition
        array_n_dims = np.ndim(array)  # Number of dimensions in input array
        new_axis_id = array_n_dims  # Start labelling new axes after last dimension of input array
        for axis in range(n_dims):  # Loop through each axis of padded array to be created
            if transposition[axis] is None:  # Axis hasn't been labelled yet (not part of input array)
                transposition[axis] = new_axis_id  # Locate it within the transposition definition
                new_axis_id += 1

        # Add missing dimensions to the end
        n_dims_to_add = n_dims - array_n_dims
        ordered_array = array
        for i in range(n_dims_to_add):  # Add required number of dimensions to the end of the input array
            ordered_array = ordered_array[..., np.newaxis]

        # Reorder the dimensions of the padded array such that they can be later assumed
        ordered_array = np.transpose(ordered_array, transposition)

        # If spectral array, verify that length of each spectrum is equal to the number of corresponding wavelengths
        if target == 'array' and np.shape(ordered_array)[-1] != len(self.original_wavelengths):
            raise ValueError("length of wavelength dimension not equal length of original_wavelengths")

        # Load the processed array into the specified object parameter
        if target == 'array':
            self.array = ordered_array
        elif target == 'background':
            self.background = ordered_array

        # If "the other array" has been loaded, warn the user if their dimensions are not compatible
        if self.array is not None and self.background is not None and \
                np.sum(np.shape(self.array)[:-1] != np.shape(self.background)) != 0:
            warnings.simplefilter('always', UserWarning)
            warnings.warn("shape of spectrum array indices does not match shape of background array")

    def load_array(self, array, names=None):
        """Load an array of spectra.

        Load `array` with dimension names `names` into the `array` parameter of the model object.

        Parameters
        ----------
        array : numpy.ndarray, ndim>1
            An array containing at least two spectra.

        names : list of str, length=`array.ndim`
            List of dimension names for `array`. Valid dimension names are 'time', 'row', 'column' and 'wavelength'.
            'wavelength' is a required dimension.

        See Also
        --------
        load_background : Load an array of spectral backgrounds.

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> from astropy.io import fits
        >>> wavelengths = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        >>> model = mcalf.models.ModelBase(original_wavelengths=wavelengths)

        Load spectra from a file:

        >>> spectra = fits.open('spectra_0000.fits')[0].data  # doctest: +SKIP
        >>> model.load_array(spectra, names=['wavelength', 'column', 'row'])  # doctest: +SKIP
        """
        self._load_data(array, names=names, target='array')

    def load_background(self, array, names=None):
        """Load an array of spectral backgrounds.

        Load `array` with dimension names `names` into `background` parameter of the model object.

        Parameters
        ----------
        array : numpy.ndarray, ndim>0
            An array containing at least two backgrounds.

        names : list of str, length=`array.ndim`
            List of dimension names for `array`. Valid dimension names are 'time', 'row' and 'column'.

        See Also
        --------
        load_array : Load and array of spectra.

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> from astropy.io import fits
        >>> wavelengths = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        >>> model = mcalf.models.ModelBase(original_wavelengths=wavelengths)

        Load background array from a file:

        >>> background = fits.open('background_0000.fits')[0].data  # doctest: +SKIP
        >>> model.load_background(background, names=['column', 'row'])  # doctest: +SKIP
        """
        self._load_data(array, names=names, target='background')

    def train(self, X, y):
        """Fit the neural network model to spectra matrix X and spectra labels y.

        Calls the :meth:`fit` method on the `neural_network` parameter of the model object.

        Parameters
        ----------
        X : numpy.ndarray or sparse matrix, shape=(n_spectra, n_wavelengths)
            The input spectra.
        y : numpy.ndarray, shape= (n_spectra,) or (n_spectra, n_outputs)
            The target class labels.

        See Also
        --------
        test : Test how well the neural network has been trained.
        """
        self.neural_network.fit(X, y)

    def test(self, X, y):
        """Test the accuracy of the trained neural network.

        Prints a table of results showing:

        1) the percentage of predictions that equal the target labels;
        2) the average classification deviation and standard deviation from the ground truth classification
           for each labelled classification;
        3) the average classification deviation and standard deviation overall.

        If the model object has an output parameter, it will create a CSV file (``output``/neural_network/test.csv)
        listing the predictions and ground truth data.

        Parameters
        ----------
        X : numpy.ndarray or sparse matrix, shape=(n_spectra, n_wavelengths)
            The input spectra.
        y : numpy.ndarray, shape= (n_spectra,) or (n_spectra, n_outputs)
            The target class labels.

        See Also
        --------
        train : Train the neural network.
        """
        # Predict the labels
        try:
            predictions = self.neural_network.predict(X)
        except NotFittedError:
            raise NotFittedError("Neural network has not been trained yet. Call 'train' on the model class first.")
        if len(predictions) != len(y):  # Must be a corresponding ground truth label to compare to
            raise ValueError("number of samples X does not equal number of labels y")

        # Score is the percentage of predictions that are equal to their ground truth label
        score = np.sum(predictions == y) / len(predictions) * 100

        # Generate the table
        info = "+---------------------------------------------+\n"
        info += "|      Neural Network Testing Statistics      |\n"
        info += "+---------------------------------------------+\n"
        info += "| Percentage predictions==labels :: {: 6.2f}%   |\n".format(score)
        info += "+---------------------------------------------+\n"
        info += "| Average deviation for each classification   |\n"
        info += "+---------------------------------------------+\n"

        # Generate the average deviation for each classification
        classifications = np.sort(np.unique([predictions, y]))
        for classification in classifications:
            dev = predictions[np.where(y == classification)[0]] - classification
            info += "|     class {: 2d} :: {: 6.2f} ± {: 4.2f}              |\n".format(classification, np.mean(dev),
                                                                                        np.std(dev))
            info += "+---------------------------------------------+\n"

        # Generate the average deviation over all classifications
        dev = predictions - y
        info += "| Average deviation overall :: {: 6.2f} ± {: 4.2f} |\n".format(np.mean(dev), np.std(dev))
        info += "+---------------------------------------------+"

        # Print the table
        print(info)

        # Print scikit-learn classification report
        print(classification_report(y, predictions))

        # Output to file columns: 'ground_truth', 'predictions'
        if self.output is not None:
            try:
                os.makedirs(os.path.join(self.output, "neural_network"))
            except FileExistsError:
                pass
            np.savetxt(os.path.join(self.output, "neural_network", "test.csv"), np.asarray([y, predictions]).T,
                       fmt='%3d', delimiter=',', header='ground_truth, prediction')

    def classify_spectra(self, time=None, row=None, column=None, spectra=None, only_normalise=False):
        """Classify the specified spectra.

        Will also normalise each spectrum such that its intensity will range from zero to one.

        Parameters
        ----------
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list,
            a :class:`numpy.ndarray`, a Python range, etc. can be used.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectra : numpy.ndarray, optional, default=None
            The explicit spectra to classify. If `only_normalise` is False, this must be 1D.
            However, if `only_normalise` is set to true, `spectra` can be of any dimension.
            It is assumed that the final dimension is wavelengths, so return shape will be the
            same as `spectra`, except with no final wavelengths dimension.
        only_normalise : bool, optional, default=False
            Whether the single spectrum given  in `spectra` should not be interpolated and corrected.
            If set to true, the only processing applied to `spectra` will be a normalisation to be
            in range 0 to 1.

        Returns
        -------
        classifications : numpy.ndarray
            Array of classifications with the same time, row and column indices as `spectra`.

        See Also
        --------
        train : Train the neural network.
        test : Test the accuracy of the neural network.
        get_spectra : Get processed spectra from the objects `array` attribute.

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> import numpy as np
        >>> wavelengths = np.linspace(8542.1, 8542.2, 30)
        >>> model = mcalf.models.ModelBase(original_wavelengths=wavelengths)

        Load a trained neural network:

        >>> import pickle
        >>> pkl = open('trained_neural_network.pkl', 'rb')  # doctest: +SKIP
        >>> model.neural_network = pickle.load(pkl)  # doctest: +SKIP

        Classify an individual spectrum:

        >>> spectrum = np.random.rand(30)
        >>> model.classify_spectra(spectra=spectrum)  # doctest: +SKIP
        array([2])

        When :code:`only_normalise=True`, classify an n-dimensional spectral array:

        >>> spectra = np.random.rand(5, 4, 3, 2, 30)
        >>> model.classify_spectra(spectra=spectra, only_normalise=True).shape  # doctest: +SKIP
        (5, 4, 3, 2)

        Load spectra from a file and classify:

        >>> from astropy.io import fits
        >>> spectra = fits.open('spectra_0000.fits')[0].data  # doctest: +SKIP
        >>> model.load_array(spectra, names=['wavelength', 'column', 'row'])  # doctest: +SKIP
        >>> model.classify_spectra(column=range(10, 15), row=[7, 16])  # doctest: +SKIP
        array([[[0, 2, 0, 3, 0],
                [4, 0, 1, 0, 0]]])
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

    def _get_time_row_column(self, time=None, row=None, column=None):
        """Validate and infer the time, row and column index.

        Takes any time, row and column index given and if any are not specified, they are returned as 0 if the
        spectral array only has one value at its dimension. If there are multiple and no index is specified,
        an error is raised due to the ambiguity.

        Parameters
        ----------
        time : optional, default=None
            The time index.
        row : optional, default=None
            The row index.
        column : optional, default=None
            The column index.

        Returns
        -------
        time
            The corrected time index.
        row
            The corrected row index.
        column
            The corrected column index.

        See Also
        --------
        mcalf.utils.misc.make_iter : Make a variable iterable.

        Notes
        -----
        No type checking is done on the input indices so it can be anything but in most cases will need to be
        either an integer or iterable. The :func:`mcalf.utils.misc.make_iter` function can be used to make
        indices iterable.
        """
        array_shape = np.shape(self.array)
        if time is None:
            # No index was specified but we can infer that it would be 0 as there is only 1 value at this dimension
            if array_shape[0] == 1:
                time = 0
            else:
                raise ValueError("time index must be specified as multiple indices exist")
        if row is None:
            if array_shape[1] == 1:
                row = 0
            else:
                raise ValueError("row index must be specified as multiple indices exist")
        if column is None:
            if array_shape[2] == 1:
                column = 0
            else:
                raise ValueError("column index must be specified as multiple indices exist")
        return time, row, column

    def get_spectra(self, time=None, row=None, column=None, spectrum=None, correct=True, background=False):
        """Gets corrected spectra from the spectral array.

        Takes either a set of indices or an explicit spectrum and optionally applied corrections and background
        removal.

        Parameters
        ----------
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list,
            a :class:`numpy.ndarray`, a Python range, etc. can be used.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectrum : ndarray of ndim=1, optional, default=None
            The explicit spectrum. If provided, `time`, `row`, and `column` are ignored.
        correct : bool, optional, default=True
            Whether to reinterpolate the spectrum and apply the prefilter correction (if exists).
        background : bool, optional, default=False
            Whether to include the background in the outputted spectra. Only removes the background if the
            relevant background array has been loaded. Does not remove background is processing an explicit spectrum.

        Returns
        -------
        spectra : ndarray

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> import numpy as np
        >>> wavelengths = np.linspace(8541.3, 8542.7, 30)
        >>> model = mcalf.models.ModelBase(original_wavelengths=wavelengths)

        Provide a single spectrum for processing, and notice output is 1D:

        >>> spectrum = model.get_spectra(spectrum=np.random.rand(30))
        >>> spectrum.ndim
        1

        Load an array of spectra:

        >>> spectra = np.random.rand(3, 4, 30)
        >>> model.load_array(spectra, names=['column', 'row', 'wavelength'])

        Extract a single (unprocessed) spectrum from the loaded array, and notice output is 4D:

        >>> spectrum = model.get_spectra(row=1, column=0, correct=False)
        >>> spectrum.shape
        (1, 1, 1, 30)
        >>> (spectrum[0, 0, 0] == spectra[0, 1]).all()
        True

        Extract an array of spectra, and notice output is 4D, and with dimensions
        time, row, column, wavelength regardless of the original dimensions and order:

        >>> spectrum = model.get_spectra(row=range(4), column=range(3))
        >>> spectrum.shape
        (1, 4, 3, 30)

        Notice that the time index can be excluded, as the loaded array only represents a single time.
        However, in this case leaving out `row` or `column` results in an error as it is ambiguous:

        >>> spectrum = model.get_spectra(row=range(4))
        Traceback (most recent call last):
         ...
        ValueError: column index must be specified as multiple indices exist
        """
        # Locate the spectra
        if spectrum is None:  # No explicit spectrum so use specified indices
            time, row, column = make_iter(*self._get_time_row_column(time=time, row=row, column=column))
            spectra = self.array[time][:, row][:, :, column]  # Iterable indices are needed to crop array correctly
        else:
            if np.ndim(spectrum) != 1:  # Keep things simple (for now)
                raise ValueError("explicit spectrum must have one dimension, got %s" % np.ndim(spectrum))
            spectra = spectrum

        # Apply option corrections to spectra
        if correct:
            spectra_list = spectra.reshape(-1, spectra.shape[-1])  # To shape: (n_spectra, n_wavelengths)
            corrected_spectra = np.array([
                reinterpolate_spectrum(spectrum, self.original_wavelengths, self.constant_wavelengths)
                for spectrum in spectra_list])  # Reinterpolate each spectrum

            if self.prefilter_response is not None:  # Apply prefilter correction if one was initialised
                corrected_spectra /= self.prefilter_response

            spectra = corrected_spectra.reshape(*spectra.shape[:-1], -1)  # Revert to the original shape

        # Apply optional background removal to spectra
        if spectrum is None and not background and self.background is not None:  # Remove the background
            # Calculate transpositions to get the array in the right shape for a quick background subtraction
            forward_transpose = list(np.roll(np.arange(spectra.ndim), 1))
            backwards_transpose = list(np.roll(np.arange(spectra.ndim), -1))
            spectra = np.transpose(spectra, axes=forward_transpose) - self.background[time][:, row][:, :, column]
            spectra = np.transpose(spectra, axes=backwards_transpose)  # Revert to original shape

        return spectra

    def _fit(self, spectrum, classification=None, spectrum_index=None):
        """Fit a single spectrum for the given profile or classification.

        .. warning::
            This call signature and docstring specify how the `_fit` method must be implemented in
            each subclass of `ModelBase`. **It is not implemented in this class.**

        Parameters
        ----------
        spectrum : numpy.ndarray, ndim=1, length=n_constant_wavelengths
            The spectrum to be fitted.
        classification : int, optional, default=None
            Classification to determine the fitted profile to use.
        spectrum_index : array_like or list or tuple, length=3, optional, default=None
            The [time, row, column] index of the `spectrum` provided. Only used for error reporting.

        Returns
        -------
        result : mcalf.models.FitResult
            Outcome of the fit returned in a :class:`mcalf.models.FitResult` object.

        See Also
        --------
        fit : The recommended method for fitting spectra.
        mcalf.models.FitResult : The object that the fit method returns.

        Notes
        -----
        This method is called for each requested spectrum by the :meth:`models.ModelBase.fit` method.
        This is where most of the adjustments to the fitting method should be made. See other
        subclasses of `models.ModelBase` for examples of how to implement this method in a
        new subclass. See :meth:`models.ModelBase.fit` for more information on how this method is
        called.
        """
        raise NotImplementedError("The `_fit` method must be implemented in a subclass of `ModelBase`."
                                  "See `models.ModelBase._fit` for more details.")

    def fit(self, time=None, row=None, column=None, spectrum=None, classifications=None,
            background=None, n_pools=None, **kwargs):
        """Fits the model to specified spectra.

        Fits the model to an array of spectra using multiprocessing if requested.

        Parameters
        ----------
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list,
            :class:`numpy.ndarray`, a Python range, etc. can be used.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectrum : numpy.ndarray, ndim=1, optional, default=None
            The explicit spectrum to fit the model to.
        classifications : int or array_like, optional, default=None
            Classifications to determine the fitted profile to use. Will use neural network to classify them if not.
            If a multidimensional array, must have the same shape as [`time`, `row`, `column`].
            Dimensions that would have length of 1 can be excluded.
        background : float, optional, default=None
            If provided, this value will be subtracted from the explicit spectrum provided in `spectrum`. Will
            not be applied to spectra found from the indices, use the :meth:`~mcalf.models.ModelBase.load_background`
            method instead.
        n_pools : int, optional, default=None
            The number of processing pools to calculate the fitting over. This allocates the fitting of different
            spectra to `n_pools` separate worker processes. When processing a large number of spectra this will make
            the fitting process take less time overall. It also distributes such that each worker process has the
            same ratio of classifications to process. This should balance out the workload between workers.
            If few spectra are being fitted, performance may decrease due to the overhead associated with splitting
            the evaluation over separate processes. If `n_pools` is not an integer greater than zero, it will fit
            the spectrum with a for loop.
        **kwargs : dict, optional
            Extra keyword arguments to pass to :meth:`~mcalf.models.ModelBase._fit`.

        Returns
        -------
        result : list of :class:`~mcalf.models.FitResult`, length=n_spectra
            Outcome of the fits returned as a list of :class:`~mcalf.models.FitResult` objects.

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> import numpy as np
        >>> wavelengths = np.linspace(8541.3, 8542.7, 30)
        >>> model = mcalf.models.ModelBase(original_wavelengths=wavelengths)

        Set up the neural network classifier:

        >>> model.neural_network = ...  # load an untrained classifier  # doctest: +SKIP
        >>> model.train(...)  # doctest: +SKIP
        >>> model.test(...)  # doctest: +SKIP

        Load the spectra and background array:

        >>> model.load_array(...)  # doctest: +SKIP
        >>> model.load_background(...)  # doctest: +SKIP

        Fit a subset of the loaded spectra, using 5 processing pools:

        >>> fits = model.fit(row=range(3, 5), column=range(200), n_pools=5)  # doctest: +SKIP
        >>> fits  # doctest: +SKIP
        ['Successful FitResult with ________ profile of classification 0',
         'Successful FitResult with ________ profile of classification 2',
         ...
         'Successful FitResult with ________ profile of classification 0',
         'Successful FitResult with ________ profile of classification 4']

        Merge the fit results into a :class:`~mcalf.models.FitResults` object:

        >>> results = mcalf.models.FitResults((500, 500), 8)
        >>> for fit in fits:  # doctest: +SKIP
        ...     results.append(fit)  # doctest: +SKIP

        See :meth:`fit_spectrum` examples for how to manually providing a `spectrum` to fit.
        """
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

            assert len(spectra) == len(indices) == len(classifications)  # Postprocessing sanity check

            # Multiprocessing not required
            if n_pools is None or (isinstance(n_pools, (int, np.integer)) and n_pools <= 0):

                print("Processing {} spectra".format(n_valid))
                results = [
                    self._fit(spectra[i], classification=classifications[i], spectrum_index=indices[i], **kwargs)
                    for i in range(len(spectra))
                ]

            elif isinstance(n_pools, (int, np.integer)) and n_pools >= 1:  # Use multiprocessing

                # Define single argument function that can be evaluated in the pools
                def func(data, kwargs=kwargs):
                    # Extract data and pass to `_fit` method
                    spectrum, index, classification = data  # pragma: no cover
                    return self._fit(spectrum, classification=classification,
                                     spectrum_index=list(index), **kwargs)  # pragma: no cover

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
            results = self._fit(spectra, classification=classifications, spectrum_index=None, **kwargs)

        return results

    def fit_spectrum(self, spectrum, **kwargs):
        """Fits the specified spectrum array.

        Passes the spectrum argument to the :meth:`~mcalf.models.ModelBase.fit` method.
        For easily iterating over a list of spectra.

        Parameters
        ----------
        spectrum : numpy.ndarray, ndim=1
            The explicit spectrum.
        **kwargs : dict, optional
            Extra keyword arguments to pass to :meth:`~mcalf.models.ModelBase.fit`.

        Returns
        -------
        result : :class:`~mcalf.models.FitResult`
            Result of the fit.

        See Also
        --------
        fit : General fitting method.

        Examples
        --------

        Create a basic model:

        >>> import mcalf.models
        >>> import numpy as np
        >>> wavelengths = np.linspace(8541.3, 8542.7, 30)
        >>> model = mcalf.models.ModelBase(original_wavelengths=wavelengths)

        **Quickly provide a spectrum and fit it.** Remember that the model must be optimised for
        the spectra that it is asked to fit. In this example the neural network is not
        called upon to classify the provided spectrum as a classification is provided directly:

        >>> spectrum = np.random.rand(30)
        >>> model.fit_spectrum(spectrum, classifications=0, background=142.2)  # doctest: +SKIP
        Successful FitResult with ________ profile of classification 0

        As the spectrum is provided manually, any background value must also be provided manually.
        Alternatively, the background can be subtracted before passing to the function, as by
        default, no background is subtracted:

        >>> model.fit_spectrum(spectrum - 142.2, classifications=0)  # doctest: +SKIP
        Successful FitResult with ________ profile of classification 0
        """
        return self.fit(spectrum=spectrum, **kwargs)

    def _curve_fit(self, model, spectrum, guess, sigma, bounds, x_scale, time=None, row=None, column=None):
        """:func:`scipy.optimize.curve_fit` wrapper with error handling.

        Passes a certain set of parameters to the :func:`scipy.optimize.curve_fit` function and catches some typical
        errors, presenting a more specific warning message.

        Parameters
        ----------
        model : callable
            The model function, f(x, …). It must take the `ModelBase.constant_wavelenghts` attribute
            as the first argument and the parameters to fit as separate remaining arguments.
        spectrum : array_like
            The dependent data, with length equal to that of the `ModelBase.constant_wavelengths` attribute.
        guess : array_like, optional
            Initial guess for the parameters to fit.
        sigma : array_like
            Determines the uncertainty in the `spectrum`. Used to weight certain regions of the spectrum.
        bounds : 2-tuple of array_like
            Lower and upper bounds on each parameter.
        x_scale : array_like
            Characteristic scale of each parameter.
        time : optional, default=None
            The time index for error handling.
        row : optional, default=None
            The row index for error handling.
        column : optional, default=None
            The column index for error handling.

        Returns
        -------
        fitted_parameters : numpy.ndarray, length=n_parameters
            The parameters that recreate the model fitted to the spectrum.
        success : bool
            Whether the fit was successful or an error had to be handled.

        See Also
        --------
        fit : General fitting method.
        fit_spectrum : Explicit spectrum fitting method.

        Notes
        -----
        More details can be found in the documentation for :func:`scipy.optimize.curve_fit`
        and :func:`scipy.optimize.least_squares`.
        """
        try:  # TODO Investigate if there is a performance gain to setting `check_finite` to False

            fitted_parameters = curve_fit(model, self.constant_wavelengths, spectrum,
                                          p0=guess, sigma=sigma, bounds=bounds, x_scale=x_scale)[0]
            success = True

        except RuntimeError:

            success = False
            location_text = " at ({}, {}, {})".format(time, row, column) if time is not None else ''
            warnings.warn("RuntimeError{}".format(location_text))
            fitted_parameters = np.full_like(guess, np.nan)

        except ValueError as e:

            success = False
            if str(e) != "`x0` violates bound constraints.":
                raise
            else:
                location_text = " at ({}, {}, {})".format(time, row, column) if time is not None else ''
                warnings.warn("`x0` violates bound constraints{}".format(location_text))
                fitted_parameters = np.full_like(guess, np.nan)

        except np.linalg.LinAlgError as e:

            success = False
            if str(e) != "SVD did not converge in Linear Least Squares":
                raise
            else:
                location_text = " at ({}, {}, {})".format(time, row, column) if time is not None else ''
                warnings.warn("SVD did not converge in Linear Least Squares{}.".format(location_text))
                fitted_parameters = np.full_like(guess, np.nan)

        return fitted_parameters, success


# Define each parameter and attribute in an ordered dictionary so definitions can be passed to child objects
DOCS = collections.OrderedDict()
DOCS['original_wavelengths'] = """
    original_wavelengths : array_like
        One-dimensional array of wavelengths that correspond to the uncorrected spectral data."""
DOCS['stationary_line_core'] = """
    stationary_line_core : float, optional, default=None
        Wavelength of the stationary line core."""
DOCS['neural_network'] = """
    neural_network : optional, default=None
        The neural network classifier object that is used to classify spectra. This attribute should be set by a
        child class of :class:`~mcalf.models.ModelBase`."""
DOCS['constant_wavelengths'] = """
    constant_wavelengths : array_like, ndim=1, optional, default= see description
        The desired set of wavelengths that the spectral data should be rescaled to represent. It is assumed
        that these have constant spacing, but that may not be a requirement if you specify your own array.
        The default value is an array from the minimum to the maximum wavelength of `original_wavelengths` in
        constant steps of `delta_lambda`, overshooting the upper bound if the maximum wavelength has not been 
        reached."""
DOCS['delta_lambda'] = """
    delta_lambda : float, optional, default=0.05
        The step used between each value of `constant_wavelengths` when its default value has to be calculated."""
DOCS['sigma'] = """
    sigma : optional, default=None
        Sigma values used to weight the fit. This attribute should be set by a child class of 
        :class:`~mcalf.models.ModelBase`."""
DOCS['prefilter_response'] = """
    prefilter_response : array_like, length=n_wavelengths, optional, default= see note
        Each constant wavelength scaled spectrum will be corrected by dividing it by this array. If `prefilter_response`
        is not given, and `prefilter_ref_main` and `prefilter_ref_wvscl` are not given, `prefilter_response` will have a
        default value of `None`."""
DOCS['prefilter_ref_main'] = """
    prefilter_ref_main : array_like, optional, default= None
        If `prefilter_response` is not specified, this will be used along with `prefilter_ref_wvscl` to generate the
        default value of `prefilter_response`."""
DOCS['prefilter_ref_wvscl'] = """
    prefilter_ref_wvscl : array_like, optional, default=None
        If `prefilter_response` is not specified, this will be used along with `prefilter_ref_main` to generate the
        default value of `prefilter_response`."""
DOCS['config'] = """
    config : str, optional, default=None
        Filename of a `.yml` file (relative to current directory) containing the initialising parameters for this
        object. Parameters provided explicitly to the object upon initialisation will override any provided in this
        file. All (or some) parameters that this object accepts can be specified in this file, except `neural_network`
        and `config`. Each line of the file should specify a different parameter and be formatted like
        `emission_guess: '[-inf, wl-0.15, 1e-6, 1e-6]'` or `original_wavelengths: 'original.fits'` for example.
        When specifying a string, use 'inf' to represent `np.inf` and 'wl' to represent `stationary_line_core` as shown.
        If the string matches a file, :func:`mcalf.utils.misc.load_parameter()` is used to load the contents 
        of the file."""
DOCS['output'] = """
    output : str, optional, default=None
        If the program wants to output data, it will place it relative to the location specified by this parameter.
        Some methods will only save data to a file if this parameter is not `None`. Such cases will be documented
        where relevant."""
DOCS['array'] = """
    array: numpy.ndarray, dimensions are ['time', 'row', 'column', 'spectra']
        Array holding spectra."""
DOCS['background'] = """
    background: numpy.ndarray, dimensions are ['time', 'row', 'column']
        Array holding spectral backgrounds."""

# Form the parameter list
BASE_PARAMETERS = copy.deepcopy(DOCS)
for k in [  # Remove some entries
    'neural_network',
    'array',
    'background',
]:
    del BASE_PARAMETERS[k]
BASE_PARAMETERS_STR = ''.join(BASE_PARAMETERS[i] for i in BASE_PARAMETERS)

# Form the attribute list
BASE_ATTRIBUTES = copy.deepcopy(DOCS)
for k in [  # Remove some entries
    'delta_lambda',
    'prefilter_ref_main',
    'prefilter_ref_wvscl',
    'config',
]:
    del BASE_ATTRIBUTES[k]
BASE_ATTRIBUTES_STR = ''.join(BASE_ATTRIBUTES[i] for i in BASE_ATTRIBUTES)

# Update the docstring with the generated strings
ModelBase.__doc__ = ModelBase.__doc__.replace(
    '${PARAMETERS}',
    BASE_PARAMETERS_STR.lstrip()
)
ModelBase.__doc__ = ModelBase.__doc__.replace(
    '${ATTRIBUTES}',
    BASE_ATTRIBUTES_STR.lstrip()
)
