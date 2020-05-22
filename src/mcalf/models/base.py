import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report

from mcalf.utils.spec import reinterpolate_spectrum
from mcalf.utils.misc import make_iter


__all__ = ['ModelBase']


class ModelBase:
    """Base class for spectral line model fitting.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes
    ----------
    array : ndarray, dimensions are ['time', 'row', 'column', 'spectra']
        Array holding spectra.
    background : ndarray, dimensions are ['time', 'row', 'column']
        Array holding spectral backgrounds.
    """
    def __init__(self):
        self.array = None  # Array that will hold any loaded spectra
        self.background = None  # Array holding constant background values for `self.array`

    def _load_data(self, array, names=None, target=None):
        """Load a specified array into the model object.

        Load `array` with dimension names `names` into the parameter specified by `target`.

        Parameters
        ----------
        array : ndarray
            The array to load.

        names : list of str, length = `array.ndims`
            List of dimension names for `array`. Valid dimension names depend on `target`.

        target : {'array', 'background'}

        See Also
        --------
        load_array : Load and array of spectra
        load_background : Load an array of spectral backgrounds
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
        n_dims = 4 if target == 'array' else 3  # ndims of (padded) array to load data into
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
        else:  # This should not happen
            raise ValueError("unknown target (impossible error)")

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
        array : ndarray of ndims > 1
            An array containing at least two spectra.

        names : list of str, length = `array.ndims`
            List of dimension names for `array`. Valid dimension names are 'time', 'row', 'column' and 'wavelength'.
            'wavelength' is a required dimension.

        See Also
        --------
        load_background : Load an array of spectral backgrounds
        """
        self._load_data(array, names=names, target='array')

    def load_background(self, array, names=None):
        """Load an array of spectral backgrounds.

        Load `array` with dimension names `names` into `background` parameter of the model object.

        Parameters
        ----------
        array : ndarray of ndim>0
            An array containing at least two backgrounds.

        names : list of str, length = `array.ndims`
            List of dimension names for `array`. Valid dimension names are 'time', 'row' and 'column'.

        See Also
        --------
        load_array : Load and array of spectra
        """
        self._load_data(array, names=names, target='background')

    def train(self, X, y):
        """Fit the neural network model to spectra matrix X and spectra labels y.

        Calls the `fit` method on the `neural_network` parameter of the model object.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_spectra, n_wavelengths)
            The input spectra.
        y : ndarray of shape (n_spectra,) or (n_spectra, n_outputs)
            The target class labels.

        See Also
        --------
        test : Test how well the neural network has been trained
        """
        self.neural_network.fit(X, y)

    def test(self, X, y):
        """Test the accuracy of the trained neural network.

        Prints a table of results showing:
            1) the percentage of predictions that equal the target labels;
            2) the average classification deviation and standard deviation from the ground truth classification
                for each labelled classification;
            3) the average classification deviation and standard deviation overall.
        If the model object has an output parameter, it will create a CSV file (`self.output`/neural_network/test.csv)
        listing the predictions and ground truth data.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_spectra, n_wavelengths)
            The input spectra.
        y : ndarray of shape (n_spectra,) or (n_spectra, n_outputs)
            The target class labels.

        See Also
        --------
        train : Train the neural network
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

    def _get_time_row_column(self, time=None, row=None, column=None):
        """Validate and infer the time, row and column index

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
        utils.make_iter : Make a variable iterable

        Notes
        -----
        No type checking is done on the input indices so it can be anything but in most cases will need to be
        either an integer or iterable. The `utils.make_iter` function can be used to make indices iterable.
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
        """Gets corrected spectra from the spectral array

        Takes either a set of indices or an explicit spectrum and optionally applied corrections and background
        removal.

        Parameters
        ----------
        time : int or iterable, optional, default=None
            The time index. The index can be either a single integer index or an iterable. E.g. a list, a NumPy
            array, a Python range, etc. can be used.
        row : int or iterable, optional, default=None
            The row index. See comment for `time` parameter.
        column : int or iterable, optional, default=None
            The column index. See comment for `time` parameter.
        spectrum : ndarray of ndim=1, optional, default=None
            The explicit spectrum.
        correct : bool, optional, default=True
            Whether to reinterpolate the spectrum and apply the prefilter correction (if exists).
        background : bool, optional, default=False
            Whether to include the background in the outputted spectra. Only removes the background if the
            relevant background array has been loaded. Does not remove background is processing an explicit spectrum.
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

    def fit_spectrum(self, spectrum, **kwargs):
        """Fits the specified spectrum array

        Passes the spectrum argument to the fit method. For easily iterating over a list of spectra.

        Parameters
        ----------
        spectrum : ndarray of ndim=1
            The explicit spectrum.
        **kwargs : dictionary, optional
            Extra keyword arguments to pass to fit.

        Returns
        -------
        result : FitResult
            Result of the fit.

        See Also
        --------
        fit : General fitting method
        """
        return self.fit(spectrum=spectrum, **kwargs)

    def _curve_fit(self, model, spectrum, guess, sigma, bounds, x_scale, time=None, row=None, column=None):
        """scipy.optimize.curve_fit wrapper with error handling

        Passes a certain set of parameters to the scipy.optimize.curve_fit function and catches some typical
        errors, presenting a more specific warning message.

        Parameters
        ----------
        model : callable
            The model function, f(x, …). It must take the `self.constant_wavelenghts` as the first argument and the
            parameters to fit as separate remaining arguments.
        spectrum : array_like
            The dependent data, with length equal to that of `self.constant_wavelengths`.
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
        fitted_parameters : ndarray, length=n_parameters
            The parameters that recreate the model fitted to the spectrum.
        success : bool
            Whether the fit was successful or an error had to be handled.

        See Also
        --------
        fit : General fitting method
        fit_spectrum : Explicit spectrum fitting method

        Notes
        -----
        More details can be found in the documentation for scipy.optimize.curve_fit and scipy.optimize.least_squares.
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
