import pytest
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from mcalf.models import ModelBase, IBIS8542Model, FitResults
from mcalf.profiles.voigt import voigt, double_voigt

from ..helpers import data_path_function, figure_test
data_path = data_path_function('models')


class DummyClassifier:

    def __init__(self, trained=False, n_features=None, classifications=None):
        self.trained = trained
        self.n_features = n_features
        self.defined_return = True if classifications is not None else False
        self.classifications = classifications

    def train(self, X, y):
        assert np.ndim(X) == 2  # (n_samples, n_features)
        assert np.ndim(y) == 1  # (n_samples)
        assert len(X) == len(y)
        self.trained = True
        self.n_features = X.shape[-1]

    def test(self, X, y):
        if not self.trained:
            raise NotFittedError()
        assert np.ndim(X) == 2  # (n_samples, n_features)
        assert np.ndim(y) == 1  # (n_samples)
        assert X.shape[-1] == self.n_features
        assert len(X) == len(y)

    def predict(self, X):
        if self.defined_return:
            return self.classifications.flatten()
        if not self.trained:
            raise NotFittedError()
        assert np.ndim(X) == 2  # (n_samples, n_features)
        assert X.shape[-1] == self.n_features
        return np.asarray(X[:, -1] * 100, dtype=int)


@pytest.mark.parametrize('model', (ModelBase, IBIS8542Model))
def test_default_parameters(model):
    # Test default parameters
    with pytest.raises(ValueError) as e:
        model()
    assert 'original_wavelengths' in str(e.value)


def test_ibis8542model_basic():
    # Will break if default parameters are changes in mcalf.models.IBIS8542Model
    wl = 1000.97
    x_orig = np.linspace(999.81, 1002.13, num=25)
    prefilter_main = 1 - np.abs(x_orig - wl) * 0.1
    prefilter_wvscl = x_orig - wl
    m = IBIS8542Model(stationary_line_core=wl, original_wavelengths=x_orig,
                      prefilter_ref_main=prefilter_main, prefilter_ref_wvscl=prefilter_wvscl)

    bg = 1327.243
    arr = voigt(x_orig, -231.42, wl+0.05, 0.2, 0.21, bg)

    m.load_array(np.array([arr, arr]), names=['row', 'wavelength'])
    m.load_background(np.array([bg, bg]), names=['row'])

    # Fit with assumed neural network classification
    fit1, fit2 = m.fit(row=[0, 1], classifications=np.array([0, 0]))
    fit3 = m.fit_spectrum(arr, classifications=0, background=bg)  # Make sure this is equiv
    truth = [-215.08275199, 1001.01035476, 0.00477063, 0.28869302]

    assert np.array_equal(fit1.parameters, fit2.parameters)
    assert np.array_equal(fit1.parameters, fit3.parameters)
    assert list(fit1.parameters) == pytest.approx(truth)


def test_ibis8542model_configfile():
    # Enter the data directory (needed to find the files referenced in the config files)
    original_dir = os.getcwd()
    os.chdir(data_path())

    # Test with config file
    IBIS8542Model(config="ibis8542model_config.yml")

    # Turn off sigma
    IBIS8542Model(config="ibis8542model_config.yml", sigma=False)

    # Test with defined prefilter
    with pytest.deprecated_call():
        IBIS8542Model(config="ibis8542model_config_prefilter.yml")

    # Test with no prefilter
    IBIS8542Model(config="ibis8542model_config_noprefilter.yml")

    # TODO Check that the parameters were imported correctly

    # Go back to original directory
    os.chdir(original_dir)


@pytest.fixture
def valid_kwargs():
    stationary_line_core = 8542.099145376844
    defaults = {
        'stationary_line_core': stationary_line_core,
        'original_wavelengths': np.linspace(stationary_line_core - 2, stationary_line_core + 2, num=25),
        'constant_wavelengths': np.linspace(stationary_line_core - 1.7, stationary_line_core + 1.7, num=30),
        'prefilter_response': np.loadtxt(data_path('ibis8542model_prefilter.csv'), delimiter=','),
    }
    return defaults


@pytest.mark.parametrize('model', (ModelBase, IBIS8542Model))
def test_validate_parameters(model, valid_kwargs):

    # Default parameters should work
    model(**valid_kwargs)

    # original_wavelengths or constant_wavelengths not sorted
    for wavelengths in ('original_wavelengths', 'constant_wavelengths'):
        with pytest.raises(ValueError) as e:
            defaults_mod = valid_kwargs.copy()
            unsorted = defaults_mod[wavelengths].copy()  # Copy the numpy array before shuffling
            unsorted[10] = unsorted[12]  # Copy the 12th element to the 10th (now out of order)
            defaults_mod[wavelengths] = unsorted  # Replace the sorted array with the unsorted
            model(**defaults_mod)  # Initialise the model
        assert wavelengths in str(e.value) and 'must be sorted ascending' in str(e.value)

    # assert warning when constant_wavelengths extrapolates original_wavelengths (1e-5 over below and above)
    delta_lambda = 0.05
    for match, i, delta in [
        ("Upper bound of `constant_wavelengths` is outside of `original_wavelengths` range.", -1, delta_lambda + 1e-5),
        ("Lower bound of `constant_wavelengths` is outside of `original_wavelengths` range.", 0, -1e-5)
    ]:
        with pytest.warns(Warning, match=match):
            defaults_mod = valid_kwargs.copy()
            extrapolate = defaults_mod['constant_wavelengths'].copy()
            extrapolate[i] = defaults_mod['original_wavelengths'][i] + delta
            defaults_mod['constant_wavelengths'] = extrapolate
            model(**defaults_mod, delta_lambda=delta_lambda)  # Initialise the model

    # stationary_line_core is not a float
    with pytest.raises(ValueError) as e:
        defaults_mod = valid_kwargs.copy()
        defaults_mod.update({'stationary_line_core': int(8542)})
        model(**defaults_mod)
    assert 'stationary_line_core' in str(e.value) and 'float' in str(e.value)

    # stationary_line_core out of wavelength range
    with pytest.raises(ValueError) as e:
        defaults_mod = valid_kwargs.copy()
        defaults_mod.update({'stationary_line_core': float(1000)})
        model(**defaults_mod)
    assert 'stationary_line_core' in str(e.value) and 'is not within' in str(e.value)

    # Check that length of prefilter response is enforced
    with pytest.raises(ValueError) as e:
        defaults_mod = valid_kwargs.copy()
        defaults_mod.update({'prefilter_response': valid_kwargs['prefilter_response'][:-1]})
        m = model(**defaults_mod)
        m._set_prefilter()
    assert 'prefilter_response' in str(e.value) and 'must be the same length' in str(e.value)

    # Check that unexpected kwargs raise an error
    with pytest.raises(TypeError) as e:
        model(**valid_kwargs, qheysnfebsy=None)
    assert 'got an unexpected keyword argument' in str(e.value) and 'qheysnfebsy' in str(e.value)

    # TODO Verify remaining conditions


@pytest.mark.parametrize('model', (ModelBase, IBIS8542Model))
def test_load_data(model, valid_kwargs):

    # Initialise model
    m = model(**valid_kwargs)

    # Unknown target
    array = np.random.rand(5, 10, 15, 30)
    with pytest.raises(ValueError) as e:
        m._load_data(array, names=['time', 'row', 'column', 'wavelength'], target='arrrrray')
    assert 'array target must be' in str(e.value) and 'arrrrray' in str(e.value)

    # Ensure a single spectrum array cannot be loaded (not supported)
    for target, names, array in [('array', ['wavelength'], np.random.rand(30)), ('background', [], 723.23)]:
        with pytest.raises(ValueError) as e:
            m._load_data(array, names=names, target=target)
        assert 'cannot load an array containing one spectrum' in str(e.value)

    # Ensure dimension names are validated
    for meth in (m.load_array, m.load_background):

        array = np.random.rand(5, 10, 15, 30)
        with pytest.raises(ValueError) as e:
            meth(array)
        assert 'dimension names must be specified' in str(e.value)

        array = np.random.rand(5, 10, 15, 30)
        with pytest.raises(ValueError) as e:
            meth(array, names=['row', 'column', 'wavelength'])
        assert 'number of dimension names do not match number of columns' in str(e.value)

        array = np.random.rand(5, 10, 15, 30)
        with pytest.raises(ValueError) as e:
            meth(array, names=['wavelength', 'row', 'column', 'wavelength'])
        assert 'duplicate dimension names found' in str(e.value)

    array = np.random.rand(5, 10, 15)
    with pytest.raises(ValueError) as e:
        m.load_array(array, names=['time', 'row', 'column'])
    assert 'array must contain a wavelength dimension' in str(e.value)

    # 'wavelengths' not valid background dimension
    array = np.random.rand(5, 10, 15)
    with pytest.raises(ValueError) as e:
        m.load_background(array, names=['row', 'column', 'wavelength'])
    assert "name 'wavelength' is not a valid dimension name" in str(e.value)

    # 'rows' should be 'row' etc.
    array = np.random.rand(5, 10, 15, 30)
    with pytest.raises(ValueError) as e:
        m.load_array(array, names=['time', 'rows', 'column', 'wavelength'])
    assert "name 'rows' is not a valid dimension name" in str(e.value)
    array = np.random.rand(5, 10, 15)
    with pytest.raises(ValueError) as e:
        m.load_background(array, names=['time', 'rows', 'column'])
    assert "name 'rows' is not a valid dimension name" in str(e.value)

    array = np.random.rand(5, 10, 15, 30)
    with pytest.raises(ValueError) as e:
        m.load_array(array, names=['time', 'row', 'column', 'wavelength'])
    assert 'length of wavelength dimension not equal length of original_wavelengths' in str(e.value)

    array = np.random.rand(5, 10, 15)
    m.load_background(array, names=['time', 'row', 'column'])
    array = np.random.rand(5, 7, 15, 25)
    with pytest.warns(UserWarning, match="shape of spectrum array indices does not match shape of background array"):
        m.load_array(array, names=['time', 'row', 'column', 'wavelength'])


@pytest.mark.parametrize('model', (ModelBase, IBIS8542Model))
def test_get_time_row_column(model, valid_kwargs):

    # Initialise model
    m = model(**valid_kwargs)
    array = np.random.rand(4, 5, 6, 25)
    names = ['time', 'row', 'column', 'wavelength']
    m.load_array(array.copy(), names=names)

    # (test get_spectra with multiple spectra -- only supports a single spectrum)
    with pytest.raises(ValueError) as e:
        m.get_spectra(spectrum=np.random.rand(10, 30))
    assert 'explicit spectrum must have one dimension' in str(e.value)

    # All dimensions loaded so make sure all dimensions required
    for dim in names[:-1]:
        with pytest.raises(ValueError) as e:
            drop_a_name = names[:-1].copy()
            drop_a_name.remove(dim)
            m._get_time_row_column(**{k: 2 for k in drop_a_name})
        assert f"{dim} index must be specified as multiple indices exist" == str(e.value)

    # Drop a dimension and don't specify it
    array = array[0]  # Drop one (arbitrary) dimension
    for dim in names[:-1]:
        drop_a_name = names.copy()
        drop_a_name.remove(dim)
        m.load_array(array.copy(), names=drop_a_name)
        m._get_time_row_column(**{k: 2 for k in drop_a_name[:-1]})


def test_modelbase_fit(valid_kwargs):
    m = ModelBase(**valid_kwargs)
    array = np.random.rand(4, 5, 6, 25)
    array[0] = np.nan  # For testing no spectra given
    names = ['time', 'row', 'column', 'wavelength']
    m.load_array(array.copy(), names=names)

    # `_fit` method must be implemented in a subclass
    with pytest.raises(NotImplementedError):
        m._fit(np.random.rand(30))

    # Must raise exception if no spectra presented for fitting
    with pytest.raises(ValueError) as e:
        m.fit(time=0, row=range(5), column=range(6))
    assert 'no valid spectra given' in str(e.value)


def test_ibis8542model_validate_parameters(valid_kwargs):

    stationary_line_core = valid_kwargs['stationary_line_core']
    defaults = {
        'absorption_guess': [-1000, stationary_line_core, 0.2, 0.1],
        'emission_guess': [1000, stationary_line_core, 0.2, 0.1],
        'absorption_min_bound': [-np.inf, stationary_line_core - 0.15, 1e-6, 1e-6],
        'emission_min_bound': [0, -np.inf, 1e-6, 1e-6],
        'absorption_max_bound': [0, stationary_line_core + 0.15, 1, 1],
        'emission_max_bound': [np.inf, np.inf, 1, 1],
        'absorption_x_scale': [1500, 0.2, 0.3, 0.5],
        'emission_x_scale': [1500, 0.2, 0.3, 0.5]
    }

    # Incorrect Lengths
    for key, value in defaults.items():  # For each parameter
        with pytest.raises(ValueError) as e:  # Catch ValueErrors
            defaults_mod = defaults.copy()  # Create a copy of the default parameters
            defaults_mod.update({key: value[:-1]})  # Crop the parameter's value
            IBIS8542Model(**valid_kwargs, **defaults_mod)  # Pass the cropped parameter with the other default params
        assert key in str(e.value) and 'length' in str(e.value)  # Error must be about length of current parameter

    # Check that the sign of the following amplitudes are enforced
    for sign, bad_number, parameters in [('positive', -10.42, ('emission_guess', 'emission_min_bound')),
                                         ('negative', +10.42, ('absorption_guess', 'absorption_max_bound'))]:
        for p in parameters:
            with pytest.raises(ValueError) as e:
                bad_value = defaults[p].copy()
                bad_value[0] = bad_number
                defaults_mod = defaults.copy()
                defaults_mod.update({p: bad_value})
                IBIS8542Model(**valid_kwargs, **defaults_mod)
            assert p in str(e.value) and sign in str(e.value)

    # TODO Verify remaining conditions


def test_ibis8542model_get_sigma():
    # Enter the data directory (needed to find the files referenced in the config files)
    original_dir = os.getcwd()
    os.chdir(data_path())

    m = IBIS8542Model(config="ibis8542model_config.yml")
    sigma = np.loadtxt("ibis8542model_sigma.csv", delimiter=',')

    assert np.array_equal(m._get_sigma(classification=0), sigma[0])

    for c in (1, 2, 3, 4):
        assert np.array_equal(m._get_sigma(classification=c), sigma[1])

    for i in (0, 1):
        assert np.array_equal(m._get_sigma(sigma=i), sigma[i])

    x = np.array([1.4325, 1421.43, -1325.342, 153.3, 1.2, 433.0])
    assert np.array_equal(m._get_sigma(sigma=x), x)

    # Go back to original directory
    os.chdir(original_dir)


@pytest.fixture(scope='module')
def ibis8542model_init():
    # Enter the data directory (needed to find the files referenced in the config files)
    original_dir = os.getcwd()
    os.chdir(data_path())

    x_orig = np.loadtxt("ibis8542model_wavelengths_original.csv")

    m = IBIS8542Model(config="ibis8542model_init_config.yml", constant_wavelengths=x_orig, original_wavelengths=x_orig,
                      prefilter_response=np.ones(25))

    m.neural_network = DummyClassifier(trained=True, n_features=25)

    # Go back to original directory
    os.chdir(original_dir)

    return m


def test_ibis8542model_classify_spectra(ibis8542model_init):

    np.random.seed(0)
    spectra = np.random.rand(30, 25) * 100 + 1000
    truth = np.array([10, 37, 72, 0, 27, 100, 81, 44, 17, 49, 22, 97, 28, 100, 22, 17,
                      64, 91, 94, 47, 0, 34, 96, 98, 97, 3, 82, 87, 40, 51], dtype=int)

    m = ibis8542model_init

    classifications = m.classify_spectra(spectra=spectra, only_normalise=True)
    assert np.array_equal(classifications, truth)

    for i in range(len(truth)):
        classifications = m.classify_spectra(spectra=spectra[i], only_normalise=False)
        assert classifications[0] == truth[i] and len(classifications) == 1

    m.neural_network.trained = False
    with pytest.raises(NotFittedError):
        m.classify_spectra(spectra=spectra, only_normalise=True)


@pytest.fixture(scope='module')
def ibis8542model_spectra(ibis8542model_init):
    """IBIS8542Model with random data loaded"""

    m = ibis8542model_init

    spectra = np.empty((2, 3, 4, len(m.original_wavelengths)), dtype=np.float64)
    classifications = np.empty((2, 3, 4), dtype=int)

    np.random.seed(253)
    a1_array = np.random.rand(*spectra.shape[:-1]) * 500 - 800
    a2_array = np.random.rand(*spectra.shape[:-1]) * 500 + 300
    b1_array = np.random.rand(*spectra.shape[:-1]) / 2 - 0.25 + m.stationary_line_core
    b2_array = np.random.rand(*spectra.shape[:-1]) / 2 - 0.25 + m.stationary_line_core
    s1_array = np.random.rand(*spectra.shape[:-1]) / 2 + 0.1
    s2_array = np.random.rand(*spectra.shape[:-1]) / 2 + 0.1
    g1_array = np.random.rand(*spectra.shape[:-1]) / 2 + 0.1
    g2_array = np.random.rand(*spectra.shape[:-1]) / 2 + 0.1
    d_array = np.random.rand(*spectra.shape[:-1]) * 600 + 700

    a1_array[0, 1] = np.nan
    a2_array[1, :2] = np.nan

    for i in range(len(spectra)):
        for j in range(len(spectra[0])):
            for k in range(len(spectra[0, 0])):
                if np.isnan(a1_array[i, j, k]):
                    classifications[i, j, k] = 4
                    spectra[i, j, k] = voigt(m.original_wavelengths, a2_array[i, j, k], b2_array[i, j, k],
                                             s2_array[i, j, k], g2_array[i, j, k], d_array[i, j, k])
                elif np.isnan(a2_array[i, j, k]):
                    classifications[i, j, k] = 0
                    spectra[i, j, k] = voigt(m.original_wavelengths, a1_array[i, j, k], b1_array[i, j, k],
                                             s1_array[i, j, k], g1_array[i, j, k], d_array[i, j, k])
                else:
                    classifications[i, j, k] = 1
                    spectra[i, j, k] = double_voigt(m.original_wavelengths, a1_array[i, j, k], a2_array[i, j, k],
                                                    b1_array[i, j, k], b2_array[i, j, k],
                                                    s1_array[i, j, k], s2_array[i, j, k],
                                                    g1_array[i, j, k], g2_array[i, j, k], d_array[i, j, k])

    m.load_array(spectra, names=['time', 'row', 'column', 'wavelength'])
    m.load_background(d_array, names=['time', 'row', 'column'])

    return m, classifications


@pytest.fixture(scope='module')
def ibis8542model_results(ibis8542model_spectra):

    # Load model with random spectra loaded
    m, classifications = ibis8542model_spectra

    # Test with explicit classifications
    result = m.fit(time=range(2), row=range(3), column=range(4), classifications=classifications)

    return result, m, classifications


def assert_results_equal(res1, res2):
    for i in range(len(res1)):
        assert res1[i].parameters == pytest.approx(res2[i].parameters, nan_ok=True)
        assert res1[i].classification == res2[i].classification
        assert res1[i].profile == res2[i].profile
        assert res1[i].success == res2[i].success
        assert np.array_equal(res1[i].index, res2[i].index)


def test_ibis8542model_wrong_length_of_classifications(ibis8542model_spectra):

    # Load model with random spectra loaded
    m, classifications = ibis8542model_spectra

    assert classifications.shape == (2, 3, 4)  # This is what the test needs to work (verifies fixture)

    # Test with too few classifications
    c1 = classifications[:, :, [0, 1]]
    assert c1.shape == (2, 3, 2)
    c2 = classifications[:, [0, 1]]
    assert c2.shape == (2, 2, 4)
    c3 = classifications[[1]]
    assert c3.shape == (1, 3, 4)
    c_wrong_shape = np.transpose(classifications)  # ...but correct size so would not fail otherwise
    assert c_wrong_shape.shape == (4, 3, 2)
    for c in [c1, c2, c3, c_wrong_shape]:
        with pytest.raises(ValueError) as e:
            m.fit(time=range(2), row=range(3), column=range(4), classifications=c)
        assert 'classifications do not match' in str(e.value)

    # Test with too many classifications
    for t, r, c in [
        # (range(2), range(3), range(4)),  # Correct values
        (1, range(3), range(4)),
        (range(2), range(1, 3), range(4)),
        (range(2), range(3), range(2, 4)),
        (range(2), range(3), 3),
        (0, 1, 2),
    ]:
        with pytest.raises(ValueError) as e:
            m.fit(time=t, row=r, column=c, classifications=classifications)
        assert 'classifications do not match' in str(e.value)

    # Test with dimensions of length 1 removed (res1 and res2 should be equivalent)

    c = classifications[:, np.newaxis, 0]
    assert c.shape == (2, 1, 4)
    res1 = m.fit(time=range(2), row=range(1), column=range(4), classifications=c)

    c = classifications[:, 0]
    assert c.shape == (2, 4)
    res2 = m.fit(time=range(2), row=range(1), column=range(4), classifications=c)

    assert len(res1) == len(res2) == 2 * 1 * 4
    assert_results_equal(res1, res2)


@pytest.fixture()
def ibis8542model_resultsobjs(ibis8542model_results):

    def fits2array(results):

        results0 = FitResults((3, 4), 8, time=0)
        results1 = FitResults((3, 4), 8, time=1)

        for i in range(len(results)):

            if results[i].index[0] == 0:
                results0.append(results[i])
            elif results[i].index[0] == 1:
                results1.append(results[i])
            else:
                raise ValueError("invalid time")

        return results0, results1

    return fits2array


def test_ibis8542model_fit(ibis8542model_results, ibis8542model_resultsobjs):

    # # METHOD 1: Test with explicit classifications
    res1, m, classifications = ibis8542model_results

    # # METHOD 2: Test with dummy classifier
    m.neural_network = DummyClassifier(trained=True, classifications=classifications)
    res2 = m.fit(time=range(2), row=range(3), column=range(4))

    assert len(res1) == len(res2) == 2*3*4
    assert_results_equal(res1, res2)

    # # METHOD 3: Test over 4 processing pools
    res3 = m.fit(time=range(2), row=range(3), column=range(4), n_pools=4)
    # (n_pools must be an integer)
    with pytest.raises(TypeError) as e:
        m.fit(time=range(2), row=range(3), column=range(4), n_pools=float(4))
    assert 'n_pools must be an integer' in str(e.value)

    # # Test that FitResults objects can be created consistently for all of the methods

    # Create the FitResults objects from the list of FitResult objects
    results10, results11 = ibis8542model_resultsobjs(res1)  # time : 0, 1
    results20, results21 = ibis8542model_resultsobjs(res2)
    results30, results31 = ibis8542model_resultsobjs(res3)

    # Compare METHOD 1, METHOD 2, METHOD 3
    for t0, t1 in [(results20, results21), (results30, results31)]:  # For each alternative method (tn)
        for results, tn in [(results10, t0), (results11, t1)]:  # Compare to main method (results)
            assert results.parameters == pytest.approx(tn.parameters, nan_ok=True)
            assert np.array_equal(results.profile, tn.profile)
            assert np.array_equal(results.classifications, tn.classifications)
            assert np.array_equal(results.success, tn.success)

    # # Test current values against expected values

    truth10, truth11 = [fits.open(data_path(f)) for f in
                        ("ibis8542model_fit_results10.fits", "ibis8542model_fit_results11.fits")]

    for results, truth in [(results10, truth10), (results11, truth11)]:

        assert results.parameters == pytest.approx(truth['PARAMETERS'].data, nan_ok=True)

        # Compress profile array to integers
        p_uniq = np.unique(results.profile)
        p = np.full_like(results.profile, -1, dtype=int)
        for i in range(len(p_uniq)):
            p[results.profile == p_uniq[i]] = i

        assert np.array_equal(p, truth['PROFILE'].data)
        assert np.array_equal(results.classifications, truth['CLASSIFICATIONS'].data)
        assert np.array_equal(results.success, truth['SUCCESS'].data)


@pytest.mark.parametrize("i,j,k", [(0, 1, 3), (1, 0, 3)])
@figure_test
def test_ibis8542model_plot_indices(pytestconfig, i, j, k, ibis8542model_spectra):
    m = ibis8542model_spectra[0]
    ax = plt.gca()
    m.plot(time=i, row=j, column=k, ax=ax)


@pytest.mark.parametrize("i", np.hstack([range(4, 8), range(18, 20)]))
@figure_test
def test_ibis8542model_plot(pytestconfig, i, ibis8542model_results):
    res, m, _ = ibis8542model_results
    m.plot(fit=res[i])


@pytest.mark.parametrize("i", np.hstack([range(4, 8), range(18, 20)]))
@figure_test
def test_ibis8542model_plot_separate(pytestconfig, i, ibis8542model_results):
    res, m, _ = ibis8542model_results
    m.plot_separate(fit=res[i])


@pytest.mark.parametrize("i", np.hstack([range(4, 8), range(18, 20)]))
@figure_test
def test_ibis8542model_plot_subtraction(pytestconfig, i, ibis8542model_results):
    res, m, _ = ibis8542model_results
    m.plot_subtraction(fit=res[i])


@pytest.mark.parametrize("i", np.hstack([range(4, 8), range(18, 20)]))
@figure_test
def test_ibis8542model_fitresult_plot(pytestconfig, i, ibis8542model_results):
    res, m, _ = ibis8542model_results
    res[i].plot(m)


@figure_test
def test_ibis8542model_plot_spectrum(pytestconfig, ibis8542model_spectra):
    m = ibis8542model_spectra[0]
    spectrum = np.linspace(-1, 1, len(m.constant_wavelengths))
    spectrum = 11 * np.exp(-spectrum ** 2 / 0.05)

    # Raises invalid spectrum dimensions
    with pytest.raises(ValueError) as e:
        m.plot(spectrum=np.array([spectrum, spectrum]))
    assert 'spectrum must have one dimension' in str(e.value)

    # Raises invalid index
    with pytest.raises(IndexError):
        m.plot(time=0, row=10.25, column=0)

    m.plot(spectrum=spectrum, fit=[5, 8542, 0.1, 0.1])


@figure_test
def test_ibis8542model_plot_no_intensity(pytestconfig, ibis8542model_results):
    res, m, _ = ibis8542model_results
    m.plot(fit=res[4], show_intensity=False, show_legend=False)


def test_ibis8542model_save(ibis8542model_results, ibis8542model_resultsobjs, tmp_path):

    # Testing mcalf.models.FitResults.save method

    res1, m = ibis8542model_results[:2]
    results10, results11 = ibis8542model_resultsobjs(res1)

    tmp10 = os.path.join(tmp_path, "ibis8542model_fit_save_results10.fits")
    results10.save(tmp10, m)
    tmp11 = os.path.join(tmp_path, "ibis8542model_fit_save_results11.fits")
    results11.save(tmp11, m)

    diff_kwargs = {
        'ignore_keywords': ['CHECKSUM', 'DATASUM'],
        'atol': 1e-6,
        'rtol': 1e-5,  # 1e-6 was failing at results10 CHI2[2, 2] on macOS CI env (but nowhere else)
    }
    for saved, truth in [(tmp10, "ibis8542model_fit_results10.fits"),
                         (tmp11, "ibis8542model_fit_results11.fits")]:
        saved = fits.open(saved)
        truth = fits.open(data_path(truth))
        diff = fits.FITSDiff(saved, truth, **diff_kwargs)
        if not diff.identical:  # If this fails tolerances *may* need to be adjusted
            fits.printdiff(saved, truth, **diff_kwargs)
            raise ValueError(f"{saved.filename()} and {truth.filename()} differ")


def test_random_state():

    # Testing that the `random_state` kwarg works as expected on the system

    # Arbitrary wavelength wavelength points
    wavelengths = np.linspace(8541, 8544, 49)

    # Initialise model
    model = IBIS8542Model(original_wavelengths=wavelengths, random_state=0)

    # Get sample classifications
    X, y = make_classification(200, 49, n_classes=5, n_informative=4, random_state=0)

    # Training #1
    model.train(X[::2], y[::2])
    score_a = cross_val_score(model.neural_network, X[1::2], y[1::2])

    # Training #2
    model.train(X[::2], y[::2])
    score_b = cross_val_score(model.neural_network, X[1::2], y[1::2])

    assert score_b == pytest.approx(score_a)
    assert score_b == pytest.approx(np.array([0.45, 0.35, 0.45, 0.45, 0.35]))
