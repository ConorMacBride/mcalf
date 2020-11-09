import pytest
import os
import numpy as np
from astropy.io import fits
from sklearn.exceptions import NotFittedError

from mcalf.models.ibis import IBIS8542Model
from mcalf.models.results import FitResults
from mcalf.profiles.voigt import voigt, double_voigt


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


def test_ibis8542model_default():
    # Test default parameters
    with pytest.raises(ValueError) as e:
        m = IBIS8542Model()
    assert 'original_wavelengths' in str(e.value)


def test_ibis8542model_basic():
    # Will break if default parameters are changes in mcalf.models.ibis.IBIS8542Model
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
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

    # Test with config file
    m1 = IBIS8542Model(config="ibis8542model_config.yml")

    # Turn off sigma
    m2 = IBIS8542Model(config="ibis8542model_config.yml", sigma=False)

    # Test with defined prefilter
    m3 = IBIS8542Model(config="ibis8542model_config_prefilter.yml")

    # Test with no prefilter
    with pytest.warns(UserWarning, match='prefilter_response'):
        m4 = IBIS8542Model(config="ibis8542model_config_noprefilter.yml")

    # TODO Check that the parameters were imported correctly


def test_ibis8542model_validate_parameters():

    stationary_line_core = 8542.099145376844
    x_orig = np.linspace(stationary_line_core - 2, stationary_line_core + 2, num=25)
    x_const = np.linspace(stationary_line_core - 1.7, stationary_line_core + 1.7, num=30)
    prefilter = np.loadtxt(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ibis8542model_prefilter.csv"),
        delimiter=','
    )
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

    def IBIS8542Model_default(stationary_line_core=stationary_line_core,
                              original_wavelengths=x_orig, constant_wavelengths=x_const,
                              prefilter=prefilter, **kwargs):
        local_defaults = defaults.copy()
        local_defaults.update(kwargs)
        IBIS8542Model(stationary_line_core=stationary_line_core,
                      original_wavelengths=original_wavelengths, constant_wavelengths=constant_wavelengths,
                      prefilter_response=prefilter, **local_defaults)

    # Default parameters should work
    IBIS8542Model_default()

    # stationary_line_core is not a float
    with pytest.raises(ValueError) as e:
        IBIS8542Model_default(stationary_line_core=int(1000))
    assert 'stationary_line_core' in str(e.value) and 'float' in str(e.value)

    # Incorrect Lengths
    for key, value in defaults.items():  # For each parameter
        with pytest.raises(ValueError) as e:  # Catch ValueErrors
            defaults_mod = defaults.copy()  # Create a copy of the default parameters
            defaults_mod.update({key: value[:-1]})  # Crop the parameter's value
            IBIS8542Model_default(**defaults_mod)  # Pass the cropped parameter with the other default parameters
        assert key in str(e.value) and 'length' in str(e.value)  # Error must be about the length of the current parameter

    # Check that the sign of the following amplitudes are enforced
    for sign, bad_number, parameters in [('positive', -10.42, ('emission_guess', 'emission_min_bound')),
                                         ('negative', +10.42, ('absorption_guess', 'absorption_max_bound'))]:
        for p in parameters:
            with pytest.raises(ValueError) as e:
                bad_value = defaults[p].copy()
                bad_value[0] = bad_number
                IBIS8542Model_default(**{p: bad_value})
            assert p in str(e.value) and sign in str(e.value)

    # TODO Verify remaining conditions


def test_ibis8542model_get_sigma():
    # Enter the data directory (needed to find the files referenced in the config files)
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

    m = IBIS8542Model(config="ibis8542model_config.yml")
    sigma = np.loadtxt("ibis8542model_sigma.csv", delimiter=',')

    assert np.array_equal(m._get_sigma(classification=0), sigma[0])

    for c in (1, 2, 3, 4):
        assert np.array_equal(m._get_sigma(classification=c), sigma[1])

    for i in (0, 1):
        assert np.array_equal(m._get_sigma(sigma=i), sigma[i])

    x = np.array([1.4325, 1421.43, -1325.342, 153.3, 1.2, 433.0])
    assert np.array_equal(m._get_sigma(sigma=x), x)


@pytest.fixture()
def ibis8542model_init():
    # Enter the data directory (needed to find the files referenced in the config files)
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

    x_orig = np.loadtxt("ibis8542model_wavelengths_original.csv")

    m = IBIS8542Model(config="ibis8542model_init_config.yml", constant_wavelengths=x_orig, original_wavelengths=x_orig,
                      prefilter_response=np.ones(25))

    m.neural_network = DummyClassifier(trained=True, n_features=25)

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


@pytest.fixture()
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


@pytest.fixture()
def ibis8542model_results(ibis8542model_spectra):

    # Load model with random spectra loaded
    m, classifications = ibis8542model_spectra

    # Test with explicit classifications
    result = m.fit(time=range(2), row=range(3), column=range(4), classifications=classifications)

    return result, m, classifications


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

    for i in range(len(res1)):
        assert np.array_equal(res1[i].parameters, res2[i].parameters)
        assert res1[i].classification == res2[i].classification
        assert res1[i].profile == res2[i].profile
        assert res1[i].success == res2[i].success
        assert np.array_equal(res1[i].index, res2[i].index)

    # # METHOD 3: Test over 4 processing pools
    res3 = m.fit(time=range(2), row=range(3), column=range(4), n_pools=4)

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

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    truth10, truth11 = [fits.open(os.path.join(path, f)) for f in
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


def test_ibis8542model_plot(ibis8542model_results):

    # Test that plots can be produced without exceptions

    res1, m, classifications = ibis8542model_results

    def test_hook(plt):
        # Test the hook and also try to stop plots showing fully
        plt.close()

    for i, j, k in [(0, 1, 3), (0, 0, 2), (1, 0, 3), (1, 2, 2)]:
        m.plot(time=i, row=j, column=k, hook=test_hook)

    for fit in res1[::3]:
        m.plot(fit=fit, hook=test_hook)
        m.plot_separate(fit=fit, hook=test_hook)
        m.plot_subtraction(fit=fit, hook=test_hook)
        fit.plot(m, hook=test_hook)

    # TODO Use a lookup table of hashes for each MPL version


def test_ibis8542model_save(ibis8542model_results, ibis8542model_resultsobjs, tmp_path):

    # Testing mcalf.models.results.FitResults.save method

    res1, m = ibis8542model_results[:2]
    results10, results11 = ibis8542model_resultsobjs(res1)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    tmp10 = os.path.join(tmp_path, "ibis8542model_fit_save_results10.fits")
    results10.save(tmp10, m)
    tmp11 = os.path.join(tmp_path, "ibis8542model_fit_save_results11.fits")
    results11.save(tmp11, m)

    for saved, truth in [(tmp10, "ibis8542model_fit_results10.fits"),
                         (tmp11, "ibis8542model_fit_results11.fits")]:
        saved = fits.open(saved)
        truth = fits.open(os.path.join(path, truth))
        for key in ('PARAMETERS', 'CLASSIFICATIONS', 'PROFILE', 'SUCCESS', 'CHI2', 'VLOSA', 'VLOSQ'):
            # TODO Work out why the default rel=1e-6 was failing (linux vs. macos) for one particular CHI2 value
            try:
                assert saved[key].data == pytest.approx(truth[key].data, nan_ok=True, rel=1e-5)
            except AssertionError:
                if os.name == 'nt':
                    pytest.xfail("known issue running this test on windows")
                else:
                    raise
