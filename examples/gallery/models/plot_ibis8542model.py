"""
Using IBIS8542Model
===================
This is an example showing how to fit an array of
spectra using the :class:`mcalf.models.IBIS8542Model`
class, and plot and export the results.
"""

#%%
# First, we shall generate some random sample data to
# demonstrate the API.
# The randomly generated spectra are not intended to
# be representative of observed spectra numerically,
# they just have a similar shape.

import numpy as np
import matplotlib.pyplot as plt

from mcalf.profiles.voigt import voigt, double_voigt
from mcalf.visualisation import plot_spectrum

# Create demo wavelength grid
w = np.linspace(8541, 8543, 20)

#%%
# We shall generate demo spectra in a 2x3 grid.
# Half of the spectra will have one spectral
# component (absorption only) and the other
# half will have two spectral components
# (mixed absorption and emission components).
#
# Demo spectral components will be modelled as
# Voigt functions with randomly generated
# parameters, each within a set range of values.


def v(classification, w, *args):
    """Voigt function wrapper."""
    if classification < 1.5:  # absorption only
        return voigt(w, *args[:4], args[-1])
    return double_voigt(w, *args)  # absorption + emission


def s():
    """Generate random spectra for a 2x3 grid."""
    np.random.seed(0)  # same spectra every time
    p = np.random.rand(9, 6)  # random Voigt parameters
    # 0  1  2  3  4  5  6  7  8  # p index
    # a1 b1 s1 g1 a2 b2 s2 g2 d  # Voigt parameter
    # absorption |emission   |background

    p[0] = 100 * p[0] - 1000  # absorption amplitude
    p[4] = 100 * p[4] + 1000  # emission amplitude

    for i in (1, 5):  # abs. and emi. peak positions
        p[i] = 0.05 * p[i] - 0.025 + 8542

    for i in (2, 3, 6, 7):  # Voigt sigmas and gammas
        p[i] = 0.1 * p[i] + 0.1

    p[8] = 300 * p[8] + 2000  # intensity background constant

    # Define each spectrum's classification
    c = [0, 2, 0, 2, 0, 2]
    # Choose single or double component spectrum
    # based on this inside the function `v()`.

    # Generate the spectra
    specs = [v(c[i], w, *p[:, i]) for i in range(6)]

    # Reshape to 2x3 grid
    return np.asarray(specs).reshape((2, 3, len(w)))


raw_data = s()

print('shape of spectral grid:', raw_data.shape)

#%%
# These spectra look as follows,

fig, axes = plt.subplots(2, 3, constrained_layout=True)
for ax, spec in zip(axes.flat, raw_data.reshape((6, raw_data.shape[-1]))):

    plot_spectrum(w, spec, normalised=False, ax=ax)

plt.show()

#%%
# `MCALF` does not model a constant background value,
# i.e., the fitting process assumes the intensity
# values far out in the spectrum's wings are zero.
#
# You can however tell `MCALF` what the constant
# background value is, and it will subtract it
# from the spectrum before fitting.
#
# This process was not made automatic as we wanted
# to give the user full control on setting the
# background value.
#
# For these demo data, we shall simply set the background
# to the first intensity value of every spectrum.
# For a real dataset, you may wish to average the
# background value throughout a range of spectral points,
# or even do a moving average throughout time.
# Functions are provided in ``mcalf.utils.smooth`` to
# assist with this process.

backgrounds = raw_data[:, :, 0]

print('shape of background intensity grid:', backgrounds.shape)

#%%
# In this example we will not demonstrate how to create
# a neural network classifier. `MCALF` offers a lot of
# flexibility when it comes to the classifier. This
# demo classifier outlines the basic API that is
# required. By default, the model has a
# :class:`sklearn.neural_network.MLPClassifier`
# object preloaded for use as the classifier.
# The methods :meth:`mcalf.models.ModelBase.train`
# and :meth:`mcalf.models.ModelBase.test` are
# provided by `MCALF` to assist with training the
# neural network. There is also a useful script
# in the `Getting Started` section under
# `User Documentation` in the sidebar for
# semi-automating the process of creating the
# ground truth dataset.
#
# Please see tutorials for packages such
# as scikit-learn for more in-depth advice on
# creating classifiers.
#
# As we only have six spectra with two distinct
# shapes, we can create a very simple classifier
# that classifies spectra based on whether their
# central intensity is greater or smaller than
# the left most intensity.


class DemoClassifier:
    @staticmethod
    def predict(X):
        y = np.zeros(len(X), dtype=int)
        y[X[:, len(X[0]) // 2] > X[:, 0]] = 2
        return y


#%%
# Everything we have been doing up to this point has been
# creating the demo data and classifier. Now we can actually
# create a model and load in the demo data, although the
# following steps would be identical for a real dataset.

from mcalf.models import IBIS8542Model

# Initialise the model with the wavelength grid
model = IBIS8542Model(original_wavelengths=w)

# Load the spectral shape classifier
model.neural_network = DemoClassifier

# Load the array of spectra and background intensities
model.load_array(raw_data, ['row', 'column', 'wavelength'])
model.load_background(backgrounds, ['row', 'column'])

#%%
# We have now fully initialised the model. We can now
# call methods to fit the model to the loaded spectra.
# In the following step we fit all the loaded spectra
# and a 1D list of :class:`~mcalf.models.FitResult`
# objects is returned.

fits = model.fit(row=[0, 1], column=range(3))

print(fits)

#%%
# The individual components of each fit can now be
# plotted separately,

fig, axes = plt.subplots(2, 3, constrained_layout=True)
for ax, fit in zip(axes.flat, fits):

    model.plot_separate(fit, show_legend=False, ax=ax)

plt.show()

#%%
# As well as fitting spectra, we can call other methods.
# In this step we'll extract the array of loaded spectra.
# However, these spectra have been re-interpolated to a
# new finer wavelength grid. (This grid can be customised
# when initialising the model.)

spectra = model.get_spectra(row=[0, 1], column=range(3))

print('new shape of spectral grid:', spectra.shape)

spectra_1d = spectra.reshape((6, spectra.shape[-1]))

fig, axes = plt.subplots(2, 3, constrained_layout=True)
for ax, spec in zip(axes.flat, spectra_1d):

    plot_spectrum(model.constant_wavelengths, spec,
                  normalised=False, ax=ax)

plt.show()

#%%
# We can also classify the loaded spectra and create plots,

classifications = model.classify_spectra(row=[0, 1], column=range(3))

print('classifications are:', classifications)

#%%
# This function plots the spectra grouped
# by classification,

from mcalf.visualisation import plot_classifications

plot_classifications(spectra_1d, classifications.flatten())

#%%
# This function plots a spatial map of
# the classifications on the 2x3 grid,

from mcalf.visualisation import plot_class_map

plot_class_map(classifications)

#%%
# The :class:`~mcalf.models.FitResult` objects in the
# ``fits`` 1D list can be merged into a grid.
# Each of these objects can be appended to a
# single :class:`~mcalf.models.FitResults` object.
# This allows for increased data portability,

from mcalf.models import FitResults

# Initialise with the grid shape and num. params.
results = FitResults((2, 3), 8)

# Append each fit to the object
for fit in fits:
    results.append(fit)

#%%
# Now that we have appended all 6 fits,
# we can access the merged data,

print(results.classifications)
print(results.profile)
print(results.success)
print(results.parameters)

#%%
# And finally, we can calculate Doppler
# velocities for both the quiescent (absorption)
# and active (emission) regimes.
# (The model needs to be given so the stationary
# line core wavelength is available.)

quiescent = results.velocities(model, vtype='quiescent')
active = results.velocities(model, vtype='active')

from mcalf.visualisation import plot_map

fig, axes = plt.subplots(2, constrained_layout=True)
plot_map(quiescent, ax=axes[0])
plot_map(active, ax=axes[1])
plt.show()

#%%
# The :class:`~mcalf.models.FitResults` object can
# also be exported to a FITS file,

results.save('ibis8542model_demo_fit.fits', model)

#%%
# The file has the following structure,

from astropy.io import fits

with fits.open('ibis8542model_demo_fit.fits') as hdul:
    hdul.info()

#%%
# This example outlines the basics of using the
# :class:`mcalf.models.IBIS8542Model` class.
# Typically millions of spectra would need to be
# fitted and processed at the same time.
# For more details on running big jobs and how
# spectra can be fitted in parallel, see
# `example1` in the `Getting Started` section
# of `User Documentation` in the sidebar.
