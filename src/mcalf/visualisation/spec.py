import numpy as np
from matplotlib import pyplot as plt

from mcalf.profiles.voigt import double_voigt, voigt
from mcalf.utils.spec import reinterpolate_spectrum


__all__ = ['plot_ibis8542', 'plot_spectrum']


def plot_ibis8542(wavelengths, spectrum, fit=None, background=0, sigma=None, sigma_scale=70, stationary_line_core=None,
                  subtraction=False, separate=False, output=None, figsize=None, legend_position='best', dpi=None,
                  fontfamily=None, reduced_legend=False, show_intensity=True, hook=None):
    """Plot an IBIS8542Model fit

    It is recommended to use the plot method on either an IBIS8542Model or a FitResult from an IBIS8542Model instead.

    Parameters
    ----------
    wavelengths : ndarray
        The x-axis values.
    spectrum : ndarray, length=n_wavelengths
        The y-axis values.
    fit : array_like, optional, default = None
        The fitted parameters.
    background : float or ndarray of length n_wavelengths, optional, default = 0
        The background to add to the fitted profiles.
    sigma : ndarray, length=n_wavelengths, optional, default = None
        The sigma profile used when fitting the parameters to `spectrum`. If given, will be plotted as shaded regions.
    sigma_scale : float, optional, default = 70
        A factor to multiply the error bars to change their prominence.
    stationary_line_core : float, optional, default = None
        If given, will show a dashed line at this wavelength.
    subtraction : bool, optional, default = False
        Whether to plot the `spectrum` minus emission fit (if exists) instead.
    separate : bool, optional, default = False
        Whether to plot the fitted profiles separately (if multiple components exist).
    output : str, optional, default = None
        If present, the filename to save the plot as.
    figsize : 2-tuple, optional
        Size of the figure.
    legend_position : str or int or pair of floats, optional, default = 'best'
        Position of the legend. See `matplotlib.pyplot.legend` documentation from possible values.
    dpi : int
        The number of dots per inch. For controlling the quality of the outputted figure.
    fontfamily : str, optional, default = None
        If provided, this family string will be added to the 'font' rc params group.
    reduced_legend : bool, optional, default = False
        Whether to add to the legend the labels that would be displayed on an absorption only plot. Useful for saving
        space when plotting both a single component fit and a multi-component fit alongside each other.
    show_intensity : bool, optional, default = True
        Whether to show the intensity axis tick labels and axis label.
    hook : callable, optional, default = None
        If provided this function must accept the current `plt' as a single argument such that it can operate upon
        it and make changes to the plot.

    See Also
    --------
    models.IBIS8542.plot : General plotting method
    models.IBIS8542.plot_separate : Plot the fit parameters separately
    models.IBIS8542.plot_subtraction : Plot the spectrum with the emission fit subtracted from it
    models.FitResult.plot : Plotting method on the fit result
    """
    # Choose the function to plot the fitted parameters with
    if fit is not None and len(fit) == 8:
        fit_function = double_voigt
    else:
        fit_function = voigt

    data_label = 'observation'  # Default label

    if fontfamily is not None:
        plt.rc('font', family=fontfamily)
    plt.figure(figsize=figsize, dpi=dpi)

    # If a subtraction is requested, make the relevant changes
    if subtraction and fit is not None and len(fit) == 8:
        spectrum = spectrum - voigt(wavelengths, *fit[4:], 0, clib=False)
        sigma = None
        fit = None
        data_label = 'observation - emission profile'

    # If sigma is given, make a shaded region around the spectral data.
    if sigma is not None and fit is not None:
        plt.fill_between(wavelengths, spectrum-sigma*sigma_scale, spectrum+sigma*sigma_scale, color='lightgrey')

    # Plot the spectral data
    label = None if reduced_legend else data_label
    plt.plot(wavelengths, spectrum, color='#006BA4', label=label)

    # Plot the fitted profiles if parameters are given
    if fit is not None:

        if separate and len(fit) > 4:  # Plot each component separately

            label = None if reduced_legend else 'combined profile'
            plt.plot(wavelengths, double_voigt(wavelengths, *fit, background, clib=False),
                     color='#FF800E', label=label)
            plt.plot(wavelengths, voigt(wavelengths, *fit[:4], background, clib=False),
                     color='#A2C8EC', label='absorption profile')
            plt.plot(wavelengths, voigt(wavelengths, *fit[4:], 0, clib=False),
                     color='#595959', label='emission profile')

        else:  # Plot a combined profile

            plt.plot(wavelengths, fit_function(wavelengths, *fit, background, clib=False),
                     color='#FF800E', label='fitted profile')

        label = None if reduced_legend else 'absorption line core'
        plt.axvline(x=fit[1], linestyle='--', color='#A2C8EC', label=label)
        if fit_function is double_voigt:
            plt.axvline(x=fit[5], linestyle=':', color='#595959', label='emission line core')

    if stationary_line_core is not None:  # Plot vertical dashed line
        label = None if reduced_legend else 'stationary line core'
        plt.axvline(x=stationary_line_core, linestyle='--', color='#ABABAB', label=label)

    plt.xlabel('Wavelength (Å)')
    if not reduced_legend and show_intensity:
        plt.ylabel('Intensity')
    if not show_intensity:
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    plt.legend(loc=legend_position)
    plt.minorticks_on()

    if hook is not None:
        hook(plt)

    if output is not None:
        plt.savefig(output, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_spectrum(wavelengths, spectrum, output=None, normalised=True, smooth=True,
                  figsize=(7, 3), dpi=600, fontfamily=None):
    """Plot a spectrum with the wavelength grid shown.

    Intended for plotting the raw data.

    Parameters
    ----------
    wavelengths : ndarray
        The x-axis values.
    spectrum : ndarray, length=n_wavelengths
        The y-axis values.
    output : str, optional, default = None
        If present, the filename to save the plot as.
    normalised : bool, optional, default = True
        Whether to normalise the spectrum using the last three spectral points.
    smooth : bool, optional, default = True
        Whether to smooth the `spectrum` with a spline.
    figsize : 2-tuple, optional, default = None
        Size of the figure.
    dpi : int, optional, default = 600
        The number of dots per inch. For controlling the quality of the outputted figure.
    fontfamily : str, optional, default = None
        If provided, this family string will be added to the 'font' rc params group.
    """

    if fontfamily is not None:
        plt.rc('font', family=fontfamily)

    dense_wavelengths = np.linspace(wavelengths[0], wavelengths[-1], num=200) if smooth else wavelengths
    dense_spectrum = reinterpolate_spectrum(spectrum, wavelengths, dense_wavelengths) if smooth else spectrum
    if normalised:
        norm = np.mean(spectrum[-3:])
        dense_spectrum /= norm
        spectrum /= norm

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(len(wavelengths)):
        ax.axvline(x=wavelengths[i], linestyle='--', linewidth=0.5, c='black')
    ax.plot(dense_wavelengths, dense_spectrum, color='black')
    ax.plot(wavelengths, spectrum, marker='o', linestyle='None', color='black', markersize=4)

    ylabel = 'Normalised Intensity ($I/I_c$)' if normalised else 'Intensity ($I$)'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Wavelength (Å)')

    fig.subplots_adjust(bottom=0.15)

    plt.show()

    if output is not None and isinstance(output, str):
        fig.savefig(output, bbox_inches='tight', dpi=dpi)
