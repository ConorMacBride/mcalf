import numpy as np
import matplotlib.pyplot as plt

from mcalf.profiles.voigt import double_voigt, voigt
from mcalf.utils.spec import reinterpolate_spectrum
from mcalf.utils.plot import hide_existing_labels


__all__ = ['plot_ibis8542', 'plot_spectrum']


def plot_ibis8542(wavelengths, spectrum, fit=None, background=0,
                  sigma=None, sigma_scale=70,
                  stationary_line_core=None,
                  subtraction=False, separate=False,
                  show_intensity=True, show_legend=True, ax=None):
    """Plot an :class:`~mcalf.models.IBIS8542Model` fit.

    .. note::
        It is recommended to use the plot method built into either the
        :class:`~mcalf.models.IBIS8542Model` class or the :class:`~mcalf.models.FitResult`
        class instead.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        The x-axis values.
    spectrum : numpy.ndarray, length=n_wavelengths
        The y-axis values.
    fit : array_like, optional, default=None
        The fitted parameters.
    background : float or numpy.ndarray, length=n_wavelengths, optional, default=0
        The background to add to the fitted profiles.
    sigma : numpy.ndarray, length=n_wavelengths, optional, default=None
        The sigma profile used when fitting the parameters to `spectrum`.
        If given, will be plotted as shaded regions.
    sigma_scale : float, optional, default=70
        A factor to multiply the error bars to change their prominence.
    stationary_line_core : float, optional, default=None
        If given, will show a dashed line at this wavelength.
    subtraction : bool, optional, default=False
        Whether to plot the `spectrum` minus emission fit (if exists) instead.
    separate : bool, optional, default=False
        Whether to plot the fitted profiles separately (if multiple components exist).
    show_intensity : bool, optional, default=True
        Whether to show the intensity axis tick labels and axis label.
    show_legend : bool, optional, default=True
        Whether to draw a legend on the axes.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which the fit will be plotted.
        Defaults to the current axis of the current figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes the lines are drawn on.

    See Also
    --------
    mcalf.models.IBIS8542Model.plot : General plotting method.
    mcalf.models.IBIS8542Model.plot_separate : Plot the fit parameters separately.
    mcalf.models.IBIS8542Model.plot_subtraction : Plot the spectrum with the emission fit subtracted from it.
    mcalf.models.FitResult.plot : Plotting method provided by the fit result.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.plot_ibis8542
    """
    if ax is None:
        ax = plt.gca()

    plot_settings = {
        'obs': {'color': '#006BA4', 'label': 'observation'},
        'abs': {'color': '#A2C8EC', 'label': 'absorption profile'},
        'emi': {'color': '#595959', 'label': 'emission profile'},
        'fit': {'color': '#FF800E', 'label': 'fitted profile'},
        'alc': {'color': '#A2C8EC', 'label': 'absorption line core', 'linestyle': '--'},
        'elc': {'color': '#595959', 'label': 'emission line core', 'linestyle': ':'},
        'slc': {'color': '#ABABAB', 'label': 'stationary line core', 'linestyle': '--'},
    }
    hide_existing_labels(plot_settings, fig=ax.get_figure())

    if fit is None:
        show_fit = show_sigma = False
        subtraction = separate = False  # not possible, ignore request
    else:  # fitted parameters provided
        show_fit = True
        show_sigma = False if sigma is None else True
        if len(fit) == 8:  # two components fitted
            fit_function = double_voigt
        else:  # one component fitted
            fit_function = voigt
            subtraction = separate = False  # not possible, ignore request

    if subtraction:
        spectrum = spectrum - voigt(wavelengths, *fit[4:], 0, clib=False)
        plot_settings['obs']['label'] = 'observation - emission'
        show_fit = show_sigma = None

    if show_sigma:  # make a shaded region around the spectral data
        ax.fill_between(wavelengths,
                        spectrum-sigma*sigma_scale, spectrum+sigma*sigma_scale,
                        color='lightgrey')

    # Plot the spectral data
    ax.plot(wavelengths, spectrum, **plot_settings['obs'])

    # Plot the fitted profiles if parameters are given
    if show_fit:

        if separate:  # Plot each component separately

            ax.plot(wavelengths,
                    double_voigt(wavelengths, *fit, background, clib=False),
                    **plot_settings['fit'])
            ax.plot(wavelengths,
                    voigt(wavelengths, *fit[:4], background, clib=False),
                    **plot_settings['abs'])
            ax.plot(wavelengths,
                    voigt(wavelengths, *fit[4:], 0, clib=False),
                    **plot_settings['emi'])

        else:  # Plot a combined profile

            ax.plot(wavelengths,
                    fit_function(wavelengths, *fit, background, clib=False),
                    **plot_settings['fit'])

        # Plot absorption line core
        ax.axvline(x=fit[1], **plot_settings['alc'])

        # Plot emission line core (if fitted)
        if fit_function is double_voigt:
            ax.axvline(x=fit[5], **plot_settings['elc'])

    # Plot stationary line core
    if stationary_line_core is not None:  # Plot vertical dashed line
        ax.axvline(x=stationary_line_core, **plot_settings['slc'])

    # Format the axis ticks and labels:
    ax.minorticks_on()
    ax.set_xlabel('wavelength (Å)')
    if show_intensity:
        ax.set_ylabel('intensity')
    else:  # hide y-axis and axes box
        for loc, spine in ax.spines.items():
            if loc != 'bottom':
                spine.set_color('none')  # don't draw spine
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticks_position('bottom')

    if show_legend:
        bbox = ax.get_position()
        ax.set_position([bbox.x0, bbox.y0, bbox.width * 0.6, bbox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), ncol=1)

    return ax


def plot_spectrum(wavelengths, spectrum, normalised=True, smooth=True, ax=None):
    """Plot a spectrum with the wavelength grid shown.

    Intended for plotting the raw data.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        The x-axis values.
    spectrum : numpy.ndarray, length=n_wavelengths
        The y-axis values.
    normalised : bool, optional, default=True
        Whether to normalise the spectrum using the last three spectral points.
    smooth : bool, optional, default=True
        Whether to smooth the `spectrum` with a spline.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes into which the fit will be plotted.
        Defaults to the current axis of the current figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes the lines are drawn on.

    Examples
    --------
    .. minigallery:: mcalf.visualisation.plot_spectrum
    """
    # Inputted wavelengths and spectrum
    wavelengths_pts = wavelengths.copy()
    spectrum_pts = spectrum.copy()

    if smooth:
        wavelengths = np.linspace(wavelengths[0], wavelengths[-1], num=200)
        spectrum = reinterpolate_spectrum(spectrum, wavelengths_pts, wavelengths)

    if normalised:
        norm = np.mean(spectrum_pts[-3:])
        spectrum_pts /= norm
        spectrum /= norm

    if ax is None:
        ax = plt.gca()

    # Plot vertical lines at each wavelength
    for x in wavelengths_pts:
        ax.axvline(x=x, linestyle='--', linewidth=0.5, color='black')

    # Plot spectral line
    ax.plot(wavelengths, spectrum, color='black')

    # Plot markers at wavelength points
    ax.plot(wavelengths_pts, spectrum_pts, marker='o', linestyle='None', color='black', markersize=4)

    # Set axis labels and ticks
    ax.set_xlabel('wavelength (Å)')
    ylabel = 'normalised intensity ($I/I_c$)' if normalised else 'intensity ($I$)'
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    return ax
