"""
Numba-optimized functions for fitting.
"""

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def power_law(x, a, b):
    """Power law function.

    Computes a power law relationship of the form y = a * x^b, where a is the
    coefficient and b is the exponent.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    a : float
        Coefficient (prefactor) of the power law.
    b : float
        Exponent of the power law.

    Returns
    -------
    array-like
        Dependent variable values computed as y = a * x^b.

    References
    ----------
    .. [1] Power law. Wikipedia. https://en.wikipedia.org/wiki/Power_law

    Examples
    --------
    >>> import numpy as np
    >>> from ezfit import power_law
    >>> x = np.linspace(1, 10, 100)
    >>> y = power_law(x, a=2.0, b=1.5)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = a * (x[i] ** b)
    return out


@njit(parallel=True, fastmath=True)
def exponential(x, a, b):
    """Exponential function.

    Computes an exponential relationship of the form y = a * exp(b * x),
    where a is the amplitude and b is the decay/growth rate.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    a : float
        Amplitude (prefactor) of the exponential.
    b : float
        Decay rate (if negative) or growth rate (if positive).

    Returns
    -------
    array-like
        Dependent variable values computed as y = a * exp(b * x).

    References
    ----------
    .. [1] Exponential function. Wikipedia. https://en.wikipedia.org/wiki/Exponential_function
    .. [2] Exponential decay. Wikipedia. https://en.wikipedia.org/wiki/Exponential_decay

    Examples
    --------
    >>> import numpy as np
    >>> from ezfit import exponential
    >>> x = np.linspace(0, 10, 100)
    >>> y = exponential(x, a=1.0, b=-0.5)  # Exponential decay
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = a * math.exp(b * x[i])
    return out


@njit(parallel=True, fastmath=True)
def gaussian(x, amplitude, center, fwhm):
    """Gaussian (normal) distribution function.

    Computes a Gaussian peak with specified amplitude, center position, and
    full width at half maximum (FWHM).

    The function is defined as::

        G(x) = amplitude * exp[-4 * ln(2) * ((x - center) / fwhm)^2]

    At x = center, the function reaches its maximum value of amplitude.
    The half maximum occurs at ``|x-center| = fwhm / 2``.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    amplitude : float
        Peak amplitude (maximum value at x = center).
    center : float
        Center position (mean) of the Gaussian.
    fwhm : float
        Full width at half maximum. Must be positive.

    Returns
    -------
    array-like
        Dependent variable values of the Gaussian function.

    References
    ----------
    .. [1] Normal distribution. Wikipedia. https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] Gaussian function. Wikipedia. https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    >>> import numpy as np
    >>> from ezfit import gaussian
    >>> x = np.linspace(-5, 5, 100)
    >>> y = gaussian(x, amplitude=1.0, center=0.0, fwhm=2.0)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    c = 4.0 * math.log(2.0)  # ~2.7726
    for i in prange(n):
        dx = (x[i] - center) / fwhm
        out[i] = amplitude * math.exp(-c * dx * dx)
    return out


@njit(parallel=True, fastmath=True)
def lorentzian(x, amplitude, center, fwhm):
    """Lorentzian (Cauchy) distribution function.

    Computes a Lorentzian peak with specified amplitude, center position, and
    full width at half maximum (FWHM).

    The function is defined as::

        L(x) = amplitude * [ (fwhm/2)^2 / ((x - center)^2 + (fwhm/2)^2) ]

    At x = center, the function reaches its maximum value of amplitude.
    The half maximum occurs at ``|x-center| = fwhm / 2``.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    amplitude : float
        Peak amplitude (maximum value at x = center).
    center : float
        Center position (location parameter) of the Lorentzian.
    fwhm : float
        Full width at half maximum. Must be positive.

    Returns
    -------
    array-like
        Dependent variable values of the Lorentzian function.

    References
    ----------
    .. [1] Cauchy distribution. Wikipedia. https://en.wikipedia.org/wiki/Cauchy_distribution
    .. [2] Spectral line shape. Wikipedia. https://en.wikipedia.org/wiki/Spectral_line_shape#Lorentzian

    Examples
    --------
    >>> import numpy as np
    >>> from ezfit import lorentzian
    >>> x = np.linspace(-5, 5, 100)
    >>> y = lorentzian(x, amplitude=1.0, center=0.0, fwhm=2.0)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    gamma = 0.5 * fwhm
    gamma2 = gamma * gamma
    for i in prange(n):
        dx = x[i] - center
        out[i] = amplitude * (gamma2 / (dx * dx + gamma2))
    return out


@njit(parallel=True, fastmath=True)
def pseudo_voigt(x, height, center, fwhm, eta):
    """Pseudo-Voigt profile function.

    Computes a pseudo-Voigt profile, which is a weighted linear combination
    of a Gaussian and a Lorentzian profile. This function is commonly used
    to model spectral line shapes and diffraction peaks.

    The function is defined as::

        y = height * [(1 - eta) * G + eta * L]

    where G is a normalized Gaussian and L is a normalized Lorentzian, both
    with the same FWHM and center position. The mixing parameter eta controls
    the relative contribution of each component
    (0 = pure Gaussian, 1 = pure Lorentzian).

    Parameters
    ----------
    x : array-like
        Independent variable values.
    height : float
        Peak height (maximum value at x = center).
    center : float
        Center position of the profile.
    fwhm : float
        Full width at half maximum. Must be positive.
    eta : float
        Mixing parameter between 0 and 1. Controls the relative contribution
        of Gaussian (1-eta) and Lorentzian (eta) components.
        - eta = 0: Pure Gaussian
        - eta = 1: Pure Lorentzian
        - 0 < eta < 1: Mixed profile

    Returns
    -------
    array-like
        Dependent variable values of the pseudo-Voigt function.

    References
    ----------
    .. [1] Voigt profile. Wikipedia. https://en.wikipedia.org/wiki/Voigt_profile
    .. [2] Pseudo-Voigt profile. Wikipedia. https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation

    Examples
    --------
    >>> import numpy as np
    >>> from ezfit import pseudo_voigt
    >>> x = np.linspace(-5, 5, 100)
    >>> # 50% Gaussian, 50% Lorentzian
    >>> y = pseudo_voigt(x, height=1.0, center=0.0, fwhm=2.0, eta=0.5)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)

    gauss_part = gaussian(x, 1.0, center, fwhm)  # peak=1
    lorentz_part = lorentzian(x, 1.0, center, fwhm)  # peak=1

    for i in prange(n):
        # Weighted sum: (1-eta)*Gauss + eta*Lorentz, then scale by 'height'
        out[i] = height * ((1.0 - eta) * gauss_part[i] + eta * lorentz_part[i])
    return out


@njit(parallel=True, fastmath=True)
def linear(x, m, b):
    """Linear function.

    The function is defined as::

        y = m * x + b

    where m is the slope and b is the y-intercept.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    m : float
        Slope of the line.
    b : float
        Y-intercept (value when x = 0).

    Returns
    -------
    array-like
        Dependent variable values computed as y = m * x + b.

    References
    ----------
    .. [1] Linear function. Wikipedia. https://en.wikipedia.org/wiki/Linear_function
    .. [2] Linear equation. Wikipedia. https://en.wikipedia.org/wiki/Linear_equation

    Examples
    --------
    >>> import numpy as np
    >>> from ezfit import linear
    >>> x = np.linspace(0, 10, 100)
    >>> y = linear(x, m=2.0, b=1.0)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = m * x[i] + b
    return out
