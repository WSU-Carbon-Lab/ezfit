import numpy as np
import math
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def power_law(x, a, b):
    """
    Power-law model: y = a * x^b
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = a * (x[i] ** b)
    return out


@njit(parallel=True, fastmath=True)
def exponential(x, a, b):
    """
    Exponential model: y = a * exp(b * x)
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = a * math.exp(b * x[i])
    return out


@njit(parallel=True, fastmath=True)
def gaussian(x, height, center, std):
    """
    Gaussian model: y = height * exp(-((x - center)^2) / (2 * std^2))
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    denom = 2.0 * (std * std)
    for i in prange(n):
        dx = x[i] - center
        out[i] = height * math.exp(-(dx * dx) / denom)
    return out


@njit(parallel=True, fastmath=True)
def lorentzian(x, height, peak_position, fwhm):
    """
    Lorentzian model: y = height / [1 + ((x - peak_position)^2 / fwhm^2)]
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    fwhm2 = fwhm * fwhm
    for i in prange(n):
        dx = x[i] - peak_position
        out[i] = height / (1.0 + (dx * dx) / fwhm2)
    return out


@njit(parallel=True, fastmath=True)
def pseudo_voigt(x, height, center, fwhm, eta):
    """
    Pseudo-Voigt model:
        y = (1 - eta) * Gaussian(x) + eta * Lorentzian(x)

    Note: The Gaussian std is related to FWHM by:
        std = FWHM / (2 * sqrt(2 ln(2)))
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)

    # Precompute Gaussian and Lorentzian shapes with height=1, then scale later.
    std = fwhm * 0.75845731515  # FWHM_TO_SIGMA
    gauss_part = gaussian(x, 1.0, center, std)  # parallel call
    lorentz_part = lorentzian(x, 1.0, center, fwhm)  # parallel call

    for i in prange(n):
        out[i] = height * ((1.0 - eta) * gauss_part[i] + eta * lorentz_part[i])
    return out


@njit(parallel=True, fastmath=True)
def linear(x, m, b):
    """
    Linear model: y = m*x + b
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = m * x[i] + b
    return out
