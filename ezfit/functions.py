"""
Numba-optimized functions for fitting.
"""

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def power_law(x, a, b):
    """
    Power law function: y = a * x^b.
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = a * (x[i] ** b)
    return out


@njit(parallel=True, fastmath=True)
def exponential(x, a, b):
    """
    Exponential function: y = a * exp(b * x).
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = a * math.exp(b * x[i])
    return out


@njit(parallel=True, fastmath=True)
def gaussian(x, amplitude, center, fwhm):
    """
    Gaussian with peak = 'amplitude' and FWHM = 'fwhm'.

    Formula:
      G(x) = amplitude * exp[-4 ln(2) * ((x - center) / fwhm)^2]

    At x=center, G = amplitude.
    The half max occurs at |x-center| = fwhm/2.
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
    """
    Lorentzian with peak = 'amplitude' and FWHM = 'fwhm'.

    L(x) = amplitude * [ (fwhm/2)^2 / ((x-center)^2 + (fwhm/2)^2 ) ]

    At x=center, L = amplitude.
    The half max occurs at |x-center| = fwhm/2.
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
    """
    Pseudo-Voigt function with peak = 'height' and FWHM = 'fwhm'.

    Pseudo-Voigt model (peak-based):
        y = height * [ (1 - eta)*G + eta*L ]

    where G and L have the same FWHM = 'fwhm' and both peak at 1.0
    when we pass amplitude=1.

    That is:
      G(x) = 1 * exp[-4 ln(2) * ((x-center)/fwhm)^2]
      L(x) = 1 * ((fwhm/2)^2 / ((x-center)^2 + (fwhm/2)^2))

    Then the final amplitude is scaled by 'height'.
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
    """
    Linear function: y = m*x + b.
    """
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = m * x[i] + b
    return out
