import numpy as np
import numba as nb
from inspect import signature


@nb.njit(cache=True, fastmath=True)
def power_law(x, a, b):
    return a * x**b


@nb.jit(cache=True, fastmath=True)
def exponential(x, a, b):
    return a * np.exp(b * x)


@nb.jit(cache=True, fastmath=True)
def gaussian(x, height, center, standard_deviation):
    return height * np.exp(-((x - center) ** 2) / (2 * standard_deviation**2))


@nb.jit(cache=True, fastmath=True)
def lorentzian(x, height, peak_position, fwhm):
    return height / (1 + ((x - peak_position) / fwhm) ** 2)


@nb.jit(cache=True, fastmath=True)
def psudo_voigt(x, height, center, fwhm, eta):
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return (1 - eta) * gaussian(x, height, center, std) + eta * lorentzian(
        x, height, center, fwhm
    )


@nb.jit(cache=True, fastmatsh=True)
def linear(x, m, b):
    return m * x + b
