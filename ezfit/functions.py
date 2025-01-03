import numpy as np
import numba as nb
from numba import types


@nb.vectorize([types.float64(types.float64, types.float64, types.float64)])
def power_law(x, a, b):
    return a * x**b


@nb.vectorize([types.float64(types.float64, types.float64, types.float64)])
def exponential(x, a, b):
    return a * np.exp(b * x)


@nb.vectorize(
    [types.float64(types.float64, types.float64, types.float64, types.float64)]
)
def gaussian(x, height, center, standard_deviation):
    return height * np.exp(-((x - center) ** 2) / (2 * standard_deviation**2))


@nb.vectorize(
    [types.float64(types.float64, types.float64, types.float64, types.float64)]
)
def lorentzian(x, height, peak_position, fwhm):
    return height / (1 + ((x - peak_position) ** 2) / fwhm**2)


@nb.vectorize(
    [
        types.float64(
            types.float64, types.float64, types.float64, types.float64, types.float64
        )
    ]
)
def psudo_voigt(x, height, center, fwhm, eta):
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return (1 - eta) * gaussian(x, height, center, std) + eta * lorentzian(
        x, height, center, fwhm
    )


def linear(x, m, b):
    return m * x + b
