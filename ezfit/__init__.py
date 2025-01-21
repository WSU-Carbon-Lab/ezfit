from ezfit.fit import FitAccessor, Model, Parameter
from ezfit.functions import (
    exponential,
    gaussian,
    linear,
    lorentzian,
    power_law,
    pseudo_voigt,
)

__version__ = "0.2.11"

__all__ = [
    "Parameter",
    "Model",
    "FitAccessor",
    "MultiPeakModel",
    "PeakAccessor",
    "power_law",
    "exponential",
    "gaussian",
    "lorentzian",
    "pseudo_voigt",
    "linear",
]
