"""Modeling functions and parameters for ezfit."""

import inspect
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ezfit.types import FitResult


@dataclass
class Parameter:
    """Data class for a parameter and its bounds."""

    value: float = 1
    fixed: bool = False
    min: float = -np.inf
    max: float = np.inf
    err: float = 0

    def __post_init__(self) -> None:
        """Check the parameter values and bounds."""
        if self.min > self.max:
            msg = "Minimum value must be less than maximum value."
            raise ValueError(msg)

        if self.min > self.value or self.value > self.max:
            msg = "Value must be within the bounds."
            raise ValueError(msg)

        if self.err < 0:
            msg = "Error must be non-negative."
            raise ValueError(msg)

        if self.fixed:
            self.min = self.value - float(np.finfo(float).eps)
            self.max = self.value + float(np.finfo(float).eps)

    def __call__(self) -> float:
        """Return the value of the parameter."""
        return self.value

    def __repr__(self) -> str:
        """Return a string representation of the parameter."""
        if self.fixed:
            return f"(value={self.value:.10f}, fixed=True)"
        v, e = rounded_values(self.value, self.err, 2)
        return f"(value = {v} ± {e}, bounds = ({self.min}, {self.max}))"

    def random(self) -> float:
        """Return a valid random value within the bounds."""
        param = np.random.normal(self.value, min(self.err, 1))
        return np.clip(param, self.min, self.max)


@dataclass
class Model:
    """Data class for a model function and its parameters."""

    func: Callable
    params: dict[str, Parameter] | None = None
    residuals: np.ndarray | None = None
    cov: np.ndarray | None = None
    cor: np.ndarray | None = None
    𝜒2: float | None = None
    r𝜒2: float | None = None
    sampler_chain: np.ndarray | None = None
    fit_result_details: FitResult | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Generate a list of parameters from the function signature."""
        if self.params is None:
            self.params = {}
        input_params = self.params.copy()
        self.params = {}
        sig_params = inspect.signature(self.func).parameters
        for i, name in enumerate(sig_params):
            if i == 0:
                continue
            if name in input_params:
                if isinstance(input_params[name], Parameter):
                    self.params[name] = input_params[name]
                elif isinstance(input_params[name], dict):
                    try:
                        self.params[name] = Parameter(**input_params[name])
                    except TypeError as e:
                        raise ValueError(
                            f"Invalid dictionary for parameter '{name}': {input_params[name]}. {e}"
                        ) from e
                else:
                    raise TypeError(
                        f"Parameter '{name}' must be a Parameter object or a dict, got {type(input_params[name])}"
                    )
            else:
                self.params[name] = Parameter()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the model at the given x values."""
        if self.params is None:
            raise ValueError("Model parameters have not been initialized.")
        nominal = self.func(x, **self.kwargs())
        if not isinstance(nominal, np.ndarray):
            nominal = np.asarray(nominal)
        return nominal

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        name = self.func.__name__
        chi = f"𝜒2: {self.𝜒2}" if self.𝜒2 is not None else "𝜒2: None"
        rchi = f"reduced 𝜒2: {self.r𝜒2}" if self.r𝜒2 is not None else "reduced 𝜒2: None"
        if self.params is None:
            return f"{name}\n{chi}\n{rchi}\n"
        params = "\n".join([f"{v} : {param}" for v, param in self.params.items()])
        with np.printoptions(suppress=True, precision=4):
            _cov = (
                self.cov
                if self.cov is not None
                else np.zeros((len(self.params), len(self.params)))
            )
            _cor = (
                self.cor
                if self.cor is not None
                else np.zeros((len(self.params), len(self.params)))
            )
            cov = f"covariance:\n{_cov.__str__()}"
            cor = f"correlation:\n{_cor.__str__()}"
        return f"{name}\n{params}\n{chi}\n{rchi}\n{cov}\n{cor}"

    def __getitem__(self, key) -> Parameter:
        """Return the parameter with the given key."""
        if self.params is None:
            msg = f"Parameter {key} not found in model."
            raise KeyError(msg)
        return self.params[key]

    def __setitem__(self, key: str, value: tuple[float, float]) -> None:
        """Set the parameter with the given key to the given value."""
        if self.params is None:
            msg = f"Parameter {key} not found in model."
            raise KeyError(msg)
        self.params[key].value = value[0]
        self.params[key].err = value[1]

    def __iter__(self) -> Generator[Any, Any, Any]:
        """Iterate over the model parameters."""
        if self.params is None:
            msg = "No parameters found in model."
            raise KeyError(msg)
        yield from [(n, val) for n, val in self.params.items()]

    def values(self) -> list[float]:
        """Yield the model parameters as a list."""
        return [param.value for _, param in iter(self)]

    def bounds(self) -> tuple[list[float], list[float]]:
        """Yield the model parameter bounds as a tuple of lists."""
        return (
            [param.min for _, param in iter(self)],
            [param.max for _, param in iter(self)],
        )

    def kwargs(self) -> dict:
        """Return the model parameters as a dictionary."""
        if self.params is None:
            msg = "No parameters found in model."
            raise KeyError(msg)
        return {k: v.value for k, v in self.params.items()}

    def random(self, x):
        """Return a valid random value within the bounds."""
        if self.params is None:
            msg = "No parameters found in model."
            raise KeyError(msg)
        random_param_values = [param.random() for param in self.params.values()]
        return self.func(x, *random_param_values)

    def describe(self) -> str:
        """Return a string description of the model and its parameters."""
        description = f"Model: {self.func.__name__}\n"
        description += f"Function Signature: {inspect.signature(self.func)}\n"
        description += "Parameters:\n"
        if not self.params:
            description += "  (No parameters defined)\n"
        else:
            for i, (name, p) in enumerate(self.params.items()):
                description += f"  [{i}] {name}: {p}\n"

        if self.𝜒2 is not None:
            description += f"\nChi-squared (𝜒2): {self.𝜒2:.4g}\n"
        if self.r𝜒2 is not None:
            description += f"Reduced Chi-squared (r𝜒2): {self.r𝜒2:.4g}\n"

        return description


def sig_fig_round(x, n):
    """Round a number to n significant figures."""
    if x == 0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x))) - (n - 1)))


def rounded_values(x, xerr, n):
    """Round the values and errors to n significant figures."""
    err = sig_fig_round(xerr, n)
    val = round(x, -int(np.floor(np.log10(err))))
    return val, err
