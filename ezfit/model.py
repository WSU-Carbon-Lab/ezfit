import inspect
from dataclasses import dataclass
from typing import Any, Generator

import numpy as np


@dataclass
class Parameter:
    """Data class for a parameter and its bounds."""

    value: float = 1
    fixed: bool = False
    min: float = -np.inf
    max: float = np.inf
    err: float = 0

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("Minimum value must be less than maximum value.")

        if self.min > self.value or self.value > self.max:
            raise ValueError("Value must be within the bounds.")

        if self.err < 0:
            raise ValueError("Error must be non-negative.")

        if self.fixed:
            self.min = self.value - np.finfo(float).eps
            self.max = self.value + np.finfo(float).eps

    def __call__(self) -> float:
        return self.value

    def __repr__(self):
        if self.fixed:
            return f"(value={self.value:.10f}, fixed=True)"
        v, e = rounded_values(self.value, self.err, 2)
        return f"(value = {v} ± {e}, bounds = ({self.min}, {self.max}))"

    def random(self) -> float:
        """Returns a valid random value within the bounds."""
        param = np.random.normal(self.value, min(self.err, 1))
        return np.clip(param, self.min, self.max)


@dataclass
class Model:
    """Data class for a model function and its parameters."""

    func: callable
    params: dict[str:Parameter] | None = None
    residuals: np.ndarray | None = None
    cov: np.ndarray | None = None
    cor: np.ndarray | None = None
    𝜒2: float | None = None
    r𝜒2: float | None = None

    def __post_init__(self, params=None):
        """Generate a list of parameters from the function signature."""
        if self.params is None:
            self.params = {}
        input_params = self.params.copy()
        self.params = {}
        for i, name in enumerate(inspect.signature(self.func).parameters):
            if i == 0:
                continue
            self.params[name] = (
                Parameter()
                if name not in input_params
                else Parameter(**input_params[name])
            )

    def __call__(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Evaluate the model at the given x values."""
        nominal = self.func(x, **self.kwargs())
        return nominal

    def __repr__(self):
        name = self.func.__name__
        chi = f"𝜒2: {self.𝜒2}" if self.𝜒2 is not None else "𝜒2: None"
        rchi = f"reduced 𝜒2: {self.r𝜒2}" if self.r𝜒2 is not None else "reduced 𝜒2: None"
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
        return self.params[key]

    def __setitem__(self, key, value: tuple[float, float]):
        self.params[key].value = value[0]
        self.params[key].err = value[1]

    def __iter__(self) -> Generator[Any, Any, Any]:
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
        return {k: v.value for k, v in self.params.items()}

    def random(self, x):
        """Returns a valid random value within the bounds."""
        params = np.array([param.random() for param in self.params.values()])
        return self.func(x, *params)


def sig_fig_round(x, n):
    """Round a number to n significant figures."""
    if x == 0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x))) - (n - 1)))


def rounded_values(x, xerr, n):
    """Round the values and errors to n significant figures."""
    err = sig_fig_round(xerr, n)
    # Round the value to the same number of decimal places as the error
    val = round(x, -int(np.floor(np.log10(err))))
    return val, err
