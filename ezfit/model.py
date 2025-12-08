"""Modeling functions and parameters for ezfit."""

import inspect
import warnings
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ezfit.constraints import parse_constraint_string
from ezfit.types import FitResult


@dataclass
class Parameter:
    """Data class for a parameter and its bounds.

    Attributes
    ----------
        value: Initial/default value of the parameter.
        fixed: Whether the parameter is fixed (not varied during fitting).
        min: Minimum allowed value (lower bound).
        max: Maximum allowed value (upper bound).
        err: Error/uncertainty on the parameter value.
        constraint: Optional constraint function that takes a dict of all parameter
            values and returns True if constraint is satisfied, False otherwise.
            Example: lambda p: p["param1"] + p["param2"] < 1.0
        distribution: Prior distribution type for MCMC sampling. Options:
            "uniform", "normal", "loguniform". Default is None (uses bounds).
        prior_args: Additional arguments for the prior distribution.
            For "normal": {"loc": mean, "scale": std}
            For "uniform": {"low": min, "high": max} (usually same as min/max)
            For "loguniform": {"low": min, "high": max}
    """

    value: float = 1
    fixed: bool = False
    min: float = -np.inf
    max: float = np.inf
    err: float = 0
    constraint: Callable[[dict[str, float]], bool] | None = None
    distribution: Literal["uniform", "normal", "loguniform"] | str | None = None
    prior_args: dict[str, Any] | None = None

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

        # Validate constraint function if provided
        if self.constraint is not None:
            if not callable(self.constraint):
                msg = "constraint must be a callable function."
                raise TypeError(msg)
            # Test constraint with a dummy parameter dict to check it's callable correctly
            # Use a dict with common parameter names to avoid KeyError for valid constraints
            try:
                test_params = {
                    "test_param": 1.0,
                    "A1": 1.0,
                    "A2": 1.0,
                    "param1": 1.0,
                    "param2": 1.0,
                    "m": 1.0,
                    "b": 1.0,
                    "w1": 1.0,
                    "w2": 1.0,
                    "c1": 1.0,
                    "c2": 1.0,
                }
                _ = self.constraint(test_params)
            except (KeyError, TypeError) as e:
                # KeyError is expected if constraint references parameters not in test dict
                # This is okay - we can't know all parameter names at validation time
                # Only raise if it's a TypeError (wrong function signature)
                if isinstance(e, TypeError):
                    msg = f"constraint function must accept a dict[str, float] and return bool: {e}"
                    raise ValueError(msg) from e
                # KeyError is acceptable - constraint will be validated at fit time
            except Exception:
                # Other exceptions might indicate a real problem
                # But we'll be lenient and let it pass - will fail at fit time if truly broken
                pass

        # Validate distribution if provided
        if self.distribution is not None:
            valid_distributions = ["uniform", "normal", "loguniform"]
            if self.distribution not in valid_distributions:
                warnings.warn(
                    f"Unknown distribution '{self.distribution}'. "
                    f"Valid options: {valid_distributions}",
                    stacklevel=2,
                )

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
                    param_dict = input_params[name].copy()
                    # Parse string constraint if provided
                    if "constraint" in param_dict and isinstance(
                        param_dict["constraint"], str
                    ):
                        # Get all parameter names for parsing
                        all_param_names = [
                            p
                            for p in sig_params.keys()
                            if p != list(sig_params.keys())[0]  # Skip x parameter
                        ]
                        try:
                            constraint_func = parse_constraint_string(
                                param_dict["constraint"], all_param_names
                            )
                            param_dict["constraint"] = constraint_func
                        except ValueError as e:
                            warnings.warn(
                                f"Could not parse constraint string for parameter '{name}': {e}",
                                stacklevel=2,
                            )
                            param_dict.pop("constraint", None)

                    try:
                        self.params[name] = Parameter(**param_dict)
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

    def plot_corner(self, **kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """Create a corner plot from MCMC chain if available.

        Args:
            **kwargs: Additional keyword arguments passed to plot_corner.

        Returns
        -------
            Tuple of (figure, axes_array).

        Raises
        ------
            ValueError: If no MCMC chain is available.
        """
        from ezfit.visualization import plot_corner

        if self.sampler_chain is None:
            msg = "No MCMC chain available. Use method='emcee' to generate a chain."
            raise ValueError(msg)

        param_names = list(self.params.keys()) if self.params else None
        return plot_corner(self.sampler_chain, param_names=param_names, **kwargs)

    def plot_trace(self, **kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """Create trace plots from MCMC chain if available.

        Args:
            **kwargs: Additional keyword arguments passed to plot_trace.

        Returns
        -------
            Tuple of (figure, axes_array).

        Raises
        ------
            ValueError: If no MCMC chain is available.
        """
        from ezfit.visualization import plot_trace

        if self.sampler_chain is None:
            msg = "No MCMC chain available. Use method='emcee' to generate a chain."
            raise ValueError(msg)

        param_names = list(self.params.keys()) if self.params else None
        return plot_trace(self.sampler_chain, param_names=param_names, **kwargs)

    def get_posterior_samples(self, discard: int | None = None, thin: int | None = None) -> np.ndarray:
        """Get posterior samples from MCMC chain.

        Args:
            discard: Number of samples to discard as burn-in. If None, uses automatic detection.
            thin: Thinning factor. If None, uses automatic thinning.

        Returns
        -------
            Array of posterior samples.

        Raises
        ------
            ValueError: If no MCMC chain is available.
        """
        if self.sampler_chain is None:
            msg = "No MCMC chain available. Use method='emcee' to generate a chain."
            raise ValueError(msg)

        chain = self.sampler_chain

        # Apply discard and thin if provided
        if discard is not None:
            if chain.ndim == 3:
                chain = chain[:, discard:, :]
            else:
                chain = chain[discard:, :]

        if thin is not None and thin > 1:
            if chain.ndim == 3:
                chain = chain[:, ::thin, :]
            else:
                chain = chain[::thin, :]

        # Flatten if 3D
        if chain.ndim == 3:
            chain = chain.reshape(-1, chain.shape[-1])

        return chain

    def summary(self) -> str:
        """Print a summary of the fit including diagnostics.

        Returns
        -------
            String summary of the fit.
        """
        summary_lines = [self.describe()]

        # Add MCMC diagnostics if available
        if self.fit_result_details is not None and isinstance(
            self.fit_result_details, dict
        ):
            diagnostics = self.fit_result_details.get("diagnostics")
            if diagnostics is not None:
                summary_lines.append("\nMCMC Diagnostics:")
                rhat = diagnostics.get("rhat", "N/A")
                if isinstance(rhat, (int, float)):
                    summary_lines.append(f"  R-hat: {rhat:.4f}")
                else:
                    summary_lines.append(f"  R-hat: {rhat}")
                ess = diagnostics.get("ess", "N/A")
                if isinstance(ess, (int, float)):
                    summary_lines.append(f"  ESS: {ess:.2f}")
                else:
                    summary_lines.append(f"  ESS: {ess}")
                summary_lines.append(
                    f"  Burn-in: {diagnostics.get('burnin', 'N/A')}"
                )
                summary_lines.append(
                    f"  Effective samples: {diagnostics.get('n_effective_samples', 'N/A')}"
                )
                converged = diagnostics.get("converged", False)
                summary_lines.append(f"  Converged: {converged}")

        return "\n".join(summary_lines)


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
