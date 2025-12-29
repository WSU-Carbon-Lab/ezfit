"""Modeling functions and parameters module for ezfit.

This module provides the core Model and Parameter classes that encapsulate
mathematical models, their parameters, and fit results. These classes form
the foundation of ezfit's fitting interface and provide a unified API for
working with fitted models.

Features
--------
- Parameter management with bounds, constraints, and priors
- Model evaluation and parameter access via dictionary-like interface
- MCMC chain visualization methods (corner plots, trace plots)
- Fit result storage and summary generation
- Automatic parameter initialization from function signatures
"""

import inspect
import warnings
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ezfit.constraints import parse_constraint_string
from ezfit.types import FitResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass
class Parameter:
    """Data class for a parameter and its bounds.

    Attributes
    ----------
    value : float
        Initial/default value of the parameter.
    fixed : bool
        Whether the parameter is fixed (not varied during fitting).
    min : float
        Minimum allowed value (lower bound).
    max : float
        Maximum allowed value (upper bound).
    err : float
        Error/uncertainty on the parameter value.
    constraint : Callable[[dict[str, float]], bool] | None
        Optional constraint function that takes a dict of all parameter values
        and returns True if constraint is satisfied, False otherwise.
        Example: lambda p: p["param1"] + p["param2"] < 1.0
    distribution : Literal["uniform", "normal", "loguniform"] | str | None
        Prior distribution type for MCMC sampling. Options: "uniform", "normal",
        "loguniform". Default is None (uses bounds).
    prior_args : dict[str, Any] | None
        Additional arguments for the prior distribution.
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
            # Test constraint with a dummy parameter dict to check it's callable
            # Use a dict with common parameter names to avoid KeyError
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
                # KeyError is expected if constraint references parameters not in test
                # dict
                # This is okay - we can't know all parameter names at validation time
                # Only raise if it's a TypeError (wrong function signature)
                if isinstance(e, TypeError):
                    msg = (
                        f"constraint function must accept a dict[str, float] "
                        f"and return bool: {e}"
                    )
                    raise TypeError(msg) from e
                # KeyError is acceptable - constraint will be validated at fit time
            except Exception:
                # Other exceptions might indicate a real problem
                # But we'll be lenient and let it pass - will fail at fit time if truly
                # broken
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
        # Handle NaN/Inf in error display
        e_str = ("N/A" if np.isnan(e) else str(e)) if not np.isfinite(e) else str(e)
        return f"(value = {v} ± {e_str}, bounds = ({self.min}, {self.max}))"

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
    x_min: float | None = None
    x_max: float | None = None
    r_squared: float | None = None
    pearson_r: float | None = None
    rmse: float | None = None
    rmsd: float | None = None
    bic: float | None = None
    aic: float | None = None
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
                param_value = input_params[name]

                if isinstance(param_value, Parameter):
                    self.params[name] = param_value
                elif isinstance(param_value, (int, float, np.number)) or (
                    isinstance(param_value, np.ndarray) and param_value.ndim == 0
                ):
                    # Convert numeric value to dict with "value" key
                    param_dict = {"value": float(param_value)}
                    try:
                        self.params[name] = Parameter(**param_dict)
                    except TypeError as e:
                        msg = (
                            f"Invalid numeric value for parameter '{name}': "
                            f"{param_value}. {e}"
                        )
                        raise ValueError(msg) from e
                elif isinstance(param_value, dict):
                    param_dict = cast("dict[str, Any]", param_value).copy()
                    # Parse string constraint if provided
                    if "constraint" in param_dict and isinstance(
                        param_dict["constraint"], str
                    ):
                        # Get all parameter names for parsing
                        all_param_names = [
                            p
                            for p in sig_params
                            if p != next(iter(sig_params.keys()))  # Skip x parameter
                        ]
                        try:
                            constraint_func = parse_constraint_string(
                                param_dict["constraint"], all_param_names
                            )
                            param_dict["constraint"] = constraint_func
                        except ValueError as e:
                            msg = (
                                f"Could not parse constraint string for "
                                f"parameter '{name}': {e}"
                            )
                            warnings.warn(msg, stacklevel=2)
                            param_dict.pop("constraint", None)

                    try:
                        self.params[name] = Parameter(**param_dict)
                    except TypeError as e:
                        msg = (
                            f"Invalid dictionary for parameter '{name}': "
                            f"{input_params[name]}. {e}"
                        )
                        raise ValueError(msg) from e
                else:
                    msg = (
                        f"Parameter '{name}' must be a Parameter object, "
                        f"a numeric value (int, float), or a dict, "
                        f"got {type(input_params[name])}"
                    )
                    raise TypeError(msg)
            else:
                self.params[name] = Parameter()

    def __call__(
        self,
        x: np.ndarray | None = None,
        *,
        start: float | None = None,
        stop: float | None = None,
        num: int = 500,
        endpoint: bool = True,
        spacing: Literal["lin", "log"] = "lin",
    ) -> np.ndarray:
        """Evaluate the model at the given x values or generate a smooth array.

        If x is provided (positional argument), it is used directly for backward
        compatibility. Otherwise, generates a smooth array using start/stop or
        stored x_bounds from fitting.

        Parameters
        ----------
        x : np.ndarray | None, optional
            Direct x values to evaluate. If provided, other parameters are ignored.
            For backward compatibility, by default None.
        start : float | None, optional
            Start value for generating x array. If None, uses stored x_min from fitting.
            By default None.
        stop : float | None, optional
            Stop value for generating x array. If None, uses stored x_max from fitting.
            By default None.
        num : int, optional
            Number of points to generate, by default 500.
        endpoint : bool, optional
            Whether to include the stop value, by default True.
        spacing : Literal["lin", "log"], optional
            Spacing type: "lin" for linear (np.linspace) or "log" for logarithmic
            (np.logspace), by default "lin".

        Returns
        -------
        np.ndarray
            Model evaluation at the specified x values.

        Raises
        ------
        ValueError
            If parameters have not been initialized, or if bounds are not available
            and not provided.
        """
        if self.params is None:
            msg = "Model parameters have not been initialized."
            raise ValueError(msg)

        # Backward compatibility: if x is provided as positional argument,
        # use it directly
        if x is not None:
            nominal = self.func(x, **self.kwargs())
            if not isinstance(nominal, np.ndarray):
                nominal = np.asarray(nominal)
            return nominal

        # Generate x array using bounds
        if start is None:
            if self.x_min is None:
                msg = (
                    "x_min is not available. Either provide 'start' parameter or "
                    "fit the model first to store x bounds."
                )
                raise ValueError(msg)
            start = self.x_min

        if stop is None:
            if self.x_max is None:
                msg = (
                    "x_max is not available. Either provide 'stop' parameter or "
                    "fit the model first to store x bounds."
                )
                raise ValueError(msg)
            stop = self.x_max

        # Generate x array based on spacing type
        if spacing == "log":
            if start <= 0 or stop <= 0:
                msg = "For log spacing, start and stop must be positive."
                raise ValueError(msg)
            x_array = np.logspace(
                np.log10(start), np.log10(stop), num=num, endpoint=endpoint
            )
        else:  # spacing == "lin"
            x_array = np.linspace(start, stop, num=num, endpoint=endpoint)

        nominal = self.func(x_array, **self.kwargs())
        if not isinstance(nominal, np.ndarray):
            nominal = np.asarray(nominal)
        return nominal

    def __repr__(self) -> str:
        """Return a compact string representation of the model."""
        name = self.func.__name__
        lines = [f"{name}()", ""]

        # Parameters table
        if self.params is not None:
            lines.append("  Parameters:")
            lines.append("    Parameter    Value          Error")
            lines.append("    " + "-" * 40)
            for param_name, param in self.params.items():
                if param.fixed:
                    lines.append(
                        f"    {param_name:<12} {param.value:12.6g}   (fixed)"
                    )
                else:
                    if not np.isfinite(param.err) or np.isnan(param.err):
                        e_str = "N/A"
                    else:
                        e_str = f"{param.err:.6g}"
                    lines.append(
                        f"    {param_name:<12} {param.value:12.6g}   ±{e_str}"
                    )
            lines.append("")

        # Statistics
        stats = []
        if self.𝜒2 is not None:
            stats.append(f"𝜒²={self.𝜒2:.6g}")
        if self.r𝜒2 is not None:
            stats.append(f"r𝜒²={self.r𝜒2:.6g}")
        if self.r_squared is not None:
            stats.append(f"R²={self.r_squared:.6f}")
        if self.rmse is not None:
            stats.append(f"RMSE={self.rmse:.6g}")
        if self.bic is not None and np.isfinite(self.bic):
            stats.append(f"BIC={self.bic:.6g}")
        if self.aic is not None and np.isfinite(self.aic):
            stats.append(f"AIC={self.aic:.6g}")

        if stats:
            lines.append("  Statistics:")
            lines.append("    " + ", ".join(stats))
            lines.append("")

        # Covariance and correlation matrices
        if self.params is not None and len(self.params) > 0:
            param_names = list(self.params.keys())
            if self.cov is not None:
                lines.append("  Covariance Matrix:")
                lines.append(self._matrix_to_text(self.cov, param_names, indent="    "))
                lines.append("")
            if self.cor is not None:
                lines.append("  Correlation Matrix:")
                lines.append(self._matrix_to_text(self.cor, param_names, indent="    "))
                lines.append("")

        return "\n".join(lines).rstrip()

    def _repr_html_(self) -> str:
        """Return compact HTML representation for Jupyter notebooks."""
        name = self.func.__name__
        div_style = "font-family: monospace; line-height: 1.4;"
        html_parts = [f'<div style="{div_style}">']

        # Header
        h3_style = "margin: 0 0 8px 0; color: #0066cc; font-size: 1.1em;"
        html_parts.append(f'<h3 style="{h3_style}"><strong>{name}()</strong></h3>')

        # Parameters table
        if self.params is not None:
            html_parts.append('<h4 style="color: #333; margin: 8px 0 4px 0;">Parameters</h4>')
            table_style = (
                "border-collapse: collapse; margin-bottom: 12px; font-size: 0.95em;"
            )
            html_parts.append(f'<table style="{table_style}">')

            # Header row
            th_style = (
                "padding: 6px 10px; text-align: left; border: 1px solid #ddd; "
                "background-color: #f0f0f0; font-weight: bold;"
            )
            html_parts.append(
                f"<tr>"
                f'<th style="{th_style}">Parameter</th>'
                f'<th style="{th_style}">Value</th>'
                f'<th style="{th_style}">Error</th>'
                f"</tr>"
            )

            # Data rows
            td_style = "padding: 6px 10px; border: 1px solid #ddd;"
            for param_name, param in self.params.items():
                if param.fixed:
                    value_str = f"{param.value:.6g}"
                    error_str = '<span style="color:#888;">(fixed)</span>'
                else:
                    value_str = f"{param.value:.6g}"
                    if not np.isfinite(param.err) or np.isnan(param.err):
                        error_str = "N/A"
                    else:
                        error_str = f"±{param.err:.6g}"

                html_parts.append(
                    f"<tr>"
                    f'<td style="{td_style}"><strong>{param_name}</strong></td>'
                    f'<td style="{td_style}">{value_str}</td>'
                    f'<td style="{td_style}">{error_str}</td>'
                    f"</tr>"
                )
            html_parts.append("</table>")

        # Statistics
        stats = []
        if self.𝜒2 is not None:
            stats.append(f"𝜒²={self.𝜒2:.6g}")
        if self.r𝜒2 is not None:
            stats.append(f"r𝜒²={self.r𝜒2:.6g}")
        if self.r_squared is not None:
            stats.append(f"R²={self.r_squared:.6f}")
        if self.rmse is not None:
            stats.append(f"RMSE={self.rmse:.6g}")
        if self.bic is not None and np.isfinite(self.bic):
            stats.append(f"BIC={self.bic:.6g}")
        if self.aic is not None and np.isfinite(self.aic):
            stats.append(f"AIC={self.aic:.6g}")

        if stats:
            html_parts.append('<h4 style="color: #333; margin: 8px 0 4px 0;">Statistics</h4>')
            html_parts.append(f'<p style="margin: 0 0 12px 0;">{", ".join(stats)}</p>')

        # Covariance and correlation matrices
        if self.params is not None and len(self.params) > 0:
            param_names = list(self.params.keys())
            matrices = []
            if self.cov is not None:
                matrices.append(("Covariance Matrix", self.cov))
            if self.cor is not None:
                matrices.append(("Correlation Matrix", self.cor))

            if matrices:
                matrix_container_style = "display: flex; gap: 20px; margin-top: 8px;"
                html_parts.append(f'<div style="{matrix_container_style}">')
                for label, matrix in matrices:
                    html_parts.append('<div>')
                    html_parts.append(
                        f'<h4 style="color: #333; margin: 8px 0 4px 0; font-size: 0.95em;">{label}</h4>'
                    )
                    html_parts.append(self._matrix_to_html(matrix, param_names))
                    html_parts.append("</div>")
                html_parts.append("</div>")

        html_parts.append("</div>")
        return "".join(html_parts)

    def _matrix_to_text(
        self, matrix: np.ndarray, param_names: list[str], indent: str = ""
    ) -> str:
        """Convert a numpy matrix to formatted text table."""
        lines = []
        # Calculate column widths
        col_width = max(len(name) for name in param_names) + 2
        num_width = 12

        # Header row
        header = indent + " " * col_width
        for name in param_names:
            header += f"{name:>{num_width}}"
        lines.append(header)
        lines.append(indent + "-" * (col_width + num_width * len(param_names)))

        # Data rows
        with np.printoptions(suppress=True, precision=6):
            for i, name in enumerate(param_names):
                row = indent + f"{name:<{col_width}}"
                for j in range(len(param_names)):
                    val = matrix[i, j]
                    row += f"{val:>{num_width}.6g}"
                lines.append(row)

        return "\n".join(lines)

    def _matrix_to_html(self, matrix: np.ndarray, param_names: list[str]) -> str:
        """Convert a numpy matrix to HTML table."""
        table_style = (
            "border-collapse: collapse; font-size: 0.9em; margin-bottom: 8px;"
        )
        html_parts = [f'<table style="{table_style}">']
        th_style = (
            "padding: 6px 10px; text-align: center; border: 1px solid #ddd; "
            "background-color: #f0f0f0; font-weight: bold;"
        )
        html_parts.append(f'<tr><th style="{th_style}"></th>')
        for name in param_names:
            html_parts.append(f'<th style="{th_style}">{name}</th>')
        html_parts.append("</tr>")

        td_style = "padding: 6px 10px; border: 1px solid #ddd; text-align: right;"
        td_bold = f"{td_style} background-color: #f0f0f0; font-weight: bold;"

        with np.printoptions(suppress=True, precision=6):
            for i, name in enumerate(param_names):
                html_parts.append(f"<tr><td style='{td_bold}'>{name}</td>")
                for j in range(len(param_names)):
                    val = matrix[i, j]
                    html_parts.append(f'<td style="{td_style}">{val:.6g}</td>')
                html_parts.append("</tr>")
        html_parts.append("</table>")
        return "".join(html_parts)

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
        yield from list(self.params.items())

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

        description += "\nGoodness-of-fit Statistics:\n"

        # Show chi-squared statistics if available
        if self.𝜒2 is not None:
            description += f"  Chi-squared (𝜒2): {self.𝜒2:.4g}\n"
        if self.r𝜒2 is not None:
            description += f"  Reduced Chi-squared (r𝜒2): {self.r𝜒2:.4g}\n"

        # Show alternative statistics
        if self.r_squared is not None:
            description += (
                f"  R² (coefficient of determination): {self.r_squared:.4g}\n"
            )
        if self.pearson_r is not None and not np.isnan(self.pearson_r):
            description += (
                f"  Pearson correlation coefficient (r): {self.pearson_r:.4g}\n"
            )
        if self.rmse is not None:
            description += f"  RMSE (Root Mean Square Error): {self.rmse:.4g}\n"
        if self.bic is not None and np.isfinite(self.bic):
            description += f"  BIC (Bayesian Information Criterion): {self.bic:.4g}\n"
        if self.aic is not None and np.isfinite(self.aic):
            description += f"  AIC (Akaike Information Criterion): {self.aic:.4g}\n"

        return description

    def plot_corner(self, **kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """Create a corner plot from MCMC chain if available.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Additional keyword arguments passed to plot_corner.

        Returns
        -------
        tuple[Any, Any]
            Tuple of (figure, axes_array).

        Raises
        ------
        ValueError
            If no MCMC chain is available.
        """
        from ezfit.visualization import plot_corner

        if self.sampler_chain is None:
            msg = "No MCMC chain available. Use method='emcee' to generate a chain."
            raise ValueError(msg)

        param_names = list(self.params.keys()) if self.params else None
        return plot_corner(self.sampler_chain, param_names=param_names, **kwargs)  # type: ignore[arg-type]

    def plot_trace(self, **kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """Create trace plots from MCMC chain if available.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Additional keyword arguments passed to plot_trace.

        Returns
        -------
        tuple[Any, Any]
            Tuple of (figure, axes_array).

        Raises
        ------
        ValueError
            If no MCMC chain is available.
        """
        from ezfit.visualization import plot_trace

        if self.sampler_chain is None:
            msg = "No MCMC chain available. Use method='emcee' to generate a chain."
            raise ValueError(msg)

        param_names = list(self.params.keys()) if self.params else None
        return plot_trace(self.sampler_chain, param_names=param_names, **kwargs)  # type: ignore[arg-type]

    def get_posterior_samples(
        self, discard: int | None = None, thin: int | None = None
    ) -> np.ndarray:
        """Get posterior samples from MCMC chain.

        Parameters
        ----------
        discard : int | None, optional
            Number of samples to discard as burn-in. If None, uses automatic detection,
            by default None.
        thin : int | None, optional
            Thinning factor. If None, uses automatic thinning, by default None.

        Returns
        -------
        np.ndarray
            Array of posterior samples with shape (n_samples, n_params).

        Raises
        ------
        ValueError
            If no MCMC chain is available.
        """
        if self.sampler_chain is None:
            msg = "No MCMC chain available. Use method='emcee' to generate a chain."
            raise ValueError(msg)

        chain = self.sampler_chain

        # Apply discard and thin if provided
        if discard is not None:
            chain = chain[:, discard:, :] if chain.ndim == 3 else chain[discard:, :]

        if thin is not None and thin > 1:
            chain = chain[:, ::thin, :] if chain.ndim == 3 else chain[::thin, :]

        # Flatten if 3D
        if chain.ndim == 3:
            chain = chain.reshape(-1, chain.shape[-1])

        return chain

    def summary(self) -> str:
        """Print a summary of the fit including diagnostics.

        Returns
        -------
        str
            String summary of the fit including model description, parameters,
            chi-squared statistics, and MCMC diagnostics if available.
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
                if isinstance(rhat, int | float):
                    summary_lines.append(f"  R-hat: {rhat:.4f}")
                else:
                    summary_lines.append(f"  R-hat: {rhat}")
                ess = diagnostics.get("ess", "N/A")
                if isinstance(ess, int | float):
                    summary_lines.append(f"  ESS: {ess:.2f}")
                else:
                    summary_lines.append(f"  ESS: {ess}")
                summary_lines.append(f"  Burn-in: {diagnostics.get('burnin', 'N/A')}")
                n_eff = diagnostics.get("n_effective_samples", "N/A")
                summary_lines.append(f"  Effective samples: {n_eff}")
                converged = diagnostics.get("converged", False)
                summary_lines.append(f"  Converged: {converged}")

        return "\n".join(summary_lines)

    def plot(
        self,
        x: np.ndarray | None = None,
        *,
        start: float | None = None,
        stop: float | None = None,
        num: int = 500,
        endpoint: bool = True,
        spacing: Literal["lin", "log"] = "lin",
        ax: "Axes | None" = None,
        **kwargs: Any,
    ) -> "Axes":
        """Plot the model curve.

        Parameters
        ----------
        x : np.ndarray | None, optional
            X values to evaluate and plot the model at. If provided, `start`, `stop`,
            `num`, `endpoint`, and `spacing` are ignored. By default None.
        start : float | None, optional
            Start value for generating x array. If None, uses stored x_min from fitting.
            By default None.
        stop : float | None, optional
            Stop value for generating x array. If None, uses stored x_max from fitting.
            By default None.
        num : int, optional
            Number of points to generate, by default 500.
        endpoint : bool, optional
            Whether to include the stop value, by default True.
        spacing : Literal["lin", "log"], optional
            Spacing type: "lin" for linear (np.linspace) or "log" for logarithmic
            (np.logspace), by default "lin".
        ax : Axes | None, optional
            Matplotlib axes to plot on. If None, creates a new figure and axes.
            By default None.
        **kwargs : Any
            All additional keyword arguments are passed directly to matplotlib's
            `plot()` function for styling (e.g., `color`, `linestyle`, `label`,
            `marker`).

        Returns
        -------
        Axes
            The matplotlib axes object.

        Raises
        ------
        ValueError
            If parameters have not been initialized, or if bounds are not available
            and not provided.

        Examples
        --------
        >>> import numpy as np
        >>> from ezfit.functions import linear
        >>> model = Model(
        ...     linear, {"m": Parameter(value=2.0), "b": Parameter(value=1.0)}
        ... )
        >>> ax = model.plot(start=0, stop=10)
        >>> # Custom styling
        >>> ax = model.plot(
        ...     start=0, stop=10, color="blue", linestyle="--", label="My Model"
        ... )
        >>> # Use specific x values
        >>> x_vals = np.linspace(0, 10, 100)
        >>> ax = model.plot(x=x_vals, color="red")
        """
        import matplotlib.pyplot as plt

        if self.params is None:
            msg = "Model parameters have not been initialized."
            raise ValueError(msg)

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots()  # noqa: RUF059

        # Generate x values for model curve
        if x is not None:
            x_model = x
        else:
            # Determine x bounds
            if start is None:
                start = self.x_min
            if stop is None:
                stop = self.x_max
            if start is None or stop is None:
                msg = (
                    "Cannot determine x bounds for plotting. Either provide 'x' array, "
                    "or 'start' and 'stop' parameters, or fit the model first to store "
                    "x bounds."
                )
                raise ValueError(msg)

            # Generate x values for model curve
            if spacing == "log":
                if start <= 0 or stop <= 0:
                    msg = "For log spacing, start and stop must be positive."
                    raise ValueError(msg)
                x_model = np.logspace(
                    np.log10(start), np.log10(stop), num=num, endpoint=endpoint
                )
            else:
                x_model = np.linspace(start, stop, num=num, endpoint=endpoint)

        # Generate model curve
        y_model = self(x_model)

        # Plot model curve with all kwargs passed to matplotlib
        ax.plot(x_model, y_model, **kwargs)

        return ax


def sig_fig_round(x: float, n: int) -> float:
    """Round a number to n significant figures.

    Parameters
    ----------
    x : float
        Number to round.
    n : int
        Number of significant figures.

    Returns
    -------
    float
        Rounded number with n significant figures.
    """
    if x == 0:
        return 0
    if not np.isfinite(x):
        # Handle NaN and Inf values
        return x
    return round(x, -int(np.floor(np.log10(abs(x))) - (n - 1)))


def rounded_values(x: float, xerr: float, n: int) -> tuple[float, float]:
    """Round the values and errors to n significant figures.

    Parameters
    ----------
    x : float
        Value to round.
    xerr : float
        Error to round.
    n : int
        Number of significant figures for error.

    Returns
    -------
    tuple[float, float]
        Tuple of (rounded_value, rounded_error).
    """
    err = sig_fig_round(xerr, n)
    if not np.isfinite(err) or err == 0:
        # Handle NaN, Inf, or zero error - just round the value normally
        val = round(x, n) if np.isfinite(x) else x
    else:
        val = round(x, -int(np.floor(np.log10(err))))
    return val, err
