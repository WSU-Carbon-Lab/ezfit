"""Fit module for ezfit."""

import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ezfit.exceptions import ColumnNotFoundError
from ezfit.model import Model
from ezfit.optimizers import (  # Import optimizer functions
    _fit_bayesian_ridge,
    _fit_curve_fit,
    _fit_differential_evolution,
    _fit_dual_annealing,
    _fit_elasticnet,
    _fit_emcee,
    _fit_lasso,
    _fit_minimize,
    _fit_polynomial,
    _fit_ridge,
    _fit_shgo,
)
from ezfit.types import (  # Import specific Kwargs types
    BayesianRidgeKwargs,  # Keep FitKwargs for type hinting
    CurveFitKwargs,
    DifferentialEvolutionKwargs,
    DualAnnealingKwargs,
    EmceeKwargs,
    FitKwargs,
    FitMethod,
    MinimizeKwargs,
    ShgoKwargs,
)


@pd.api.extensions.register_dataframe_accessor("fit")
class FitAccessor:
    """Accessor for fitting data in a pandas DataFrame to a given model."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __call__(
        self,
        model: Callable[..., Any],
        x: str,
        y: str,
        yerr: str | None = None,
        plot: bool = True,
        method: FitMethod = "curve_fit",
        fit_kwargs: FitKwargs | None = None,
        residuals: Literal["none", "res", "percent", "rmse"] = "res",
        color_error: str = "C0",
        color_model: str = "C3",
        color_residuals: str = "C0",
        fmt_error: str = ".",
        ls_model: str = "-",
        ls_residuals: str = "",
        marker_residuals: str = ".",
        err_kws: dict[str, Any] | None = None,
        mod_kws: dict[str, Any] | None = None,
        res_kws: dict[str, Any] | None = None,
        **parameters: dict[str, Any],
    ) -> tuple[Model, Axes | None, Axes | None]:
        """Fit the data to the model and optionally plot the results.

        Calls the [FitAccessor.fit](#fitaccessorfit) and
        [FitAccessor.plot](#fitaccessorplot) methods in sequence.

        Args:
            model: The model function to fit the data to.
            x: The name of the column in the DataFrame for the independent variable.
            y: The name of the column in the DataFrame for the dependent variable.
            yerr: The name of the column for the error on the dependent variable.
            plot: Whether to plot the results. Defaults to True.
            method: The fitting method to use. Defaults to "curve_fit".
                    Available methods: 'curve_fit', 'minimize',
                    'differential_evolution', 'shgo', 'dual_annealing',
                    'emcee', 'bayesian_ridge'.
                    'bayesian_ridge' requires scikit-learn and is only valid for
                    linear models. 'emcee' requires emcee.
                    Methods other than 'curve_fit' and 'bayesian_ridge' may require sigma (yerr).
            fit_kwargs: Keyword arguments passed to the fitting function
                        (e.g., `scipy.optimize.curve_fit`, `scipy.optimize.minimize`, etc.).
            residuals: The type of residuals to plot ("none", "res", "percent", "rmse").
                       Defaults to "res".
            color_error: Color for data points/error bars. Defaults to "C0".
            color_model: Color for the fitted model line. Defaults to "C3".
            color_residuals: Color for the residuals plot. Defaults to "C0".
            fmt_error: Marker style for data points. Defaults to ".".
            ls_model: Line style for the model line. Defaults to "-".
            ls_residuals: Line style for residuals. Defaults to "".
            marker_residuals: Marker style for residuals. Defaults to ".".
            err_kws: Additional keyword arguments for data/error bar plotting (`plt.errorbar`).
            mod_kws: Additional keyword arguments for model line plotting (`plt.plot`).
            res_kws: Additional keyword arguments for residuals plotting (`plt.plot`).
            **parameters: Specification of model parameters (initial values, bounds, fixed).
                          Passed as keyword arguments, e.g., `param_name={"value": 1, "min": 0}`.

        Returns
        -------
            A tuple containing the fitted Model object, the main plot Axes (or None),
            and the residuals plot Axes (or None).

        Raises
        ------
            ColumnNotFoundError: If a specified column (x, y, yerr) is not found.
            ImportError: If a required library (e.g., scikit-learn, emcee) is not installed
                         for the chosen method.
            ValueError: If an invalid method is chosen, if required arguments (like sigma)
                        are missing for a method, or if the fit fails.
        """
        if err_kws is None:
            err_kws = {}
        if mod_kws is None:
            mod_kws = {}
        if res_kws is None:
            res_kws = {}

        fitted_model = self.fit(
            model=model,
            x=x,
            y=y,
            yerr=yerr,
            method=method,
            fit_kwargs=fit_kwargs,
            **parameters,
        )

        ax = None
        ax_res = None
        if plot:
            ax, ax_res = self.plot(
                x=x,
                y=y,
                model=fitted_model,
                yerr=yerr,
                residuals=residuals,
                color_error=color_error,
                color_model=color_model,
                color_residuals=color_residuals,
                fmt_error=fmt_error,
                ls_model=ls_model,
                ls_residuals=ls_residuals,
                marker_residuals=marker_residuals,
                err_kws=err_kws,
                mod_kws=mod_kws,
                res_kws=res_kws,
            )

        return fitted_model, ax, ax_res

    def fit(
        self,
        model: Callable[..., Any],
        x: str,
        y: str,
        yerr: str | None = None,
        method: Literal[
            "curve_fit",
            "minimize",
            "differential_evolution",
            "shgo",
            "dual_annealing",
            "emcee",
            "bayesian_ridge",
            "ridge",
            "lasso",
            "elasticnet",
            "polynomial",
        ] = "curve_fit",  # Updated available methods
        fit_kwargs: FitKwargs | None = None,
        **parameters: dict[str, Any],
    ) -> Model:
        """Fit the data to the model.

        Args:
            model: The model function to fit the data to.
            x: The name of the column for the independent variable.
            y: The name of the column for the dependent variable.
            yerr: The name of the column for the error on the dependent variable.
            method: The fitting method to use. Defaults to "curve_fit".
                    Available methods: 'curve_fit', 'minimize',
                    'differential_evolution', 'shgo', 'dual_annealing',
                    'emcee', 'bayesian_ridge'.
                    'bayesian_ridge' requires scikit-learn and is only valid for
                    linear models. 'emcee' requires emcee.
                    Methods other than 'curve_fit' and 'bayesian_ridge' require sigma (yerr).
            fit_kwargs: Keyword arguments passed to the underlying fitting function
                        (e.g., `scipy.optimize.curve_fit`, `scipy.optimize.minimize`,
                        `sklearn.linear_model.BayesianRidge`, `emcee.EnsembleSampler`).
            **parameters: Specification of model parameters (initial values, bounds, fixed).

        Returns
        -------
            The fitted Model object.

        Raises
        ------
            ColumnNotFoundError: If a specified column (x, y, yerr) is not found.
            ImportError: If a required library (e.g., scikit-learn, emcee) is not installed
                         for the chosen method.
            ValueError: If an invalid method is chosen, if required arguments (like sigma)
                        are missing for a method, or if the fit fails.
        """
        # Validate columns
        for col in [x, y] + ([yerr] if yerr else []):
            if col not in self._df.columns:
                raise ColumnNotFoundError(col)

        # Prepare data
        xdata = self._df[x].to_numpy(dtype=float)
        ydata = self._df[y].to_numpy(dtype=float)
        sigma = self._df[yerr].to_numpy(dtype=float) if yerr else None

        # Initialize model
        model_obj = Model(func=model, params=parameters)

        if fit_kwargs is None:
            fit_kwargs = {}

        # --- Select and call the appropriate optimizer ---
        fit_result: FitResult

        try:
            if method == "curve_fit":
                fit_result = _fit_curve_fit(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=cast("CurveFitKwargs", fit_kwargs),  # Cast kwargs
                )
            elif method == "minimize":
                if sigma is None:
                    raise ValueError("Method 'minimize' requires 'yerr' (sigma).")
                fit_result = _fit_minimize(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=cast("MinimizeKwargs", fit_kwargs),  # Cast kwargs
                )
            elif method == "differential_evolution":
                if sigma is None:
                    raise ValueError(
                        "Method 'differential_evolution' requires 'yerr' (sigma)."
                    )
                fit_result = _fit_differential_evolution(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=cast(
                        "DifferentialEvolutionKwargs", fit_kwargs
                    ),  # Cast kwargs
                )
            elif method == "shgo":
                if sigma is None:
                    raise ValueError("Method 'shgo' requires 'yerr' (sigma).")
                fit_result = _fit_shgo(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=cast("ShgoKwargs", fit_kwargs),  # Cast kwargs
                )
            elif method == "dual_annealing":
                if sigma is None:
                    raise ValueError("Method 'dual_annealing' requires 'yerr' (sigma).")
                fit_result = _fit_dual_annealing(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=cast("DualAnnealingKwargs", fit_kwargs),  # Cast kwargs
                )
            elif method == "emcee":
                if sigma is None:
                    raise ValueError("Method 'emcee' requires 'yerr' (sigma).")
                fit_result = _fit_emcee(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=cast("EmceeKwargs", fit_kwargs),  # Cast kwargs
                )
            elif method == "bayesian_ridge":
                # Check for linear model is now inside _fit_bayesian_ridge
                fit_result = _fit_bayesian_ridge(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    # sigma is not directly used by BR fit, but might be needed for chi2 calc
                    sigma=sigma,
                    fit_kwargs=cast("BayesianRidgeKwargs", fit_kwargs),  # Cast kwargs
                )
            elif method == "ridge":
                fit_result = _fit_ridge(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=fit_kwargs or {},
                )
            elif method == "lasso":
                fit_result = _fit_lasso(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=fit_kwargs or {},
                )
            elif method == "elasticnet":
                fit_result = _fit_elasticnet(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=fit_kwargs or {},
                )
            elif method == "polynomial":
                fit_result = _fit_polynomial(
                    model=model_obj,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    fit_kwargs=fit_kwargs or {},
                )
            else:
                # This should ideally be caught by Literal, but added for safety
                msg = f"Unsupported fitting method: {method}"
                raise ValueError(msg)

        except (RuntimeError, ValueError, ImportError) as e:
            # Catch potential errors from optimizer functions or import issues
            msg = f"Fitting with method '{method}' failed: {e}"
            raise ValueError(msg) from e
        # --- End of optimizer selection ---

        # --- Update model_obj with results from fit_result ---
        model_obj.residuals = fit_result["residuals"]
        model_obj.𝜒2 = fit_result["chi2"]
        model_obj.r𝜒2 = fit_result["rchi2"]
        model_obj.cov = fit_result["cov"]
        model_obj.cor = fit_result["cor"]
        model_obj.fit_result_details = fit_result.get(
            "details"
        )  # Store extra details if present

        # Update model parameters with fitted values and errors
        popt = fit_result["popt"]
        perr = fit_result["perr"]
        for i, name in enumerate(model_obj.params):
            # Ensure we don't try to access perr if it's None or shorter than popt
            # (e.g., Bayesian Ridge might not provide all errors)
            error = perr[i] if perr is not None and i < len(perr) else np.nan
            model_obj[name] = (popt[i], error)
        # --- End of model update ---

        return model_obj

    def plot(
        self,
        x: str,
        y: str,
        model: Model,
        yerr: str | None = None,
        ax: Axes | None = None,
        residuals: Literal["none", "res", "percent", "rmse"] = "res",
        color_error: str = "C0",
        color_model: str = "C3",
        color_residuals: str = "C0",
        fmt_error: str = ".",
        ls_model: str = "-",
        ls_residuals: str = "",
        marker_residuals: str = ".",
        err_kws: dict[str, Any] | None = None,
        mod_kws: dict[str, Any] | None = None,
        res_kws: dict[str, Any] | None = None,
    ) -> Axes | tuple[Axes, Axes]:
        """Plot the data, model, and residuals.

        Args:
            x: The name of the column for the independent variable.
            y: The name of the column for the dependent variable.
            model: The fitted Model object containing the function and parameters.
            yerr: The name of the column for the error on the dependent variable.
            ax: An existing Matplotlib Axes object to plot on. If None, a new figure/axes is created.
            residuals: The type of residuals to plot ("none", "res", "percent", "rmse").
            color_error: Color for data points/error bars.
            color_model: Color for the fitted model line.
            color_residuals: Color for the residuals plot.
            fmt_error: Marker style for data points.
            ls_model: Line style for the model line.
            ls_residuals: Line style for residuals.
            marker_residuals: Marker style for residuals.
            err_kws: Additional keyword arguments for `plt.errorbar`.
            mod_kws: Additional keyword arguments for model line `plt.plot`.
            res_kws: Additional keyword arguments for residuals `plt.plot`.

        Returns
        -------
            The main plot Axes object, or a tuple of (main Axes, residuals Axes)
            if residuals are plotted.

        Raises
        ------
            ColumnNotFoundError: If a specified column (x, y, yerr) is not found.
            ValueError: If an invalid residuals metric is specified or model has no parameters.
        """
        if err_kws is None:
            err_kws = {}
        if mod_kws is None:
            mod_kws = {}
        if res_kws is None:
            res_kws = {}

        # Validate columns
        for col in [x, y] + ([yerr] if yerr else []):
            if col not in self._df.columns:
                raise ColumnNotFoundError(col)

        # Prepare data
        xdata = self._df[x].to_numpy(dtype=float)
        ydata = self._df[y].to_numpy(dtype=float)
        sigma = self._df[yerr].to_numpy(dtype=float) if yerr else None

        # Setup plot axes
        if ax is None:
            if residuals != "none":
                fig, (main_ax, res_ax) = plt.subplots(
                    2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
                )
                fig.subplots_adjust(hspace=0.05)
            else:
                fig, main_ax = plt.subplots()
                res_ax = None
        else:
            # Assume user handles subplot creation if providing an axis
            main_ax = ax
            res_ax = (
                None  # Cannot easily create residual plot on user-provided single axis
            )
            if residuals != "none":
                warnings.warn(
                    "Residual plot cannot be automatically created when providing a single Axes object."
                )
                residuals = "none"

        # Plot data
        main_ax.errorbar(
            xdata,
            ydata,
            yerr=sigma,
            fmt=fmt_error,
            color=color_error,
            label="Data",
            **err_kws,
        )

        # Plot model
        if model.params is None:
            raise ValueError("Model has no parameters to evaluate.")

        # Generate smoother x for plotting model line
        x_smooth = np.linspace(xdata.min(), xdata.max(), 500)
        y_model = model(x_smooth)  # Use the Model object's __call__

        main_ax.plot(
            x_smooth, y_model, ls=ls_model, color=color_model, label="Model", **mod_kws
        )
        main_ax.set_ylabel(y)
        main_ax.legend()

        # Plot residuals if requested
        if residuals != "none" and res_ax is not None:
            y_model_at_data = model(xdata)  # Evaluate model at original x points
            res_val = ydata - y_model_at_data

            if residuals == "res":
                if sigma is not None:
                    plot_res = res_val / sigma
                    res_ylabel = "Residuals\n($\\sigma$)"
                else:
                    plot_res = res_val
                    res_ylabel = "Residuals"
            elif residuals == "percent":
                # Avoid division by zero
                valid_idx = ydata != 0
                plot_res = np.full_like(ydata, np.nan)
                plot_res[valid_idx] = 100 * res_val[valid_idx] / ydata[valid_idx]
                res_ylabel = "Residuals\n(%)"
            elif residuals == "rmse":
                plot_res = np.sqrt(res_val**2)
                res_ylabel = "RMSE"
            else:
                msg = f"Invalid residuals type: {residuals}"
                raise ValueError(msg)

            res_ax.plot(
                xdata,
                plot_res,
                marker=marker_residuals,
                ls=ls_residuals,
                color=color_residuals,
                **res_kws,
            )
            res_ax.axhline(0, color="k", linestyle="--", alpha=0.5)
            res_ax.set_xlabel(x)
            res_ax.set_ylabel(res_ylabel)

            # Align y-axis labels
            main_ax.yaxis.set_label_coords(-0.1, 0.5)
            res_ax.yaxis.set_label_coords(-0.1, 0.5)

            return main_ax, res_ax
        else:
            main_ax.set_xlabel(x)
            return main_ax
