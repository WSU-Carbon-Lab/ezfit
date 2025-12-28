"""Fit module for ezfit.

This module provides the primary interface for fitting mathematical models to data
stored in a pandas DataFrame. It offers highly flexible fitting routines, comprehensive
data validation, and powerful plotting tools, all accessible via a modern
pandas DataFrame accessor, `df.fit`.

Features
--------
- Versatile model fitting with support for a wide range of optimizers:
  curve_fit, minimize, differential evolution, SHGO, dual annealing (from SciPy),
  MCMC via emcee, and regularized regressions (ridge, lasso, elasticnet, bayesian_ridge)
  via scikit-learn.
- Robust data validation, error checking, and convenient data extraction.
- Direct integration with pandas DataFrames, enabling one-liner fitting workflows.
- Customizable and publication-ready plotting of data, models, uncertainties, and
  residuals.
- Extensible type system for fit methods and argument validation.

Classes
-------
FitData
    Container for validated and extracted independent, dependent, and error data.
PlotOptions
    Stores configuration for fit visualizations.
FitAccessor
    Pandas DataFrame accessor for fitting and plotting models to tabular data.

Exceptions
----------
ezfit.exceptions.ColumnNotFoundError
    Raised if a required DataFrame column is missing during fitting or plotting.

Dependencies
------------
numpy, pandas, matplotlib, scipy, scikit-learn
Optionally: emcee, corner, arviz, tqdm (for advanced MCMC/use)

Example
-------
>>> import numpy as np
>>> import pandas as pd
>>> from ezfit.functions import power_law
>>> df = pd.DataFrame({"x": np.linspace(1, 10, 50)})
>>> df["y"] = power_law(df["x"].values, a=2.0, b=1.5) + np.random.normal(0, 1, 50)
>>> model, ax, _ = df.fit(power_law, x="x", y="y", plot=True)

See Also
--------
ezfit.model.Model
    Underlying model representation for evaluated functions and parameter management.
ezfit.optimizers
    Backend optimizer implementations.
ezfit.types
    Type hints for fit methods and fit keyword options.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

    from ezfit.types import (  # Import specific Kwargs types
        FitKwargs,
        FitMethod,
        FitResult,
    )


@dataclass
class FitData:
    """Container for validated fit data.

    This class encapsulates the validated and extracted data from a DataFrame
    for use in fitting operations.

    Attributes
    ----------
    xdata : np.ndarray
        Independent variable data as numpy array.
    ydata : np.ndarray
        Dependent variable data as numpy array.
    sigma : np.ndarray | None
        Error/uncertainty on dependent variable, or None if not provided.
    """

    xdata: np.ndarray
    ydata: np.ndarray
    sigma: np.ndarray | None

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        x: str,
        y: str,
        yerr: str | None = None,
    ) -> FitData:
        """Create FitData from DataFrame with validation.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        x : str
            Column name for independent variable.
        y : str
            Column name for dependent variable.
        yerr : str | None, optional
            Optional column name for errors on dependent variable, by default None.

        Returns
        -------
        FitData
            Validated FitData object containing extracted arrays.

        Raises
        ------
        ColumnNotFoundError
            If any required column is missing from the DataFrame.
        """
        columns_to_check = [x, y]
        if yerr:
            columns_to_check.append(yerr)

        for col in columns_to_check:
            if col not in df.columns:
                raise ColumnNotFoundError(col)

        xdata = df[x].to_numpy(dtype=float)
        ydata = df[y].to_numpy(dtype=float)
        sigma = df[yerr].to_numpy(dtype=float) if yerr else None

        return cls(xdata=xdata, ydata=ydata, sigma=sigma)


@dataclass
class PlotOptions:
    """Options for plotting fit results.

    This class groups all plotting-related parameters into a single object
    to reduce parameter list length and improve code organization.

    Attributes
    ----------
    residuals : Literal["none", "res", "percent", "rmse"]
        Type of residuals to plot, by default "res".
    color_error : str
        Color for data points/error bars, by default "C0".
    color_model : str
        Color for the fitted model line, by default "C3".
    color_residuals : str
        Color for the residuals plot, by default "C0".
    fmt_error : str
        Marker style for data points, by default ".".
    ls_model : str
        Line style for the model line, by default "-".
    ls_residuals : str
        Line style for residuals, by default "".
    marker_residuals : str
        Marker style for residuals, by default ".".
    err_kws : dict[str, Any] | None
        Additional keyword arguments for data/error bar plotting, by default None.
    mod_kws : dict[str, Any] | None
        Additional keyword arguments for model line plotting, by default None.
    res_kws : dict[str, Any] | None
        Additional keyword arguments for residuals plotting, by default None.
    """

    residuals: Literal["none", "res", "percent", "rmse"] = "res"
    color_error: str = "C0"
    color_model: str = "C3"
    color_residuals: str = "C0"
    fmt_error: str = "."
    ls_model: str = "-"
    ls_residuals: str = ""
    marker_residuals: str = "."
    err_kws: dict[str, Any] | None = field(default=None)
    mod_kws: dict[str, Any] | None = field(default=None)
    res_kws: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize default dictionaries if None."""
        if self.err_kws is None:
            self.err_kws = {}
        if self.mod_kws is None:
            self.mod_kws = {}
        if self.res_kws is None:
            self.res_kws = {}


@pd.api.extensions.register_dataframe_accessor("fit")
class FitAccessor:
    """Accessor for fitting data in a pandas DataFrame to a given model."""

    # Registry for fit methods
    _FIT_METHODS: ClassVar[dict[FitMethod, Callable[..., Any]]] = {
        "curve_fit": _fit_curve_fit,
        "minimize": _fit_minimize,
        "differential_evolution": _fit_differential_evolution,
        "shgo": _fit_shgo,
        "dual_annealing": _fit_dual_annealing,
        "emcee": _fit_emcee,
        "bayesian_ridge": _fit_bayesian_ridge,
        "ridge": _fit_ridge,
        "lasso": _fit_lasso,
        "elasticnet": _fit_elasticnet,
        "polynomial": _fit_polynomial,
    }

    # Methods that require sigma (yerr)
    _METHODS_REQUIRING_SIGMA: ClassVar[set[FitMethod]] = {
        "minimize",
        "differential_evolution",
        "shgo",
        "dual_annealing",
        "emcee",
    }

    # Type mapping for fit_kwargs casting
    _KWARGS_TYPE_MAP: ClassVar[dict[FitMethod, str]] = {
        "curve_fit": "CurveFitKwargs",
        "minimize": "MinimizeKwargs",
        "differential_evolution": "DifferentialEvolutionKwargs",
        "shgo": "ShgoKwargs",
        "dual_annealing": "DualAnnealingKwargs",
        "emcee": "EmceeKwargs",
        "bayesian_ridge": "BayesianRidgeKwargs",
    }

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the FitAccessor.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to fit data from.
        """
        self._df = df

    def __call__(
        self,
        model: Callable[..., Any],
        x: str,
        y: str,
        yerr: str | None = None,
        plot: bool = True,  # noqa: FBT001
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

        Parameters
        ----------
        model : Callable[..., Any]
            The model function to fit the data to.
        x : str
            The name of the column in the DataFrame for the independent variable.
        y : str
            The name of the column in the DataFrame for the dependent variable.
        yerr : str | None, optional
            The name of the column for the error on the dependent variable,
            by default None.
        plot : bool, optional
            Whether to plot the results, by default True.
        method : FitMethod, optional
            The fitting method to use, by default "curve_fit".
            Available methods: 'curve_fit', 'minimize', 'differential_evolution',
            'shgo', 'dual_annealing', 'emcee', 'bayesian_ridge', 'ridge', 'lasso',
            'elasticnet', 'polynomial'.
            'bayesian_ridge' requires scikit-learn and is only valid for linear models.
            'emcee' requires emcee.
            Methods other than 'curve_fit' and 'bayesian_ridge' may require
            sigma (yerr).
        fit_kwargs : FitKwargs | None, optional
            Keyword arguments passed to the fitting function
            (e.g., `scipy.optimize.curve_fit`, `scipy.optimize.minimize`, etc.),
            by default None.
        residuals : Literal["none", "res", "percent", "rmse"], optional
            The type of residuals to plot. Set to "none" to disable residuals plot,
            by default "res".
        color_error : str, optional
            Color for data points/error bars, by default "C0".
        color_model : str, optional
            Color for the fitted model line, by default "C3".
        color_residuals : str, optional
            Color for the residuals plot, by default "C0".
        fmt_error : str, optional
            Marker style for data points, by default ".".
        ls_model : str, optional
            Line style for the model line, by default "-".
        ls_residuals : str, optional
            Line style for residuals, by default "".
        marker_residuals : str, optional
            Marker style for residuals, by default ".".
        err_kws : dict[str, Any] | None, optional
            Additional keyword arguments for data/error bar plotting (`plt.errorbar`),
            by default None.
        mod_kws : dict[str, Any] | None, optional
            Additional keyword arguments for model line plotting (`plt.plot`),
            by default None.
        res_kws : dict[str, Any] | None, optional
            Additional keyword arguments for residuals plotting (`plt.plot`),
            by default None.
        **parameters : dict[str, Any]
            Specification of model parameters (initial values, bounds, fixed).
            Passed as keyword arguments, e.g., `param_name={"value": 1, "min": 0}`.

        Returns
        -------
        tuple[Model, Axes | None, Axes | None]
            A tuple containing the fitted Model object, the main plot Axes (or None),
            and the residuals plot Axes (or None).

        Raises
        ------
        ColumnNotFoundError
            If a specified column (x, y, yerr) is not found.
        ImportError
            If a required library (e.g., scikit-learn, emcee) is not installed
            for the chosen method.
        ValueError
            If an invalid method is chosen, if required arguments (like sigma)
            are missing for a method, or if the fit fails.
        TypeError
            If the model is not a callable or if the parameters are not a dictionary
            or if the parameters are not valid for the model.
        """
        fitted_model = self.fit(
            model=model,
            x=x,
            y=y,
            yerr=yerr,
            method=method,
            fit_kwargs=fit_kwargs,
            **parameters,
        )

        if not plot:
            return fitted_model, None, None

        plot_options = PlotOptions(
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

        axes = self.plot(
            x=x,
            y=y,
            model=fitted_model,
            yerr=yerr,
            plot_options=plot_options,
        )

        if isinstance(axes, tuple):
            return fitted_model, axes[0], axes[1]
        return fitted_model, axes, None

    def fit(
        self,
        model: Callable[..., Any],
        x: str,
        y: str,
        yerr: str | None = None,
        method: FitMethod = "curve_fit",
        fit_kwargs: FitKwargs | None = None,
        **parameters: dict[str, Any],
    ) -> Model:
        """Fit the data to the model.

        Parameters
        ----------
        model : Callable[..., Any]
            The model function to fit the data to.
        x : str
            The name of the column for the independent variable.
        y : str
            The name of the column for the dependent variable.
        yerr : str | None, optional
            The name of the column for the error on the dependent variable,
            by default None.
        method : FitMethod, optional
            The fitting method to use, by default "curve_fit".
            Available methods: 'curve_fit', 'minimize', 'differential_evolution',
            'shgo', 'dual_annealing', 'emcee', 'bayesian_ridge', 'ridge', 'lasso',
            'elasticnet', 'polynomial'.
            'bayesian_ridge' requires scikit-learn and is only valid for linear models.
            'emcee' requires emcee.
            Methods other than 'curve_fit' and 'bayesian_ridge' require sigma (yerr).
        fit_kwargs : FitKwargs | None, optional
            Keyword arguments passed to the underlying fitting function
            (e.g., `scipy.optimize.curve_fit`, `scipy.optimize.minimize`,
            `sklearn.linear_model.BayesianRidge`, `emcee.EnsembleSampler`),
            by default None.
        **parameters : dict[str, Any]
            Specification of model parameters (initial values, bounds, fixed).

        Returns
        -------
        Model
            The fitted Model object.

        Raises
        ------
        ColumnNotFoundError
            If a specified column (x, y, yerr) is not found.
        ImportError
            If a required library (e.g., scikit-learn, emcee) is not installed
            for the chosen method.
        ValueError
            If an invalid method is chosen, if required arguments (like sigma)
            are missing for a method, or if the fit fails.
        """
        fit_data = FitData.from_dataframe(self._df, x, y, yerr)
        # Model.__init__ handles conversion from dict[str, dict[str, Any]]
        # to dict[str, Parameter]
        model_obj = Model(func=model, params=parameters)  # type: ignore[arg-type]

        if fit_kwargs is None:
            fit_kwargs = {}

        self._validate_method_requirements(method, fit_data.sigma)

        fit_result = self._execute_fit_method(
            method=method,
            model=model_obj,
            fit_data=fit_data,
            fit_kwargs=fit_kwargs,
        )

        self._update_model_from_fit_result(model_obj, fit_result, fit_data)

        return model_obj

    def _validate_method_requirements(
        self, method: FitMethod, sigma: np.ndarray | None
    ) -> None:
        """Validate that method requirements are met.

        Parameters
        ----------
        method : FitMethod
            The fitting method to validate.
        sigma : np.ndarray | None
            The error array, or None if not provided.

        Raises
        ------
        ValueError
            If method requires sigma but it's not provided.
        """
        if method in self._METHODS_REQUIRING_SIGMA and sigma is None:
            msg = f"Method '{method}' requires 'yerr' (sigma)."
            raise ValueError(msg)

    def _execute_fit_method(
        self,
        method: FitMethod,
        model: Model,
        fit_data: FitData,
        fit_kwargs: FitKwargs,
    ) -> FitResult:
        """Execute the appropriate fit method.

        Parameters
        ----------
        method : FitMethod
            The fitting method to use.
        model : Model
            The Model object to fit.
        fit_data : FitData
            The validated fit data.
        fit_kwargs : FitKwargs
            Keyword arguments for the fit method.

        Returns
        -------
        FitResult
            The FitResult from the optimizer.

        Raises
        ------
        ValueError
            If method is not supported or fit fails.
        """
        if method not in self._FIT_METHODS:
            msg = f"Unsupported fitting method: {method}"
            raise ValueError(msg)

        fit_func = self._FIT_METHODS[method]

        try:
            if method in self._KWARGS_TYPE_MAP:
                # Type casting for methods with specific kwargs types
                fit_kwargs = cast("Any", fit_kwargs)
            else:
                fit_kwargs = fit_kwargs or {}

            fit_result = fit_func(
                model=model,
                xdata=fit_data.xdata,
                ydata=fit_data.ydata,
                sigma=fit_data.sigma,
                fit_kwargs=fit_kwargs,
            )
        except (RuntimeError, ValueError, ImportError) as e:
            msg = f"Fitting with method '{method}' failed: {e}"
            raise ValueError(msg) from e
        else:
            return fit_result

    def _update_model_from_fit_result(
        self, model_obj: Model, fit_result: FitResult, fit_data: FitData | None = None
    ) -> None:
        """Update model object with fit results.

        Parameters
        ----------
        model_obj : Model
            The Model object to update.
        fit_result : FitResult
            The FitResult from the optimizer.
        fit_data : FitData | None, optional
            The fit data used for fitting. If provided, x_bounds are stored.
            By default None.
        """
        model_obj.residuals = fit_result["residuals"]
        model_obj.𝜒2 = fit_result["chi2"]
        model_obj.r𝜒2 = fit_result["rchi2"]
        model_obj.cov = fit_result["pcov"]
        model_obj.cor = fit_result["cor"]
        model_obj.r_squared = fit_result.get("r_squared")
        model_obj.pearson_r = fit_result.get("pearson_r")
        model_obj.rmse = fit_result.get("rmse")
        model_obj.rmsd = fit_result.get("rmsd")
        model_obj.bic = fit_result.get("bic")
        model_obj.aic = fit_result.get("aic")
        # fit_result_details accepts the details field which can be various types
        model_obj.fit_result_details = cast("Any", fit_result.get("details"))
        model_obj.sampler_chain = fit_result.get("sampler_chain")

        # Store x_bounds if fit_data is provided
        if fit_data is not None:
            model_obj.x_min = float(np.min(fit_data.xdata))
            model_obj.x_max = float(np.max(fit_data.xdata))

        popt = fit_result["popt"]
        perr = fit_result["perr"]
        if model_obj.params is not None:
            for i, name in enumerate(model_obj.params):
                error = perr[i] if perr is not None and i < len(perr) else np.nan
                model_obj[name] = (popt[i], error)

    def plot(
        self,
        x: str,
        y: str,
        model: Model,
        yerr: str | None = None,
        ax: Axes | None = None,
        plot_options: PlotOptions | None = None,
        residuals: Literal["none", "res", "percent", "rmse"] | None = None,
        color_error: str | None = None,
        color_model: str | None = None,
        color_residuals: str | None = None,
        fmt_error: str | None = None,
        ls_model: str | None = None,
        ls_residuals: str | None = None,
        marker_residuals: str | None = None,
        err_kws: dict[str, Any] | None = None,
        mod_kws: dict[str, Any] | None = None,
        res_kws: dict[str, Any] | None = None,
    ) -> Axes | tuple[Axes, Axes]:
        """Plot the data, model, and residuals.

        Parameters
        ----------
        x : str
            The name of the column for the independent variable.
        y : str
            The name of the column for the dependent variable.
        model : Model
            The fitted Model object containing the function and parameters.
        yerr : str | None, optional
            The name of the column for the error on the dependent variable,
            by default None.
        ax : Axes | None, optional
            An existing Matplotlib Axes object to plot on. If None, a new figure/axes
            is created, by default None.
        plot_options : PlotOptions | None, optional
            PlotOptions object containing all plotting parameters. If provided,
            overrides individual plotting parameters, by default None.
        residuals : Literal["none", "res", "percent", "rmse"] | None, optional
            The type of residuals to plot. Set to "none" to disable residuals plot.
            Overrides plot_options if provided, by default None (defaults to "res").
        color_error : str | None, optional
            Color for data points/error bars. Overrides plot_options if provided,
            by default None.
        color_model : str | None, optional
            Color for the fitted model line. Overrides plot_options if provided,
            by default None.
        color_residuals : str | None, optional
            Color for the residuals plot. Overrides plot_options if provided,
            by default None.
        fmt_error : str | None, optional
            Marker style for data points. Overrides plot_options if provided,
            by default None.
        ls_model : str | None, optional
            Line style for the model line. Overrides plot_options if provided,
            by default None.
        ls_residuals : str | None, optional
            Line style for residuals. Overrides plot_options if provided,
            by default None.
        marker_residuals : str | None, optional
            Marker style for residuals. Overrides plot_options if provided,
            by default None.
        err_kws : dict[str, Any] | None, optional
            Additional keyword arguments for `plt.errorbar`.
            Overrides plot_options if provided, by default None.
        mod_kws : dict[str, Any] | None, optional
            Additional keyword arguments for model line `plt.plot`.
            Overrides plot_options if provided, by default None.
        res_kws : dict[str, Any] | None, optional
            Additional keyword arguments for residuals `plt.plot`.
            Overrides plot_options if provided,
            by default None.

        Returns
        -------
        Axes | tuple[Axes, Axes]
            The main plot Axes object, or a tuple of (main Axes, residuals Axes)
            if residuals are plotted.

        Raises
        ------
        ColumnNotFoundError
            If a specified column (x, y, yerr) is not found.
        ValueError
            If an invalid residuals metric is specified or model has no parameters.
        """
        if plot_options is None:
            plot_options = PlotOptions(
                residuals=residuals if residuals is not None else "res",
                color_error=color_error or "C0",
                color_model=color_model or "C3",
                color_residuals=color_residuals or "C0",
                fmt_error=fmt_error or ".",
                ls_model=ls_model or "-",
                ls_residuals=ls_residuals or "",
                marker_residuals=marker_residuals or ".",
                err_kws=err_kws,
                mod_kws=mod_kws,
                res_kws=res_kws,
            )
        else:
            if residuals is not None:
                plot_options.residuals = residuals
            if color_error is not None:
                plot_options.color_error = color_error
            if color_model is not None:
                plot_options.color_model = color_model
            if color_residuals is not None:
                plot_options.color_residuals = color_residuals
            if fmt_error is not None:
                plot_options.fmt_error = fmt_error
            if ls_model is not None:
                plot_options.ls_model = ls_model
            if ls_residuals is not None:
                plot_options.ls_residuals = ls_residuals
            if marker_residuals is not None:
                plot_options.marker_residuals = marker_residuals
            if err_kws is not None:
                plot_options.err_kws = err_kws
            if mod_kws is not None:
                plot_options.mod_kws = mod_kws
            if res_kws is not None:
                plot_options.res_kws = res_kws

        fit_data = FitData.from_dataframe(self._df, x, y, yerr)

        if model.params is None:
            msg = "Model has no parameters to evaluate."
            raise ValueError(msg)

        main_ax, res_ax = self._setup_axes(ax, plot_options.residuals)

        self._plot_data(main_ax, fit_data, plot_options)
        self._plot_model(main_ax, fit_data.xdata, model, plot_options)
        main_ax.set_ylabel(y)
        main_ax.legend()

        if plot_options.residuals != "none" and res_ax is not None:
            self._plot_residuals(res_ax, fit_data, model, plot_options, x)
            return main_ax, res_ax

        main_ax.set_xlabel(x)
        return main_ax

    def _setup_axes(
        self, ax: Axes | None, residuals: Literal["none", "res", "percent", "rmse"]
    ) -> tuple[Axes, Axes | None]:
        """Set up matplotlib axes for plotting.

        Parameters
        ----------
        ax : Axes | None
            Optional existing axes to plot on.
        residuals : Literal["none", "res", "percent", "rmse"]
            Type of residuals to plot.

        Returns
        -------
        tuple[Axes, Axes | None]
            Tuple of (main_ax, res_ax). res_ax is None if residuals are "none" or
            if ax is provided.
        """
        if ax is None:
            if residuals != "none":
                fig, (main_ax, res_ax) = plt.subplots(
                    2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
                )
                fig.subplots_adjust(hspace=0.05)
            else:
                _, main_ax = plt.subplots()
                res_ax = None
        else:
            main_ax = ax
            res_ax = None
            if residuals != "none":
                msg = (
                    "Residual plot cannot be automatically created when "
                    "providing a single Axes object."
                )
                warnings.warn(msg, stacklevel=3)

        return main_ax, res_ax

    def _plot_data(
        self, ax: Axes, fit_data: FitData, plot_options: PlotOptions
    ) -> None:
        """Plot the data points with error bars.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.
        fit_data : FitData
            The data to plot.
        plot_options : PlotOptions
            Plotting options.
        """
        ax.errorbar(
            fit_data.xdata,
            fit_data.ydata,
            yerr=fit_data.sigma,
            fmt=plot_options.fmt_error,
            color=plot_options.color_error,
            label="Data",
            **plot_options.err_kws,  # type: ignore[arg-type]
        )

    def _plot_model(
        self, ax: Axes, xdata: np.ndarray, model: Model, plot_options: PlotOptions
    ) -> None:
        """Plot the fitted model line.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.
        xdata : np.ndarray
            The x data points.
        model : Model
            The fitted model.
        plot_options : PlotOptions
            Plotting options.
        """
        # Use new Model.__call__() API with automatic bounds
        try:
            if model.x_min is not None and model.x_max is not None:
                x_smooth = np.linspace(model.x_min, model.x_max, 500)
                y_model = model(x_smooth)
            else:

                def _raise(exc):
                    raise exc

                msg = "x_bounds not available"
                _raise(ValueError(msg))
        except (ValueError, AttributeError):
            # Fallback to old method if bounds are not available
            x_smooth = np.linspace(xdata.min(), xdata.max(), 500)
            y_model = model(x_smooth)

        ax.plot(
            x_smooth,
            y_model,
            ls=plot_options.ls_model,
            color=plot_options.color_model,
            label="Model",
            **plot_options.mod_kws,  # type: ignore[arg-type]
        )

    def _plot_residuals(
        self,
        ax: Axes,
        fit_data: FitData,
        model: Model,
        plot_options: PlotOptions,
        x_label: str,
    ) -> None:
        """Plot residuals.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.
        fit_data : FitData
            The data used for fitting.
        model : Model
            The fitted model.
        plot_options : PlotOptions
            Plotting options.
        x_label : str
            Label for the x-axis.
        """
        y_model_at_data = model(fit_data.xdata)
        res_val = fit_data.ydata - y_model_at_data

        if plot_options.residuals == "res":
            if fit_data.sigma is not None:
                plot_res = res_val / fit_data.sigma
                res_ylabel = "Residuals\n($\\sigma$)"
            else:
                plot_res = res_val
                res_ylabel = "Residuals"
        elif plot_options.residuals == "percent":
            valid_idx = fit_data.ydata != 0
            plot_res = np.full_like(fit_data.ydata, np.nan)
            plot_res[valid_idx] = 100 * res_val[valid_idx] / fit_data.ydata[valid_idx]
            res_ylabel = "Residuals\n(%)"
        elif plot_options.residuals == "rmse":
            plot_res = np.sqrt(res_val**2)
            res_ylabel = "RMSE"
        else:
            msg = f"Invalid residuals type: {plot_options.residuals}"
            raise ValueError(msg)

        ax.plot(
            fit_data.xdata,
            plot_res,
            marker=plot_options.marker_residuals,
            ls=plot_options.ls_residuals,
            color=plot_options.color_residuals,
            **plot_options.res_kws,  # type: ignore[arg-type]
        )
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(res_ylabel)

        ax.yaxis.set_label_coords(-0.1, 0.5)
        fig = ax.get_figure()
        if fig is not None:
            fig.align_ylabels()
