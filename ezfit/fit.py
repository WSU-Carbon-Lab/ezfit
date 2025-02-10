"""DataFrame accessor for fitting data stored in a DataFrame."""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ezfit.exceptions import ColumnNotFoundError
from ezfit.model import Model, Parameter
from ezfit.types import FitKwargs


@pd.api.extensions.register_dataframe_accessor("fit")
class FitAccessor:
    """
    Accessor for fitting data in a pandas DataFrame to a given model.

    Usage:
    ```python

    df = pd.DataFrame({
        "x": np.linspace(0, 10, 100),
        "y": np.linspace(0, 10, 100) + np.random.normal(0, 1, 100),
        "yerr": 0.1 * np.ones(100)
    })

    model, ax, ax_res = df.fit(
        model=ezfit.linear,
        x="x",
        y="y",
        yerr="yerr",
        m={"value": 1, "min": 0, "max": 2},
        b={"value": 0, "min": -1, "max": 1},
    )
    ```
    """

    def __init__(self, df):
        self._df = df

    def __call__(
        self,
        model: callable,
        x: str,
        y: str,
        yerr: str = None,
        *,
        plot: bool = True,
        fit_kwargs: FitKwargs = None,
        residuals: Literal["none", "res", "percent", "rmse"] = "res",
        color_error: str = "C0",
        color_model: str = "C3",
        color_residuals: str = "C0",
        fmt_error: str = ".",
        ls_model: str = "-",
        ls_residuals: str = "",
        marker_residuals: str = ".",
        err_kws: dict = {},
        mod_kws: dict = {},
        res_kws: dict = {},
        **parameters: dict[str, Parameter],
    ) -> tuple[Model, plt.Axes | None, plt.Axes | None]:
        """
        Fit the data to the model and plot the results.

        Calls the `fit` and `plot` methods in sequence fitting the data to the model
        and plotting the results.

        Args:
            model (callable):
                The model function to fit the data to.
            x (str):
                The name of the column in the DataFrame to use as the independent
                variable.
            y (str):
                The name of the column in the DataFrame to use as the dependent
                variable.
            yerr (str, optional):
                The name of the column in the DataFrame to use as the
                error on the dependent variable.
            plot (bool, optional):
                Whether to plot the results.
                fit_kwargs (FitKwargs, optional): Keyword arguments to pass to
                `scipy.optimize.curve_fit`.
            residuals (Literal["none", "res", "percent", "rmse"], optional):
                The type of residuals to plot.
            color_error (str, optional):
                The color of the error bars.
            color_model (str, optional):
                The color of the model line.
            color_residuals (str, optional):
                The color of the residuals.
            fmt_error (str, optional):
                The marker style for the error bars.
            ls_model (str, optional):
                The line style for the model line.
            ls_residuals (str, optional):
                The line style for the residuals.
            marker_residuals (str, optional):
                The marker style for the residuals.
            err_kws (dict, optional):
                Additional keyword arguments for errorbar plotting.
            mod_kws (dict, optional):
                Additional keyword arguments for model line plotting.
            res_kws (dict, optional):
                Additional keyword arguments for residuals plotting.
            parameters (dict[str, Parameter]):
                Specification of the model parameters, their initial values, and bounds.

        Returns:
            tuple[Model, plt.Axes | None, plt.Axes | None]:
                The fitted model and the axes objects for the main plot and residuals
                plot.

        Raises:
            ColumnNotFoundError:
                If the specified column is not found in the DataFrame.
        """
        model = self.fit(model, x, y, yerr, fit_kwargs=fit_kwargs, **parameters)
        if plot:
            ax = plt.gca()
            ax, ax_res = self.plot(
                x,
                y,
                model,
                yerr,
                ax=ax,
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
            return model, ax, ax_res
        return model

    def fit(
        self,
        model: callable,
        x: str,
        y: str,
        yerr: str | None = None,
        *,
        fit_kwargs: FitKwargs = None,
        **parameters: dict[str, Parameter],
    ):
        """
        Fit the data to the model.

        Args:
            model (callable):
                The model function to fit the data to.
            x (str):
                The name of the column in the DataFrame to use as the independent
                variable.
            y (str):
                The name of the column in the DataFrame to use as the dependent
                variable.
            yerr (str | None, optional):
                The name of the column in the DataFrame to use as the error on the
                dependent variable.
            fit_kwargs (FitKwargs, optional):
                Keyword arguments to pass to `scipy.optimize.curve_fit`.
            parameters (dict[str, Parameter]):
                Specification of the model parameters, their initial values, and bounds.

        Returns:
            Model:
                The fitted model.

        Raises:
            ColumnNotFoundError:
                If the specified column is not found in the DataFrame.
        """
        if x not in self._df.columns:
            raise ColumnNotFoundError(x)

        if y not in self._df.columns:
            raise ColumnNotFoundError(y)

        if yerr is not None and yerr not in self._df.columns:
            raise ColumnNotFoundError(yerr)

        xdata = self._df[x].values
        ydata = self._df[y].values
        yerr = self._df[yerr].values if yerr is not None else None

        data_model = Model(model, parameters)
        p0 = data_model.values()
        bounds = data_model.bounds()

        if fit_kwargs is None:
            fit_kwargs = {}

        popt, pcov, infodict, _, _ = curve_fit(
            data_model.func,
            xdata,
            ydata,
            p0=p0,
            sigma=yerr,
            bounds=bounds,
            absolute_sigma=True if yerr is not None else False,
            full_output=True,
            **fit_kwargs,
        )

        for i, (name, _) in enumerate(data_model):
            data_model[name] = popt[i], np.sqrt(pcov[i, i])

        data_model.residuals = infodict["fvec"]
        data_model.𝜒2 = np.sum(data_model.residuals**2)
        dof = len(xdata) - len(popt)
        data_model.r𝜒2 = data_model.𝜒2 / dof
        data_model.cov = pcov
        data_model.cor = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))

        return data_model

    def plot(
        self,
        x: str,
        y: str,
        model: Model,
        yerr: str = None,
        *,
        ax: plt.Axes = None,
        residuals: Literal["none", "res", "percent", "rmse"] = "res",
        color_error: str = "C0",
        color_model: str = "C3",
        color_residuals: str = "C0",
        fmt_error: str = ".",
        ls_model: str = "-",
        ls_residuals: str = "",
        marker_residuals: str = ".",
        err_kws: dict = {},
        mod_kws: dict = {},
        res_kws: dict = {},
    ) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
        """
        Plot the data, model, and residuals.

        Plot the data using matplotlib's `errorbar` function, the model using the
        defualt `plot` artist, and a difference metric between the data and model as
        a plot below the main plot. The difference metric can be the residuals, the
        percent difference, or the root mean squared error.

        Args:
            x (str):
                The name of the column in the DataFrame to use as the independent
                variable.
            y (str):
                The name of the column in the DataFrame to use as the dependent
                variable.
            model (callable):
                The model function to fit the data to.
            yerr (str, optional):
                The name of the column in the DataFrame to use as the
                error on the dependent variable.
            plot (bool, optional):
                Whether to plot the results.
                fit_kwargs (FitKwargs, optional): Keyword arguments to pass to
                `scipy.optimize.curve_fit`.
            residuals (Literal["none", "res", "percent", "rmse"], optional):
                The type of residuals to plot.
                - "none": Do not plot residuals.
                - "res": Plot the residuals.
                    residuals = (ydata - nominal) / yerr
                - "percent": Plot the percent difference.
                    percent = 100 * (ydata - nominal) / ydata
                - "rmse": Plot the root mean squared error.
                    rmse = sqrt((ydata - nominal)^2)
            color_error (str, optional):
                The color of the error bars.
            color_model (str, optional):
                The color of the model line.
            color_residuals (str, optional):
                The color of the residuals.
            fmt_error (str, optional):
                The marker style for the error bars.
            ls_model (str, optional):
                The line style for the model line.
            ls_residuals (str, optional):
                The line style for the residuals.
            marker_residuals (str, optional):
                The marker style for the residuals.
            err_kws (dict, optional):
                Additional keyword arguments for errorbar plotting.
            mod_kws (dict, optional):
                Additional keyword arguments for model line plotting.
            res_kws (dict, optional):
                Additional keyword arguments for residuals plotting.
            parameters (dict[str, Parameter]):
                Specification of the model parameters, their initial values, and bounds.

        Returns:
            plt.Axes | tuple[plt.Axes, plt.Axes]:
            The axes object for the main plot and residuals plot if `residuals` is not
            "none".

        Raises:
            ColumnNotFoundError:
            If the specified column is not found in the DataFrame. ValueError: If an
            invalid residuals metric is specified.
        """

        warnings.filterwarnings("ignore")

        if ax is None:
            ax = plt.gca()

        if x not in self._df.columns:
            raise ColumnNotFoundError(x)

        if y not in self._df.columns:
            raise ColumnNotFoundError(y)

        if yerr is not None and yerr not in self._df.columns:
            raise ColumnNotFoundError(yerr)

        xdata = self._df[x].values
        ydata = self._df[y].values
        yerr_values = self._df[yerr].values if yerr is not None else None
        nominal = model(xdata)

        err_kws = {
            "color": color_error,
            "fmt": fmt_error,
            "ms": 8,
            "zorder": 0,
            "alpha": 1,
            **err_kws,
        }
        mod_kws = {"c": color_model, "ls": ls_model, "zorder": 1, **mod_kws}
        res_kws = {
            "c": color_residuals,
            "ls": ls_residuals,
            "marker": marker_residuals,
            **res_kws,
        }

        ax.errorbar(
            xdata,
            ydata,
            yerr=yerr_values,
            label=y,
            **err_kws,
        )
        ax.plot(xdata, nominal, label=model.func.__name__, **mod_kws)

        if residuals == "none" or residuals is None:
            return ax

        ax_res = ax.inset_axes([0, -0.3, 1, 0.2])
        match residuals:
            case "res":
                err_metric = (
                    ydata - nominal
                    if yerr_values is None
                    else (ydata - nominal) / yerr_values
                )
                lines = [
                    0,
                    np.percentile(err_metric, 95),
                    np.percentile(err_metric, 99.6),
                    np.percentile(err_metric, 5),
                    np.percentile(err_metric, 0.4),
                ]
            case "percent":
                err_metric = 100 * (ydata - nominal) / ydata
                lines = [
                    0,
                    np.percentile(err_metric, 95),
                    np.percentile(err_metric, 99.6),
                    np.percentile(err_metric, 5),
                    np.percentile(err_metric, 0.4),
                ]
            case "rmse":
                err_metric = np.sqrt((ydata - nominal) ** 2)
                lines = [
                    0,
                    np.percentile(err_metric, 95),
                    np.percentile(err_metric, 99.6),
                ]
            case _:
                raise ValueError("Invalid residuals metric")
        ls = ["-", "--", ":", "--", ":"]
        for i, line in enumerate(lines):
            ax_res.axhline(line, color="grey", linestyle=ls[i], alpha=0.5)

        ax_res.plot(xdata, err_metric, **res_kws)

        ax.set_xlim(min(xdata), max(xdata))
        ax_res.set_xlim(min(xdata), max(xdata))
        ax_res.set_xlabel(x)
        ax_res.get_figure().tight_layout()
        ax_res.ticklabel_format(
            axis="y", style="sci", useMathText=True, scilimits=(0, 0)
        )
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_ylabel(y)
        ax.legend()
        return ax, ax_res


if __name__ == "__main__":
    from functions import linear

    df = pd.DataFrame(
        {
            "q": np.linspace(2e-5, 1e-3, 1000),
            "int": np.linspace(9.8, 9.4, 1000) + np.random.normal(0, 0.001, 1000),
            "int_err": 0.01 * np.ones(1000),
        }
    )

    model, ax, ax_res = df.query("0 < `q` < 10").fit(
        linear,
        "q",
        "int",
        "int_err",
        color_error="C2",
        m={"value": -400, "max": -300, "min": -500},
        err_kws={
            "fmt": "o",
            "ms": 1,
            "capsize": 2,
            "ecolor": "black",
            "errorevery": 10,
        },
    )
    ax.set_title("Linear fit")
    ax_res.set_title("Residuals")
    ax.figure.set_size_inches(12, 6)
    print(model)
    plt.show()
