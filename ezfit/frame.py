"""Module for fitting data in a pandas DataFrame to a given model."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class ColumnNotFoundError(Exception):
    def __init__(self, column):
        self.column = column
        self.message = f"Column '{column}' not found in DataFrame."


@dataclass
class Parameter:
    """Data class for a parameter and its bounds."""

    value: float = 0.0
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

    def __repr__(self):
        return f"(value={self.value} ± {self.err}, bounds=({self.min}, {self.max}))"


@dataclass
class Model:
    """Data class for a model function and its parameters."""

    func: callable
    params: dict[str:Parameter] | None = None
    residuals: np.ndarray | None = None
    𝜒2: float | None = None
    r𝜒2: float | None = None

    def __post_init__(self, params=None):
        """Generate a list of parameters from the function signature."""
        if self.params is None:
            self.params = {}

        for i, name in enumerate(self.func.__code__.co_varnames):
            if i == 0:
                continue

            self.params[name] = (
                Parameter()
                if name not in self.params
                else Parameter(**self.params[name])
            )

    def __call__(
        self, x, bounds=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Evaluate the model at the given x values."""
        nominal = self.func(x, *[param.value for param in self.params.values()])
        if not bounds:
            return nominal

        lower = self.func(
            x, *[param.value - param.err for param in self.params.values()]
        )
        upper = self.func(
            x, *[param.value + param.err for param in self.params.values()]
        )
        return nominal, lower, upper

    def __repr__(self):
        name = self.func.__name__
        params = "\n".join([f"{v} : {param}" for v, param in self.params.items()])
        return f"{name}:\n𝜒2: {self.𝜒2}\nreduced 𝜒2: {self.r𝜒2}\n{params}"


@pd.api.extensions.register_dataframe_accessor("fit")
class FitAccessor:
    """Fitting accessor for pandas DataFrames."""

    def __init__(self, df):
        self._df = df

    def __call__(
        self,
        model: callable,
        x: str,
        y: str,
        yerr: str | None = None,
        plot: bool = True,
        plot_kwargs: dict = {"data_kwargs": {}, "model_kwargs": {}},
        **parameters: dict[str, Parameter],
    ) -> Model | tuple[Model, plt.Axes | None]:
        model = self.fit(model, x, y, yerr, **parameters)
        if plot:
            ax = plt.gca()
            self.plot(x, y, model, yerr, ax, **plot_kwargs)
            plt.show()
            return model, ax
        return model

    def fit(
        self,
        model: callable,
        x: str,
        y: str,
        yerr: str | None = None,
        **parameters: dict[str, Parameter],
    ):
        """Fit the data in the DataFrame to the given model.

        Parameters
        ----------
        model : Model
            The model to fit the data to.
        x : str
            The column name of the independent variable.
        y : str
            The column name of the dependent variable.
        **kwargs
            Additional keyword arguments to pass to `curve_fit`.

        Returns
        -------
        Model
            The fitted model.
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
        p0 = [param.value for param in data_model.params.values()]
        bounds = (
            [param.min for param in data_model.params.values()],
            [param.max for param in data_model.params.values()],
        )

        popt, pcov, infodict, _, _ = curve_fit(
            data_model.func,
            xdata,
            ydata,
            p0=p0,
            sigma=yerr,
            bounds=bounds,
            absolute_sigma=True,
            full_output=True,
        )

        for i, name in enumerate(data_model.params):
            data_model.params[name].value = popt[i]
            data_model.params[name].err = np.round(np.sqrt(pcov[i, i]), 4)

        data_model.residuals = infodict["fvec"]
        data_model.𝜒2 = np.sum(data_model.residuals**2)
        dof = len(xdata) - len(popt)
        data_model.r𝜒2 = data_model.𝜒2 / dof

        return data_model

    def plot(
        self,
        x: str,
        y: str,
        model: Model,
        yerr: str = None,
        shaded_bounds: bool = True,
        ax=None,
        data_kwargs={},
        model_kwargs={},
    ):
        """Plot the data and the model on the given axis.

        Parameters
        ----------
        x : str
            The column name of the independent variable.
        y : str
            The column name of the dependent variable.
        model : Model
            The model to plot.
        ax : matplotlib.axes.Axes
            The axis to plot on.
        **kwargs
            Additional keyword arguments to pass to `plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the plot.
        """
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
        yerr = self._df[yerr].values if yerr is not None else None

        ax.errorbar(
            xdata, ydata, yerr=yerr, label=y, fmt=".", color="C0", **data_kwargs
        )

        if shaded_bounds:
            ymodel, ylow, yhigh = model(xdata, bounds=True)
            plt.plot(xdata, ymodel, color="C1", **model_kwargs)
            ax.fill_between(
                xdata, ylow, yhigh, alpha=0.5, label="Fit", color="C1", **model_kwargs
            )
        else:
            ymodel = model(xdata)
            ax.plot(xdata, ymodel, color="C1", **model_kwargs)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()

        return (
            ax,
            model,
        )
