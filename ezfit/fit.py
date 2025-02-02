"""Module for fitting data in a pandas DataFrame to a given model."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Generator, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class ColumnNotFoundError(Exception):
    def __init__(self, column):
        self.column = column
        self.message = f"Column '{column}' not found in DataFrame."


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


class FitKwargs(TypedDict):
    check_finite: bool
    method: Literal["lm", "trf", "dogbox"]
    jac: Callable | str | None
    nan_policy: Literal["raise", "omit"] | None


@pd.api.extensions.register_dataframe_accessor("fit")
class FitAccessor:
    """
    Accessor for fitting data in a pandas DataFrame to a given model.

    Usage:
    ```python
    import ezfit
    import pandas as pd

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

        Parameters
        ----------
        model : callable
            The model function to fit the data to. This function needs to take the form
            `def model(x, *params) -> np.ndarray` where `x` is the independent variable
            and `params` are the model parameters.
            ```python
            # Example model function
            def model(x, m, b):
                return m * x + b
            ```
        x : str
            The name of the column in the DataFrame to use as the independent variable.
        y : str
            The name of the column in the DataFrame to use as the dependent variable.
        yerr : str, optional
            The name of the column in the DataFrame to use as the error on the dependent
            variable, by default None.
        plot : bool, optional
            Whether to plot the results, by default True.
        fit_kwargs : FitKwargs, optional
            Keyword arguments to pass to `scipy.optimize.curve_fit`, by default None.
            Valid keys are:
                - `check_finite` : bool
                - `method` : str
                - `jac` : callable | str | None
            See the `scipy.optimize.curve_fit` documentation for more information.

        parameters : dict[str, Parameter]
            Spcification of the model parameters, their initial values, and bounds. This
            is passed as keyword arguments where the key is the parameter name and the
            value is a dictionary with keys `value`, `min`, and `max`.
            ```python
            # Example parameters
            m = {"value": 1, "min": 0, "max": 2}
            b = {"value": 0, "min": -1, "max": 1}
            ```


        Plotting parameters
        -------------------
        residuals : Literal[&quot;none&quot;, &quot;res&quot;, &quot;percent&quot;, &quot;rmse&quot;], optional
            The type of residuals to plot, by default "res;
        color_error : str, optional
            The color of the error bars, by default &quot;C0&quot;
        color_model : str, optional
            The color of the model line, by default &quot;C3&quot;
        color_residuals : str, optional
            The color of the residuals, by default &quot;C0&quot;
        fmt_error : str, optional
            The marker style for the error bars, by default &quot;.&quot;
        ls_model : str, optional
            The line style for the model line, by default &quot;-&quot;
        ls_residuals : str, optional
            The line style for the residuals, by default &quot;&quot;
        marker_residuals : str, optional
            The marker style for the residuals, by default &quot;.&quot;
        err_kws : dict, optional
            _description_, by default {}
        mod_kws : dict, optional
            _description_, by default {}
        res_kws : dict, optional
            _description_, by default {}

        Returns
        -------
        tuple[Model, plt.Axes | None, plt.Axes | None]
            The fitted model and the axes objects for the main plot and residuals plot.
            Usage:
            ```python
            model, ax, ax_res = df.fit(...)
            ```

        Raises
        ------
        ColumnNotFoundError
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

        Parameters
        ----------
        model : callable
            The model function to fit the data to. This function needs to take the form
            `def model(x, *params) -> np.ndarray` where `x` is the independent variable
            and `params` are the model parameters.
            ```python
            # Example model function
            def model(x, m, b):
                return m * x + b
            ```
        x : str
            The name of the column in the DataFrame to use as the independent variable.
        y : str
            The name of the column in the DataFrame to use as the dependent variable.
        yerr : str | None, optional
            The name of the column in the DataFrame to use as the error on the dependent
        fit_kwargs : FitKwargs, optional
            Keyword arguments to pass to `scipy.optimize.curve_fit`, by default None.

        Returns
        -------
        Model
            The fitted model.

        Raises
        ------
        ColumnNotFoundError
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
        yerr = (
            self._df[yerr].values if yerr is not None else None
        )  # 0.01 * np.ones_like(ydata)

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

        Parameters
        ----------
        x : str
            Column name for the independent variable.
        y : str
            Column name for the dependent variable.
        model : Model
            The fitted model.
        yerr : str, optional
            Column name for the error on the dependent variable, by default None.
        ax : plt.Axes, optional
            The axes object to plot on, by default None.
        residuals : Literal[&quot;none&quot;, &quot;res&quot;, &quot;percent&quot;, &quot;rmse&quot;], optional
            The type of residuals to plot, by default "res;
        color_error : str, optional
            The color of the error bars, by default &quot;C0&quot;
        color_model : str, optional
            The color of the model line, by default &quot;C3&quot;
        color_residuals : str, optional
            The color of the residuals, by default &quot;C0&quot;
        fmt_error : str, optional
            The marker style for the error bars, by default &quot;.&quot;
        ls_model : str, optional
            The line style for the model line, by default &quot;-&quot;
        ls_residuals : str, optional
            The line style for the residuals, by default &quot;&quot;
        marker_residuals : str, optional
            The marker style for the residuals, by default &quot;.&quot;
        err_kws : dict, optional
            keyword arguements for matplotlib.pyplot.errorbar, by default {}
        mod_kws : dict, optional
            keyword arguements for matplotlib.pyplot.plot, by default {}
        res_kws : dict, optional
            keyword arguements for matplotlib.pyplot.plot, by default {}

        Returns
        -------
        plt.Axes | tuple[plt.Axes, plt.Axes]
            The axes object for the main plot and residuals plot if `residuals` is not
            "none".

        Raises
        ------
        ColumnNotFoundError
            If the specified column is not found in the DataFrame.
        ValueError
            If an invalid residuals metric is specified.
        """
        import warnings

        warnings.filterwarnings("ignore")

        if ax is None:
            ax = plt.gca()

        if x not in self._df.columns:
            raise ColumnNotFoundError(x)

        if y not in self._df.columns:
            raise ColumnNotFoundError(y)

        if yerr is not None and yerr not in self._df.columns:
            raise ColumnNotFoundError(yerr)

        # Extract data
        xdata = self._df[x].values
        ydata = self._df[y].values
        yerr_values = self._df[yerr].values if yerr is not None else None
        nominal = model(xdata)
        # set default values for plotting
        err_kws = {
            "color": color_error,
            "fmt": fmt_error,
            "ms": 4,
            "zorder": 0,
            "alpha": 0.5,
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

        #  add residuals plotted on new axis below
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
        }
    )

    model, ax, ax_res = df.query("0 < `q` < 10").fit(
        linear,
        "q",
        "int",
        m={"value": -400, "max": -300, "min": -500},
    )
    ax.set_title("Linear fit")
    ax_res.set_title("Residuals")
    print(model)
    plt.show()
