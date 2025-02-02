# Fit

[Ezfit Index](../README.md#ezfit-index) / [EZFIT A Dead simple interface for fitting in python](./index.md#ezfit-a-dead-simple-interface-for-fitting-in-python) / Fit

> Auto-generated documentation for [ezfit.fit](../../ezfit/fit.py) module.

## FitAccessor

[Show source in fit.py:17](../../ezfit/fit.py#L17)

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

#### Signature

```python
class FitAccessor:
    def __init__(self, df): ...
```

### FitAccessor().__call__

[Show source in fit.py:44](../../ezfit/fit.py#L44)

Fit the data to the model and plot the results.

Calls the [FitAccessor().fit](#fitaccessorfit) and [FitAccessor().plot](#fitaccessorplot) methods in sequence fitting the data to the model
and plotting the results.

#### Arguments

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
    - `fit_kwargs` *FitKwargs, optional* - Keyword arguments to pass to
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

#### Returns

tuple[Model, plt.Axes | None, plt.Axes | None]:
    The fitted model and the axes objects for the main plot and residuals
    plot.

#### Raises

ColumnNotFoundError:
    If the specified column is not found in the DataFrame.

#### Signature

```python
def __call__(
    self,
    model: callable,
    x: str,
    y: str,
    yerr: str = None,
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
    **parameters: dict[str, Parameter]
) -> tuple[Model, plt.Axes | None, plt.Axes | None]: ...
```

#### See also

- [FitKwargs](./types.md#fitkwargs)
- [Model](./model.md#model)
- [Parameter](./model.md#parameter)

### FitAccessor().fit

[Show source in fit.py:146](../../ezfit/fit.py#L146)

Fit the data to the model.

#### Arguments

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

#### Returns

Model:
    The fitted model.

#### Raises

ColumnNotFoundError:
    If the specified column is not found in the DataFrame.

#### Signature

```python
def fit(
    self,
    model: callable,
    x: str,
    y: str,
    yerr: str | None = None,
    fit_kwargs: FitKwargs = None,
    **parameters: dict[str, Parameter]
): ...
```

#### See also

- [FitKwargs](./types.md#fitkwargs)
- [Parameter](./model.md#parameter)

### FitAccessor().plot

[Show source in fit.py:228](../../ezfit/fit.py#L228)

Plot the data, model, and residuals.

Plot the data using matplotlib's `errorbar` function, the model using the
defualt `plot` artist, and a difference metric between the data and model as
a plot below the main plot. The difference metric can be the residuals, the
percent difference, or the root mean squared error.

#### Arguments

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
    - `fit_kwargs` *FitKwargs, optional* - Keyword arguments to pass to
    `scipy.optimize.curve_fit`.
residuals (Literal["none", "res", "percent", "rmse"], optional):
    The type of residuals to plot.
    - `-` *"none"* - Do not plot residuals.
    - `-` *"res"* - Plot the residuals.
        residuals = (ydata - nominal) / yerr
    - `-` *"percent"* - Plot the percent difference.
        percent = 100 * (ydata - nominal) / ydata
    - `-` *"rmse"* - Plot the root mean squared error.
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

#### Returns

plt.Axes | tuple[plt.Axes, plt.Axes]:
The axes object for the main plot and residuals plot if `residuals` is not
"none".

#### Raises

ColumnNotFoundError:
If the specified column is not found in the DataFrame. ValueError: If an
invalid residuals metric is specified.

#### Signature

```python
def plot(
    self,
    x: str,
    y: str,
    model: Model,
    yerr: str = None,
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
) -> plt.Axes | tuple[plt.Axes, plt.Axes]: ...
```

#### See also

- [Model](./model.md#model)
