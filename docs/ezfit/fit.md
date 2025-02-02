# Fit

[Ezfit Index](../README.md#ezfit-index) / [Ezfit](./index.md#ezfit) / Fit

> Auto-generated documentation for [ezfit.fit](../../ezfit/fit.py) module.

## ColumnNotFoundError

[Show source in fit.py:13](../../ezfit/fit.py#L13)

#### Signature

```python
class ColumnNotFoundError(Exception):
    def __init__(self, column): ...
```



## FitAccessor

[Show source in fit.py:164](../../ezfit/fit.py#L164)

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

#### Signature

```python
class FitAccessor:
    def __init__(self, df): ...
```

### FitAccessor().__call__

[Show source in fit.py:193](../../ezfit/fit.py#L193)

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

- [FitKwargs](#fitkwargs)
- [Model](#model)
- [Parameter](#parameter)

### FitAccessor().fit

[Show source in fit.py:320](../../ezfit/fit.py#L320)

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

- [FitKwargs](#fitkwargs)
- [Parameter](#parameter)

### FitAccessor().plot

[Show source in fit.py:409](../../ezfit/fit.py#L409)

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

- [Model](#model)



## FitKwargs

[Show source in fit.py:156](../../ezfit/fit.py#L156)

#### Signature

```python
class FitKwargs(TypedDict): ...
```



## Model

[Show source in fit.py:74](../../ezfit/fit.py#L74)

Data class for a model function and its parameters.

#### Signature

```python
class Model: ...
```

### Model().__call__

[Show source in fit.py:100](../../ezfit/fit.py#L100)

Evaluate the model at the given x values.

#### Signature

```python
def __call__(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray: ...
```

### Model().__post_init__

[Show source in fit.py:85](../../ezfit/fit.py#L85)

Generate a list of parameters from the function signature.

#### Signature

```python
def __post_init__(self, params=None): ...
```

### Model().bounds

[Show source in fit.py:139](../../ezfit/fit.py#L139)

Yield the model parameter bounds as a tuple of lists.

#### Signature

```python
def bounds(self) -> tuple[list[float], list[float]]: ...
```

### Model().kwargs

[Show source in fit.py:146](../../ezfit/fit.py#L146)

Return the model parameters as a dictionary.

#### Signature

```python
def kwargs(self) -> dict: ...
```

### Model().random

[Show source in fit.py:150](../../ezfit/fit.py#L150)

Returns a valid random value within the bounds.

#### Signature

```python
def random(self, x): ...
```

### Model().values

[Show source in fit.py:135](../../ezfit/fit.py#L135)

Yield the model parameters as a list.

#### Signature

```python
def values(self) -> list[float]: ...
```



## Parameter

[Show source in fit.py:35](../../ezfit/fit.py#L35)

Data class for a parameter and its bounds.

#### Signature

```python
class Parameter: ...
```

### Parameter().random

[Show source in fit.py:67](../../ezfit/fit.py#L67)

Returns a valid random value within the bounds.

#### Signature

```python
def random(self) -> float: ...
```



## rounded_values

[Show source in fit.py:26](../../ezfit/fit.py#L26)

Round the values and errors to n significant figures.

#### Signature

```python
def rounded_values(x, xerr, n): ...
```



## sig_fig_round

[Show source in fit.py:19](../../ezfit/fit.py#L19)

Round a number to n significant figures.

#### Signature

```python
def sig_fig_round(x, n): ...
```
