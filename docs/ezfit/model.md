# Model

[Ezfit Index](../README.md#ezfit-index) / [EZFIT A Dead simple interface for fitting in python](./index.md#ezfit-a-dead-simple-interface-for-fitting-in-python) / Model

> Auto-generated documentation for [ezfit.model](../../ezfit/model.py) module.

## Model

[Show source in model.py:48](../../ezfit/model.py#L48)

Data class for a model function and its parameters.

#### Signature

```python
class Model: ...
```

### Model().__call__

[Show source in model.py:74](../../ezfit/model.py#L74)

Evaluate the model at the given x values.

#### Signature

```python
def __call__(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray: ...
```

### Model().__post_init__

[Show source in model.py:59](../../ezfit/model.py#L59)

Generate a list of parameters from the function signature.

#### Signature

```python
def __post_init__(self, params=None): ...
```

### Model().bounds

[Show source in model.py:113](../../ezfit/model.py#L113)

Yield the model parameter bounds as a tuple of lists.

#### Signature

```python
def bounds(self) -> tuple[list[float], list[float]]: ...
```

### Model().kwargs

[Show source in model.py:120](../../ezfit/model.py#L120)

Return the model parameters as a dictionary.

#### Signature

```python
def kwargs(self) -> dict: ...
```

### Model().random

[Show source in model.py:124](../../ezfit/model.py#L124)

Returns a valid random value within the bounds.

#### Signature

```python
def random(self, x): ...
```

### Model().values

[Show source in model.py:109](../../ezfit/model.py#L109)

Yield the model parameters as a list.

#### Signature

```python
def values(self) -> list[float]: ...
```



## Parameter

[Show source in model.py:9](../../ezfit/model.py#L9)

Data class for a parameter and its bounds.

#### Signature

```python
class Parameter: ...
```

### Parameter().random

[Show source in model.py:41](../../ezfit/model.py#L41)

Returns a valid random value within the bounds.

#### Signature

```python
def random(self) -> float: ...
```



## rounded_values

[Show source in model.py:137](../../ezfit/model.py#L137)

Round the values and errors to n significant figures.

#### Signature

```python
def rounded_values(x, xerr, n): ...
```



## sig_fig_round

[Show source in model.py:130](../../ezfit/model.py#L130)

Round a number to n significant figures.

#### Signature

```python
def sig_fig_round(x, n): ...
```
