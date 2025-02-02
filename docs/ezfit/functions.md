# Functions

[Ezfit Index](../README.md#ezfit-index) / [EZFIT A Dead simple interface for fitting in python](./index.md#ezfit-a-dead-simple-interface-for-fitting-in-python) / Functions

> Auto-generated documentation for [ezfit.functions](../../ezfit/functions.py) module.

## exponential

[Show source in functions.py:23](../../ezfit/functions.py#L23)

Exponential function: y = a * exp(b * x)

#### Signature

```python
@njit(parallel=True, fastmath=True)
def exponential(x, a, b): ...
```



## gaussian

[Show source in functions.py:35](../../ezfit/functions.py#L35)

Gaussian with peak = 'amplitude' and FWHM = 'fwhm'.

Formula:
  G(x) = amplitude * exp[-4 ln(2) * ((x - center) / fwhm)^2]

At x=center, G = amplitude.
The half max occurs at |x-center| = fwhm/2.

#### Signature

```python
@njit(parallel=True, fastmath=True)
def gaussian(x, amplitude, center, fwhm): ...
```



## linear

[Show source in functions.py:102](../../ezfit/functions.py#L102)

Linear function: y = m*x + b

#### Signature

```python
@njit(parallel=True, fastmath=True)
def linear(x, m, b): ...
```



## lorentzian

[Show source in functions.py:55](../../ezfit/functions.py#L55)

Lorentzian with peak = 'amplitude' and FWHM = 'fwhm'.

L(x) = amplitude * [ (fwhm/2)^2 / ((x-center)^2 + (fwhm/2)^2 ) ]

At x=center, L = amplitude.
The half max occurs at |x-center| = fwhm/2.

#### Signature

```python
@njit(parallel=True, fastmath=True)
def lorentzian(x, amplitude, center, fwhm): ...
```



## power_law

[Show source in functions.py:11](../../ezfit/functions.py#L11)

Power law function: y = a * x^b

#### Signature

```python
@njit(parallel=True, fastmath=True)
def power_law(x, a, b): ...
```



## pseudo_voigt

[Show source in functions.py:75](../../ezfit/functions.py#L75)

Pseudo-Voigt model (peak-based):
    y = height * [ (1 - eta)*G + eta*L ]

where G and L have the same FWHM = 'fwhm' and both peak at 1.0
when we pass amplitude=1.

That is:
  G(x) = 1 * exp[-4 ln(2) * ((x-center)/fwhm)^2]
  L(x) = 1 * ((fwhm/2)^2 / ((x-center)^2 + (fwhm/2)^2))

Then the final amplitude is scaled by 'height'.

#### Signature

```python
@njit(parallel=True, fastmath=True)
def pseudo_voigt(x, height, center, fwhm, eta): ...
```
