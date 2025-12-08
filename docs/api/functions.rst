Built-in Functions
==================

ezfit provides several commonly used model functions optimized with numba.

.. automodule:: ezfit.functions
   :members:
   :undoc-members:

Available Functions
-------------------

Linear
~~~~~~

.. autofunction:: ezfit.functions.linear

Exponential
~~~~~~~~~~~

.. autofunction:: ezfit.functions.exponential

Power Law
~~~~~~~~~

.. autofunction:: ezfit.functions.power_law

Gaussian
~~~~~~~~

.. autofunction:: ezfit.functions.gaussian

Lorentzian
~~~~~~~~~~

.. autofunction:: ezfit.functions.lorentzian

Pseudo-Voigt
~~~~~~~~~~~~

.. autofunction:: ezfit.functions.pseudo_voigt

Examples
--------

Using Built-in Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ezfit import gaussian
   import pandas as pd

   df = pd.read_csv("peak_data.csv")

   # Fit a Gaussian peak
   model, ax, _ = df.fit(
       gaussian, "x", "y", "yerr",
       amplitude={"value": 10.0, "min": 0},
       center={"value": 5.0},
       fwhm={"value": 2.0, "min": 0}
   )

Combining Functions
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ezfit import gaussian
   import numpy as np

   def two_peaks(x, A1, c1, w1, A2, c2, w2, B):
       """Sum of two Gaussians plus baseline"""
       return gaussian(x, A1, c1, w1) + gaussian(x, A2, c2, w2) + B

   model, ax, _ = df.fit(two_peaks, "x", "y", "yerr", ...)
