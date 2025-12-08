Fit Accessor
============

The ``FitAccessor`` is a pandas DataFrame accessor that provides a simple interface for fitting data.

.. autoclass:: ezfit.fit.FitAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import ezfit

   df = pd.read_csv("data.csv")

   def line(x, m, b):
       return m * x + b

   model, ax, ax_res = df.fit(line, "x", "y", "yerr")

With Parameter Bounds
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       m={"value": 1.0, "min": 0, "max": 10},
       b={"value": 0.0, "min": -5, "max": 5}
   )

Using Different Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Default: curve_fit
   model, ax, _ = df.fit(line, "x", "y", "yerr")

   # Use MCMC
   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={"nwalkers": 50, "nsteps": 1000}
   )

   # Use scikit-learn
   model, ax, _ = df.fit(line, "x", "y", method="ridge")
