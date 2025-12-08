Quick Start Guide
=================

This guide will get you up and running with ezfit in just a few minutes.

Basic Fitting
-------------

The simplest way to use ezfit is to fit a pandas DataFrame to a model function:

.. code-block:: python

   import pandas as pd
   import ezfit
   import matplotlib.pyplot as plt

   # Load your data
   df = pd.read_csv("data.csv")

   # Define a model function
   def line(x, m, b):
       """Linear model: y = m*x + b"""
       return m * x + b

   # Fit the data
   model, ax, ax_res = df.fit(line, "x", "y", "yerr")

   # Show the plot
   plt.show()

   # Print fit results
   print(model)

This will:
- Fit your data to the model
- Automatically plot the data, fit, and residuals
- Return a Model object with fit parameters and statistics

Specifying Initial Values and Bounds
-------------------------------------

You can provide initial values and bounds for parameters:

.. code-block:: python

   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       m={"value": 1.0, "min": 0, "max": 10},
       b={"value": 0.0, "min": -5, "max": 5}
   )

Choosing Different Fitting Methods
-----------------------------------

ezfit supports multiple optimization methods:

.. code-block:: python

   # Default: scipy.optimize.curve_fit
   model, ax, _ = df.fit(line, "x", "y", "yerr")

   # Use scikit-learn Ridge regression
   model, ax, _ = df.fit(line, "x", "y", method="ridge")

   # Use MCMC (requires emcee)
   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={"nwalkers": 50, "nsteps": 1000}
   )

Accessing Fit Results
---------------------

The Model object contains all fit information:

.. code-block:: python

   # Parameter values and errors
   print(model["m"])  # Parameter object
   print(model["m"].value)  # Fitted value
   print(model["m"].err)  # Uncertainty

   # Goodness of fit
   print(model.𝜒2)  # Chi-squared
   print(model.r𝜒2)  # Reduced chi-squared

   # Covariance and correlation matrices
   print(model.cov)
   print(model.cor)

Next Steps
----------

- Learn about :doc:`loading_data`
- Explore :doc:`defining_models`
- See :doc:`fitting_methods` for all available optimizers
- Check out the :doc:`../notebooks/beginner_fitting` tutorial
