.. ezfit documentation master file

Welcome to ezfit's documentation!
==================================

**ezfit** is a dead simple interface for fitting in Python, designed for people who are new not just to Python but to coding, fitting, and programmatically interacting with data.

.. note::

   If you have experience with Excel but need to fit data using least squares fitting, this is the tool for you!

Quick Start
-----------

.. code-block:: python

   import pandas as pd
   import ezfit

   # Load your data
   df = pd.read_csv("data.csv")

   # Define a simple model
   def line(x, m, b):
       return m * x + b

   # Fit the data
   model, ax, _ = df.fit(line, "x", "y", "yerr")

   # View results
   print(model)

Installation
------------

.. code-block:: bash

   pip install ezfit

Or with optional dependencies for MCMC:

.. code-block:: bash

   pip install ezfit[mcmc]

Features
--------

* **Simple API**: Fit dataframes directly with a clean, intuitive interface
* **Multiple Optimizers**: Choose from scipy optimizers, scikit-learn methods, or MCMC
* **Parameter Constraints**: Specify parameter relationships and bounds easily
* **MCMC Diagnostics**: Built-in convergence diagnostics and visualization
* **Beautiful Plots**: Automatic plotting with residuals and model visualization

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/loading_data
   user_guide/defining_models
   user_guide/fitting_methods
   user_guide/constraints
   user_guide/mcmc
   user_guide/visualization

Tutorials
---------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks/beginner_fitting
   notebooks/intermediate_fitting
   notebooks/advanced_fitting

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/fit
   api/model
   api/functions
   api/constraints
   api/visualization
   api/optimizers

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_linear
   examples/polynomial
   examples/gaussian
   examples/constrained_fitting
   examples/mcmc_analysis

Reference
---------

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/optimizers

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
