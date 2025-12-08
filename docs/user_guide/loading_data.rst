Loading Data
============

ezfit works with pandas DataFrames, so loading data is straightforward.

CSV Files
---------

The most common format:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("data.csv")
   print(df.head())

Your CSV should have columns for:
- Independent variable (e.g., "x", "time", "energy")
- Dependent variable (e.g., "y", "intensity", "signal")
- Optional: Error on dependent variable (e.g., "yerr", "error", "sigma")

Tab-Delimited Files
-------------------

For space-separated data:

.. code-block:: python

   df = pd.read_csv("data.txt", sep=r"\s+", skiprows=1)

Skipping Rows
-------------

If your file has header information:

.. code-block:: python

   df = pd.read_csv("data.csv", skiprows=2)  # Skip first 2 rows

Data Cleaning
-------------

Remove bad data points:

.. code-block:: python

   # Remove points where x < 0
   df = df[df["x"] > 0]

   # Remove outliers
   df = df[df["y"] < 100]

Verify Your Data
----------------

Always plot your data first:

.. code-block:: python

   import matplotlib.pyplot as plt

   df.plot(x="x", y="y", yerr="yerr", fmt="o")
   plt.show()

Example Data
------------

ezfit includes utilities to generate example data for testing:

.. code-block:: python

   from ezfit.examples import generate_linear_data

   df = generate_linear_data(n_points=50, slope=2.0, intercept=1.0, seed=42)
