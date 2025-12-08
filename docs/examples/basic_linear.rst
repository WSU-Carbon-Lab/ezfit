Basic Linear Fitting
=====================

This example demonstrates the simplest use case: fitting a line to data.

Code
----

.. code-block:: python

   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import ezfit

   # Generate example data
   from ezfit.examples import generate_linear_data

   df = generate_linear_data(n_points=50, slope=2.0, intercept=1.0, seed=42)

   # Define model
   def line(x, m, b):
       return m * x + b

   # Fit
   model, ax, ax_res = df.fit(line, "x", "y", "yerr")

   plt.show()

   # Print results
   print(model)

Output
------

The fit produces a plot with data, model, and residuals, plus fit statistics.
