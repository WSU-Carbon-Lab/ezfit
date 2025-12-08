Polynomial Fitting
==================

Using scikit-learn for polynomial regression.

Code
----

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   import ezfit
   from ezfit.examples import generate_polynomial_data

   # Generate polynomial data
   df = generate_polynomial_data(
       n_points=50,
       coefficients=[1.0, -2.0, 0.5],  # Quadratic
       seed=42
   )

   # Simple linear model (will be converted to polynomial features)
   def line(x, m, b):
       return m * x + b

   # Fit with polynomial method
   model, ax, ax_res = df.fit(
       line, "x", "y", "yerr",
       method="polynomial",
       fit_kwargs={"degree": 2}  # Quadratic
   )

   plt.show()
   print(model)
