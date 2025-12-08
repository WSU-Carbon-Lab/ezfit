Gaussian Peak Fitting
=====================

Fitting a Gaussian peak using the built-in function.

Code
----

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   import ezfit
   from ezfit.examples import generate_gaussian_data

   # Generate data
   df = generate_gaussian_data(
       n_points=100,
       amplitude=10.0,
       center=5.0,
       fwhm=2.0,
       seed=42
   )

   # Use built-in Gaussian function
   model, ax, ax_res = df.fit(
       ezfit.gaussian, "x", "y", "yerr",
       amplitude={"value": 9.0, "min": 0},
       center={"value": 5.0},
       fwhm={"value": 2.0, "min": 0}
   )

   plt.show()
   print(model)
