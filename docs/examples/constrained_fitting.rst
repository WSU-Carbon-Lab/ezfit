Constrained Fitting
===================

Example of fitting with parameter constraints.

Code
----

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   import ezfit
   from ezfit.examples import generate_multi_peak_data

   # Generate data with two peaks
   df = generate_multi_peak_data(n_points=200, seed=42)

   from ezfit import gaussian

   def two_peaks(x, A1, c1, w1, A2, c2, w2, B):
       return gaussian(x, A1, c1, w1) + gaussian(x, A2, c2, w2) + B

   # Fit with constraint: peak 1 narrower than peak 2
   model, ax, ax_res = df.fit(
       two_peaks, "x", "y", "yerr",
       method="minimize",
       fit_kwargs={"method": "SLSQP"},
       A1={"value": 7.0, "min": 0, "max": 15},
       c1={"value": 7.0, "min": 5, "max": 9},
       w1={"value": 2.0, "min": 0.5, "max": 5, "constraint": "w1 < w2"},
       A2={"value": 5.0, "min": 0, "max": 15},
       c2={"value": 12.0, "min": 10, "max": 14},
       w2={"value": 3.0, "min": 0.5, "max": 5},
       B={"value": 0.5, "min": 0, "max": 2}
   )

   plt.show()
   print(model)
   print(f"\\nConstraint satisfied: {model['w1'].value < model['w2'].value}")
