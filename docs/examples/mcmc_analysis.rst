MCMC Analysis
=============

Complete MCMC workflow with diagnostics and visualization.

Code
----

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   import ezfit
   from ezfit.examples import generate_multi_peak_data

   # Generate data
   df = generate_multi_peak_data(n_points=200, seed=42)

   from ezfit import gaussian

   def two_peaks(x, A1, c1, w1, A2, c2, w2, B):
       return gaussian(x, A1, c1, w1) + gaussian(x, A2, c2, w2) + B

   # Step 1: Get initial guess
   model_init, _, _ = df.fit(
       two_peaks, "x", "y", "yerr",
       method="curve_fit",
       A1={"value": 7.0, "min": 0, "max": 15},
       c1={"value": 7.0, "min": 5, "max": 9},
       w1={"value": 2.0, "min": 0.5, "max": 5},
       A2={"value": 5.0, "min": 0, "max": 15},
       c2={"value": 12.0, "min": 10, "max": 14},
       w2={"value": 3.0, "min": 0.5, "max": 5},
       B={"value": 0.5, "min": 0, "max": 2}
   )

   # Step 2: Run MCMC
   model_mcmc, ax, ax_res = df.fit(
       two_peaks, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={"nwalkers": 50, "nsteps": 2000, "progress": True},
       A1={"value": model_init["A1"].value, "min": 0, "max": 15},
       c1={"value": model_init["c1"].value, "min": 5, "max": 9},
       w1={"value": model_init["w1"].value, "min": 0.5, "max": 5},
       A2={"value": model_init["A2"].value, "min": 0, "max": 15},
       c2={"value": model_init["c2"].value, "min": 10, "max": 14},
       w2={"value": model_init["w2"].value, "min": 0.5, "max": 5},
       B={"value": model_init["B"].value, "min": 0, "max": 2}
   )

   plt.show()

   # Step 3: Check diagnostics
   print(model_mcmc.summary())

   # Step 4: Visualize
   model_mcmc.plot_trace()
   plt.show()

   model_mcmc.plot_corner()
   plt.show()
