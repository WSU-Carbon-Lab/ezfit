Visualization
=============

ezfit automatically creates plots, but you can customize them extensively.

Automatic Plotting
------------------

By default, ``df.fit()`` creates plots:

.. code-block:: python

   model, ax, ax_res = df.fit(line, "x", "y", "yerr", plot=True)

This shows:
- Data with error bars
- Fitted model line
- Residuals plot

Customizing Plots
-----------------

Control plot appearance:

.. code-block:: python

   model, ax, ax_res = df.fit(
       line, "x", "y", "yerr",
       color_error="blue",
       color_model="red",
       color_residuals="gray",
       fmt_error="o",
       ls_model="--",
       marker_residuals="."
   )

Residual Types
--------------

Choose residual visualization:

.. code-block:: python

   # Normalized residuals (default)
   model, ax, ax_res = df.fit(..., residuals="res")

   # Percent residuals
   model, ax, ax_res = df.fit(..., residuals="percent")

   # RMSE
   model, ax, ax_res = df.fit(..., residuals="rmse")

   # No residuals
   model, ax, _ = df.fit(..., residuals="none")

MCMC Visualization
------------------

After MCMC fits, use specialized plots:

.. code-block:: python

   # Corner plot (posterior distributions)
   model.plot_corner()

   # Trace plots (chain convergence)
   model.plot_trace()

   # Posterior distributions
   from ezfit.visualization import plot_posterior
   plot_posterior(model.sampler_chain, param_names=["A", "B", "C"])
