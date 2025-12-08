MCMC Fitting
============

Monte Carlo Markov Chain (MCMC) methods provide a Bayesian approach to parameter estimation, giving you full posterior distributions rather than just point estimates.

Why Use MCMC?
-------------

Traditional least-squares fitting assumes:
- Parameters have Gaussian uncertainties
- The covariance matrix fully describes uncertainties
- The best-fit is a single "true" answer

MCMC provides:
- **Full posterior distributions** - not just means and covariances
- **Handles non-Gaussian uncertainties** - captures asymmetric errors
- **Shows parameter correlations** - visualizes relationships
- **Robust uncertainty quantification** - especially for complex models

When to Use MCMC
----------------

Use MCMC when:
- You need robust uncertainty estimates
- Parameter posteriors may be non-Gaussian
- You want to understand parameter correlations
- Traditional methods give unreliable uncertainties
- You have complex, multi-modal parameter spaces

Basic Usage
-----------

.. code-block:: python

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={
           "nwalkers": 50,    # Number of walkers (chains)
           "nsteps": 2000     # Number of steps per walker
       }
   )

Key Parameters
--------------

nwalkers
~~~~~~~~

Number of parallel chains (walkers). More walkers provide better sampling but are slower.

- **Too few**: Poor exploration of parameter space
- **Too many**: Unnecessary computation
- **Recommended**: 2-4x the number of parameters, minimum 20-50

nsteps
~~~~~~

Number of steps each walker takes. More steps = better statistics but slower.

- **Too few**: Chain may not converge
- **Recommended**: Start with 1000-2000, check diagnostics

Convergence Diagnostics
-----------------------

ezfit automatically computes convergence diagnostics:

.. code-block:: python

   print(model.summary())

This shows:
- **R-hat (Gelman-Rubin statistic)**: Should be < 1.1 for convergence
- **ESS (Effective Sample Size)**: Number of independent samples
- **Burn-in**: Automatically detected
- **Converged**: Boolean indicating convergence

Visualization
-------------

Trace Plots
~~~~~~~~~~~

Check chain convergence:

.. code-block:: python

   model.plot_trace()
   plt.show()

Good chains look like "hairy caterpillars" - well-mixed with no trends.

Corner Plots
~~~~~~~~~~~~

Visualize posterior distributions and correlations:

.. code-block:: python

   model.plot_corner()
   plt.show()

Shows:
- Marginal distributions for each parameter
- Parameter correlations (off-diagonal)
- 16th, 50th (median), and 84th percentiles

Accessing Samples
-----------------

Get posterior samples for custom analysis:

.. code-block:: python

   samples = model.get_posterior_samples()
   print(f"Shape: {samples.shape}")  # (n_samples, n_params)

   # Custom statistics
   param_samples = samples[:, 0]  # First parameter
   print(f"95% CI: {np.percentile(param_samples, [2.5, 97.5])}")

Best Practices
--------------

1. **Start with traditional fitting** to get good initial guesses
2. **Use sufficient walkers** (2-4x number of parameters)
3. **Check convergence** before trusting results
4. **Visualize chains** to ensure good mixing
5. **Use corner plots** to understand parameter relationships

Example Workflow
---------------

.. code-block:: python

   # Step 1: Get initial guess
   model_init, _, _ = df.fit(model_func, "x", "y", "yerr", method="curve_fit")

   # Step 2: Run MCMC from best-fit
   model_mcmc, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={"nwalkers": 50, "nsteps": 2000},
       # Use best-fit values as starting point
       param1={"value": model_init["param1"].value, "min": 0, "max": 10},
       param2={"value": model_init["param2"].value, "min": -5, "max": 5}
   )

   # Step 3: Check diagnostics
   print(model_mcmc.summary())

   # Step 4: Visualize
   model_mcmc.plot_trace()
   model_mcmc.plot_corner()
   plt.show()
