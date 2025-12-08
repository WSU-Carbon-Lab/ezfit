Fitting Methods
===============

ezfit supports multiple optimization methods, each suited for different scenarios.

Available Methods
-----------------

curve_fit (Default)
~~~~~~~~~~~~~~~~~~~~

The default method uses ``scipy.optimize.curve_fit``, which implements the Levenberg-Marquardt algorithm.

**Best for:**
- Simple, well-behaved functions
- When you have good initial guesses
- Fast fitting of smooth functions

**Example:**

.. code-block:: python

   model, ax, _ = df.fit(line, "x", "y", "yerr", method="curve_fit")

minimize
~~~~~~~~

Uses ``scipy.optimize.minimize`` with various algorithms (L-BFGS-B, SLSQP, etc.).

**Best for:**
- When you need specific optimization algorithms
- Constrained optimization (with SLSQP)
- Custom objective functions

**Example:**

.. code-block:: python

   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       method="minimize",
       fit_kwargs={"method": "L-BFGS-B"}
   )

differential_evolution
~~~~~~~~~~~~~~~~~~~~~~~

Global optimizer that searches the entire parameter space.

**Best for:**
- Functions with multiple local minima
- When initial guesses are poor
- Complex, rugged objective surfaces

**Example:**

.. code-block:: python

   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       method="differential_evolution",
       fit_kwargs={"maxiter": 1000, "seed": 42}
   )

emcee (MCMC)
~~~~~~~~~~~~

Markov Chain Monte Carlo sampling for Bayesian parameter estimation.

**Best for:**
- Quantifying parameter uncertainties
- Non-Gaussian posterior distributions
- Understanding parameter correlations
- Robust uncertainty estimation

**Example:**

.. code-block:: python

   model, ax, _ = df.fit(
       line, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={"nwalkers": 50, "nsteps": 2000}
   )

scikit-learn Methods
~~~~~~~~~~~~~~~~~~~~

Ridge, Lasso, ElasticNet, and Polynomial regression via scikit-learn.

**Best for:**
- Linear models
- Regularized regression
- Polynomial fitting

**Example:**

.. code-block:: python

   model, ax, _ = df.fit(line, "x", "y", method="ridge")
   model, ax, _ = df.fit(line, "x", "y", method="polynomial", fit_kwargs={"degree": 3})

Choosing the Right Method
-------------------------

Decision Tree
~~~~~~~~~~~~~

1. **Is your model linear in parameters?**
   - Yes → Consider ``ridge``, ``lasso``, or ``bayesian_ridge``
   - No → Continue

2. **Do you need full posterior distributions?**
   - Yes → Use ``emcee`` (MCMC)
   - No → Continue

3. **Does your function have multiple local minima?**
   - Yes → Use ``differential_evolution`` or ``dual_annealing``
   - No → Use ``curve_fit`` (default)

4. **Do you have good initial guesses?**
   - Yes → ``curve_fit`` or ``minimize``
   - No → Use global optimizer (``differential_evolution``)

Performance Comparison
----------------------

+----------------------+--------+----------+------------------+
| Method               | Speed  | Robust   | Uncertainty      |
+======================+========+==========+==================+
| curve_fit            | Fast   | Good     | Covariance       |
+----------------------+--------+----------+------------------+
| minimize             | Fast   | Good     | Hessian-based    |
+----------------------+--------+----------+------------------+
| differential_        | Slow   | Excellent| Approximate      |
| evolution            |        |          |                  |
+----------------------+--------+----------+------------------+
| emcee                | Very   | Excellent| Full posterior  |
|                      | Slow   |          |                  |
+----------------------+--------+----------+------------------+
| sklearn methods      | Fast   | Good     | Limited          |
+----------------------+--------+----------+------------------+
