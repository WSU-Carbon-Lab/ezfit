Parameter Constraints
=====================

ezfit allows you to specify relationships between parameters using constraints.

Why Use Constraints?
--------------------

Constraints are useful when:
- Parameters must satisfy physical relationships
- You need to enforce parameter ordering
- Parameter combinations must be limited
- You want to incorporate prior knowledge

Types of Constraints
--------------------

Simple Comparisons
~~~~~~~~~~~~~~~~~~

You can specify that one parameter must be less than or greater than another:

.. code-block:: python

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       param1={"value": 1.0, "constraint": "param1 < param2"},
       param2={"value": 2.0}
   )

Sum Constraints
~~~~~~~~~~~~~~~

Constrain the sum of multiple parameters:

.. code-block:: python

   from ezfit import sum_less_than

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       A1={"value": 5.0, "constraint": sum_less_than(["A1", "A2"], 10.0)},
       A2={"value": 4.0}
   )

Custom Function Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex relationships, use lambda functions:

.. code-block:: python

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       param1={
           "value": 1.0,
           "constraint": lambda p: p["param1"]**2 + p["param2"]**2 < 1.0
       },
       param2={"value": 0.5}
   )

Constraint Syntax
-----------------

String Constraints
~~~~~~~~~~~~~~~~~~

Simple string constraints support:
- ``"param1 < param2"``
- ``"param1 > param2"``
- ``"sum(param1, param2) < value"``

Function Constraints
~~~~~~~~~~~~~~~~~~~~

Available helper functions:
- ``less_than(param1, param2)``
- ``greater_than(param1, param2)``
- ``sum_less_than(params_list, value)``
- ``sum_greater_than(params_list, value)``
- ``product_equals(params_list, value)``

Lambda Constraints
~~~~~~~~~~~~~~~~~~

For maximum flexibility:

.. code-block:: python

   constraint = lambda p: p["A"] + p["B"] * p["C"] < 10.0

Supported Methods
-----------------

Constraints work with:
- ``minimize`` (with SLSQP method)
- ``differential_evolution``
- ``emcee`` (MCMC)

Note: ``curve_fit`` does not support general constraints, only bounds.

Examples
--------

Example 1: Peak Ordering
~~~~~~~~~~~~~~~~~~~~~~~~

Ensure peak 1 is narrower than peak 2:

.. code-block:: python

   model, ax, _ = df.fit(
       two_peaks, "x", "y", "yerr",
       w1={"value": 2.0, "constraint": "w1 < w2"},
       w2={"value": 3.0}
   )

Example 2: Physical Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a sum of exponentials, ensure decay rates are ordered:

.. code-block:: python

   model, ax, _ = df.fit(
       double_exp, "x", "y", "yerr",
       lambda1={"value": 0.5, "constraint": "lambda1 < lambda2"},
       lambda2={"value": 1.0}
   )

Example 3: MCMC with Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constraints work seamlessly with MCMC:

.. code-block:: python

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       method="emcee",
       fit_kwargs={"nwalkers": 50, "nsteps": 2000},
       param1={"value": 1.0, "constraint": "param1 < param2"},
       param2={"value": 2.0}
   )
