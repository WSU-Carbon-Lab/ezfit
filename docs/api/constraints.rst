Parameter Constraints
======================

The constraints module provides utilities for specifying relationships between parameters.

.. automodule:: ezfit.constraints
   :no-members:
   :undoc-members:

Constraint Functions
--------------------

.. autofunction:: ezfit.constraints.less_than
   :no-index:

.. autofunction:: ezfit.constraints.greater_than
   :no-index:

.. autofunction:: ezfit.constraints.sum_less_than
   :no-index:

.. autofunction:: ezfit.constraints.sum_greater_than
   :no-index:

.. autofunction:: ezfit.constraints.product_equals
   :no-index:

.. autofunction:: ezfit.constraints.parse_constraint_string
   :no-index:

Examples
--------

String Constraints
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple string constraint
   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       param1={"value": 1.0, "constraint": "param1 < param2"},
       param2={"value": 2.0}
   )

Function Constraints
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ezfit import sum_less_than

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       A1={
           "value": 5.0,
           "constraint": sum_less_than(["A1", "A2"], 10.0)
       },
       A2={"value": 4.0}
   )

Lambda Constraints
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model, ax, _ = df.fit(
       model_func, "x", "y", "yerr",
       param1={
           "value": 1.0,
           "constraint": lambda p: p["param1"] + p["param2"] < 5.0
       },
       param2={"value": 2.0}
   )
