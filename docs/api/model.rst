Model and Parameter
===================

The ``Model`` and ``Parameter`` classes are the core data structures in ezfit.

Model
-----

.. autoclass:: ezfit.model.Model
   :members:
   :undoc-members:
   :show-inheritance:

Parameter
---------

.. autoclass:: ezfit.model.Parameter
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ezfit import Model, Parameter

   def line(x, m, b):
       return m * x + b

   # Create model with parameters
   model = Model(
       func=line,
       params={
           "m": Parameter(value=1.0, min=0, max=10),
           "b": Parameter(value=0.0, min=-5, max=5)
       }
   )

   # Evaluate model
   x = np.linspace(0, 10, 100)
   y = model(x)

Accessing Parameters
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After fitting
   print(model["m"].value)  # Parameter value
   print(model["m"].err)    # Parameter uncertainty
   print(model["m"].min)    # Lower bound
   print(model["m"].max)    # Upper bound

MCMC Visualization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After MCMC fit
   model.plot_corner()  # Corner plot
   model.plot_trace()   # Trace plots
   model.summary()      # Print diagnostics
