Defining Models
===============

In ezfit, you define your model as a Python function.

Function Signature
------------------

Your model function must have the independent variable as the first argument, followed by the parameters:

.. code-block:: python

   def model(x, param1, param2, param3, ...):
       # Model calculation
       return result

Simple Example
--------------

Linear model:

.. code-block:: python

   def line(x, m, b):
       """Linear model: y = m*x + b"""
       return m * x + b

Using Built-in Functions
-------------------------

ezfit provides optimized functions:

.. code-block:: python

   from ezfit import gaussian, exponential, power_law

   # Use directly
   model, ax, _ = df.fit(gaussian, "x", "y", "yerr", ...)

   # Or combine
   def two_peaks(x, A1, c1, w1, A2, c2, w2):
       return gaussian(x, A1, c1, w1) + gaussian(x, A2, c2, w2)

Complex Models
--------------

You can define any mathematical relationship:

.. code-block:: python

   def complex_model(x, A, B, C, D):
       """Custom model"""
       return A * np.sin(x / B) * np.exp(-x / C) + D

Best Practices
--------------

1. **Use descriptive parameter names**
2. **Include docstrings** explaining the model
3. **Use NumPy operations** for vectorization
4. **Keep it simple** - complex models are harder to fit
