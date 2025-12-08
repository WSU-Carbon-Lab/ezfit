Visualization Tools
===================

The visualization module provides tools for plotting MCMC chains and fit results.

.. automodule:: ezfit.visualization
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: ezfit.visualization.plot_corner

.. autofunction:: ezfit.visualization.plot_trace

.. autofunction:: ezfit.visualization.plot_posterior

.. autofunction:: ezfit.visualization.plot_arviz_summary

Examples
--------

Corner Plot
~~~~~~~~~~~

.. code-block:: python

   from ezfit.visualization import plot_corner

   # After MCMC fit
   fig, axes = plot_corner(
       model.sampler_chain,
       param_names=["A", "B", "C"]
   )
   plt.show()

Trace Plot
~~~~~~~~~~

.. code-block:: python

   from ezfit.visualization import plot_trace

   fig, axes = plot_trace(
       model.sampler_chain,
       param_names=["A", "B", "C"]
   )
   plt.show()

Using Model Methods
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convenience methods on Model object
   model.plot_corner()  # Corner plot
   model.plot_trace()   # Trace plots
