Installation
============

ezfit can be installed using pip or uv.

Basic Installation
-------------------

.. code-block:: bash

   pip install ezfit

Or using uv:

.. code-block:: bash

   uv pip install ezfit

Dependencies
------------

ezfit requires:
- Python >= 3.10
- numpy >= 1.26.0
- pandas >= 2.2.2
- scipy >= 1.13.0
- matplotlib >= 3.10.0
- numba >= 0.60.0
- scikit-learn >= 1.3.0

Optional Dependencies
--------------------

For MCMC fitting and advanced diagnostics:

.. code-block:: bash

   pip install ezfit[mcmc]

This installs:
- emcee >= 3.1.0 (MCMC sampler)
- corner (corner plots)
- arviz >= 0.18.0 (MCMC diagnostics)

For Development
---------------

To install with development dependencies:

.. code-block:: bash

   git clone https://github.com/WSU-Carbon-Lab/ezfit.git
   cd ezfit
   pip install -e ".[dev]"

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import ezfit
   import pandas as pd
   import numpy as np

   # Create simple test data
   x = np.linspace(0, 10, 50)
   y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
   yerr = np.full_like(y, 0.5)

   df = pd.DataFrame({"x": x, "y": y, "yerr": yerr})

   def line(x, m, b):
       return m * x + b

   model, ax, _ = df.fit(line, "x", "y", "yerr")
   print("Installation successful!")
   print(model)
