# ezfit Documentation Site Map

## Overview

This document provides a comprehensive map of the ezfit documentation structure, built from Sphinx source files.

## Documentation Structure

### Main Pages

- **index.html** - Main documentation landing page with quick start guide
- **README.html** - Project README documentation
- **genindex.html** - General index of all documentation items
- **py-modindex.html** - Python module index
- **search.html** - Documentation search page

### User Guide (`user_guide/`)

Comprehensive guides for using ezfit:

1. **installation.html** - Installation instructions and dependencies
2. **quickstart.html** - Quick start guide for new users
3. **loading_data.html** - How to load and prepare data for fitting
4. **defining_models.html** - Guide to defining model functions
5. **fitting_methods.html** - Overview of available fitting methods
   - curve_fit (default, Levenberg-Marquardt)
   - minimize (scipy.optimize.minimize with various algorithms)
   - differential_evolution (global optimizer)
   - emcee (MCMC Bayesian fitting)
   - sklearn methods (ridge, lasso, elasticnet, polynomial)
6. **constraints.html** - Parameter constraints and relationships
7. **mcmc.html** - MCMC fitting guide and diagnostics
8. **visualization.html** - Plotting and visualization tools

### Tutorials (`notebooks/`)

Interactive Jupyter notebook tutorials:

1. **beginner_fitting.html** - Basic fitting tutorial

   - Loading data
   - Defining simple models (line, exponential decay)
   - Fitting data with df.fit()
   - Interpreting results
   - Parameter bounds and initial values

2. **intermediate_fitting.html** - Advanced optimization methods

   - Different optimization methods (curve_fit, minimize, differential_evolution)
   - When to use each method
   - Handling complex data (Gaussian peaks, rugged surfaces)
   - Importance of initial guesses

3. **advanced_fitting.html** - MCMC and constraints
   - MCMC fitting with emcee
   - MCMC diagnostics (R-hat, ESS, trace plots, corner plots)
   - Posterior sample analysis
   - Parameter constraints (string-based and function-based)
   - Comparing uncertainty estimates

### API Reference (`api/`)

Complete API documentation:

1. **fit.html** - FitAccessor class

   - `df.fit()` - Main fitting method
   - `df.fit.plot()` - Plotting method
   - Available fitting methods and parameters

2. **model.html** - Model and Parameter classes

   - Model class for storing fit results
   - Parameter class for parameter specification
   - Model evaluation and parameter access

3. **functions.html** - Built-in model functions

   - `linear(x, m, b)` - Linear function [Wikipedia: Linear function](https://en.wikipedia.org/wiki/Linear_function)
   - `exponential(x, a, b)` - Exponential function [Wikipedia: Exponential function](https://en.wikipedia.org/wiki/Exponential_function)
   - `power_law(x, a, b)` - Power law [Wikipedia: Power law](https://en.wikipedia.org/wiki/Power_law)
   - `gaussian(x, amplitude, center, fwhm)` - Gaussian distribution [Wikipedia: Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
   - `lorentzian(x, amplitude, center, fwhm)` - Lorentzian distribution [Wikipedia: Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
   - `pseudo_voigt(x, height, center, fwhm, eta)` - Pseudo-Voigt profile [Wikipedia: Voigt profile](https://en.wikipedia.org/wiki/Voigt_profile)

4. **constraints.html** - Constraint functions

   - `less_than(param1, param2)` - Parameter inequality constraint
   - `greater_than(param1, param2)` - Parameter inequality constraint
   - `sum_less_than(params, value)` - Sum constraint
   - `sum_greater_than(params, value)` - Sum constraint
   - `product_equals(params, value)` - Product constraint
   - `parse_constraint_string()` - Parse string constraints

5. **optimizers.html** - Optimizer implementations

   - Internal optimizer functions for each method

6. **visualization.html** - Visualization functions
   - `plot_corner()` - Corner plots for MCMC
   - `plot_trace()` - Trace plots for MCMC
   - `plot_posterior()` - Posterior distribution plots
   - `plot_arviz_summary()` - ArviZ summary plots

### Examples (`examples/`)

Code examples demonstrating specific use cases:

1. **basic_linear.html** - Basic linear fitting example
2. **polynomial.html** - Polynomial fitting example
3. **gaussian.html** - Gaussian peak fitting example
4. **constrained_fitting.html** - Fitting with parameter constraints
5. **mcmc_analysis.html** - MCMC analysis example

### Source Code (`_modules/`)

Automatically generated source code documentation:

- `ezfit/constraints.html` - Constraints module source
- `ezfit/fit.html` - Fit module source
- `ezfit/model.html` - Model module source
- `ezfit/visualization.html` - Visualization module source

## Key Features Documented

### Fitting Methods

- **curve_fit**: Fast, good for well-behaved functions with good initial guesses
- **minimize**: Fast, supports various algorithms (L-BFGS-B, SLSQP, etc.)
- **differential_evolution**: Slow but excellent for rugged surfaces, global optimization
- **emcee**: Very slow but excellent, provides full posterior distributions
- **sklearn methods**: Fast, good for linear models and regularized regression

### Built-in Functions

All pre-built functions include:

- Complete NumPy-style docstrings
- Parameter descriptions
- Return value descriptions
- References to Wikipedia articles
- Usage examples

### Parameter Constraints

- String-based constraints (e.g., "w1 < w2")
- Function-based constraints
- Sum/product constraints
- Parameter relationships

### MCMC Features

- Convergence diagnostics (R-hat, ESS)
- Trace plots
- Corner plots
- Posterior sample access
- Burn-in detection

## Navigation Tips

1. **New Users**: Start with `user_guide/quickstart.html` or `notebooks/beginner_fitting.html`
2. **API Reference**: Use `api/` pages for detailed function documentation
3. **Examples**: Check `examples/` for specific use cases
4. **Search**: Use the search page to find specific topics or functions

## Build Information

- **Sphinx Version**: 8.1.3
- **Theme**: Furo
- **Extensions**:
  - myst_nb (for Jupyter notebook support)
  - sphinx.ext.autodoc
  - sphinx.ext.napoleon (NumPy docstrings)
  - sphinx.ext.intersphinx
  - sphinx_copybutton
  - sphinx_design

## Last Updated

Generated from Sphinx build on the current date.
