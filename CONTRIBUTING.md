# Contributing to ezfit

Thank you for your interest in contributing to ezfit!

## Building Documentation Locally

To build and view the documentation locally:

1. Install documentation dependencies:

   ```bash
   pip install -e ".[docs]"
   ```

2. Build the documentation:

   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   Open `docs/_build/html/index.html` in your browser.

## Documentation Structure

- `docs/conf.py` - Sphinx configuration
- `docs/index.rst` - Main documentation index
- `docs/user_guide/` - User guide documentation
- `docs/api/` - API reference (auto-generated from docstrings)
- `docs/notebooks/` - Tutorial notebooks
- `docs/examples/` - Code examples
- `notebooks/` - Jupyter notebook tutorials

## Adding Documentation

### User Guide

Add new RST files to `docs/user_guide/` and update `docs/index.rst`.

### API Documentation

API docs are auto-generated from docstrings. Ensure your docstrings follow NumPy style.

### Notebooks

Add notebooks to `notebooks/` and reference them in `docs/notebooks/`.

## Code Style

- Follow PEP 8
- Use NumPy-style docstrings
- Run `ruff check` and `ruff format` before committing
- Type hints are required for all functions
