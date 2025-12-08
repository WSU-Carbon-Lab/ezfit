"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------

project = "ezfit"
copyright = "2024, WSU Carbon Lab"
author = "Harlan D Heilman"
release = "0.5.2"

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_nb",  # For Jupyter notebook support
    "sphinx_design",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# MyST-NB settings
nb_execution_mode = "off"  # Don't execute notebooks during build
nb_execution_timeout = 600

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "ezfit Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # Include custom CSS
html_logo = None  # Add logo if available
html_favicon = None  # Add favicon if available

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066CC",
        "color-brand-content": "#0066CC",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4A9EFF",
        "color-brand-content": "#4A9EFF",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/WSU-Carbon-Lab/ezfit",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Options for autodoc -----------------------------------------------------

autodoc_mock_imports = ["numba", "emcee", "corner", "arviz"]
