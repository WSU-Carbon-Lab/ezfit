# Building the Documentation

## Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or using uv:

```bash
uv pip install -e ".[docs]"
```

## Building

### HTML Documentation

```bash
cd docs
make html
```

The documentation will be built in `docs/_build/html/`.

### View Locally

Open `docs/_build/html/index.html` in your browser.

### Clean Build

```bash
make clean
make html
```

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `user_guide/` - User guide documentation
- `api/` - API reference documentation
- `notebooks/` - Tutorial notebooks
- `_static/` - Static files (CSS, images)

## Adding New Documentation

1. Add RST files to appropriate directory
2. Update `index.rst` to include new pages
3. Rebuild documentation

## Notebooks

The notebooks in `../notebooks/` are automatically included in the documentation via MyST-NB.
