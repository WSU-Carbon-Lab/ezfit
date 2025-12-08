"""Visualization tools for MCMC chains and fit results.

This module provides functions for creating corner plots, trace plots,
and posterior distribution visualizations.
"""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    import corner
except ImportError:
    corner = None

try:
    import arviz as az
except ImportError:
    az = None


def plot_corner(
    chain: np.ndarray,
    param_names: list[str] | None = None,
    fig: "Figure | None" = None,
    **kwargs: dict[str, Any],
) -> tuple["Figure", np.ndarray]:
    """Create a corner plot from MCMC chain.

    Args:
        chain: MCMC chain array of shape (n_samples, n_params) or
               (n_walkers, n_steps, n_params).
        param_names: List of parameter names for labels. If None, uses "param0", "param1", etc.
        fig: Existing matplotlib Figure to plot on. If None, creates new figure.
        **kwargs: Additional keyword arguments passed to corner.corner().

    Returns
    -------
        Tuple of (figure, axes_array).

    Raises
    ------
        ImportError: If corner package is not installed.
    """
    if corner is None:
        msg = (
            "corner package is required for corner plots. "
            "Install with: pip install corner"
        )
        raise ImportError(msg)

    # Flatten chain if 3D
    if chain.ndim == 3:
        n_walkers, n_steps, n_params = chain.shape
        chain_flat = chain.reshape(-1, n_params)
    elif chain.ndim == 2:
        chain_flat = chain
        n_params = chain.shape[1]
    else:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    if param_names is None:
        param_names = [f"param{i}" for i in range(n_params)]

    if len(param_names) != n_params:
        msg = f"Number of parameter names ({len(param_names)}) must match number of parameters ({n_params})"
        raise ValueError(msg)

    # Default kwargs for corner plot
    corner_kwargs = {
        "labels": param_names,
        "show_titles": True,
        "title_kwargs": {"fontsize": 10},
        "plot_datapoints": False,
        "plot_density": True,
        "plot_contours": True,
    }
    corner_kwargs.update(kwargs)

    # Create corner plot
    fig = corner.corner(chain_flat, fig=fig, **corner_kwargs)  # type: ignore

    return fig, fig.axes  # type: ignore


def plot_trace(
    chain: np.ndarray,
    param_names: list[str] | None = None,
    fig: "Figure | None" = None,
    axes: "np.ndarray | None" = None,
) -> tuple["Figure", np.ndarray]:
    """Create trace plots for MCMC chain.

    Args:
        chain: MCMC chain array of shape (n_walkers, n_steps, n_params) or
               (n_samples, n_params).
        param_names: List of parameter names for labels. If None, uses "param0", "param1", etc.
        fig: Existing matplotlib Figure to plot on. If None, creates new figure.
        axes: Existing axes array to plot on. If None, creates new axes.

    Returns
    -------
        Tuple of (figure, axes_array).
    """
    # Handle different chain shapes
    if chain.ndim == 3:
        n_walkers, n_steps, n_params = chain.shape
        # Plot each walker separately
        plot_walkers = True
    elif chain.ndim == 2:
        n_samples, n_params = chain.shape
        n_steps = n_samples
        plot_walkers = False
    else:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    if param_names is None:
        param_names = [f"param{i}" for i in range(n_params)]

    if len(param_names) != n_params:
        msg = f"Number of parameter names ({len(param_names)}) must match number of parameters ({n_params})"
        raise ValueError(msg)

    # Create figure and axes if not provided
    if fig is None or axes is None:
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
        if n_params == 1:
            axes = np.array([axes])

    # Plot traces
    for i, (param_name, ax) in enumerate(zip(param_names, axes, strict=False)):
        if plot_walkers:
            # Plot each walker
            for walker_idx in range(n_walkers):
                ax.plot(chain[walker_idx, :, i], alpha=0.3, linewidth=0.5)
            # Plot mean across walkers
            mean_trace = np.mean(chain[:, :, i], axis=0)
            ax.plot(mean_trace, "k-", linewidth=1.5, label="Mean")
        else:
            ax.plot(chain[:, i], "k-", linewidth=1.0)

        ax.set_ylabel(param_name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")

    if plot_walkers:
        axes[0].legend()

    plt.tight_layout()

    return fig, axes


def plot_posterior(
    chain: np.ndarray,
    param_names: list[str] | None = None,
    fig: "Figure | None" = None,
    axes: "np.ndarray | None" = None,
    bins: int = 50,
) -> tuple["Figure", np.ndarray]:
    """Create posterior distribution plots for MCMC chain.

    Args:
        chain: MCMC chain array of shape (n_samples, n_params) or
               (n_walkers, n_steps, n_params).
        param_names: List of parameter names for labels. If None, uses "param0", "param1", etc.
        fig: Existing matplotlib Figure to plot on. If None, creates new figure.
        axes: Existing axes array to plot on. If None, creates new axes.
        bins: Number of bins for histograms.

    Returns
    -------
        Tuple of (figure, axes_array).
    """
    # Flatten chain if 3D
    if chain.ndim == 3:
        chain_flat = chain.reshape(-1, chain.shape[-1])
    elif chain.ndim == 2:
        chain_flat = chain
    else:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    n_params = chain_flat.shape[1]

    if param_names is None:
        param_names = [f"param{i}" for i in range(n_params)]

    if len(param_names) != n_params:
        msg = f"Number of parameter names ({len(param_names)}) must match number of parameters ({n_params})"
        raise ValueError(msg)

    # Create figure and axes if not provided
    if fig is None or axes is None:
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_params == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

    # Plot posterior for each parameter
    for i, (param_name, ax) in enumerate(zip(param_names, axes, strict=False)):
        param_samples = chain_flat[:, i]

        # Histogram
        ax.hist(param_samples, bins=bins, density=True, alpha=0.7, edgecolor="black")

        # Add percentiles
        percentiles = np.percentile(param_samples, [16, 50, 84])
        ax.axvline(percentiles[1], color="red", linestyle="--", linewidth=2, label="Median")
        ax.axvline(percentiles[0], color="red", linestyle=":", linewidth=1, label="16th/84th")
        ax.axvline(percentiles[2], color="red", linestyle=":", linewidth=1)

        ax.set_xlabel(param_name)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Hide extra axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    return fig, axes


def plot_arviz_summary(
    chain: np.ndarray,
    param_names: list[str] | None = None,
    **kwargs: dict[str, Any],
) -> "Figure | None":
    """Create arviz summary plots for MCMC chain.

    This function uses arviz for advanced MCMC diagnostics and visualization.

    Args:
        chain: MCMC chain array of shape (n_walkers, n_steps, n_params) or
               (n_samples, n_params).
        param_names: List of parameter names for labels. If None, uses "param0", "param1", etc.
        **kwargs: Additional keyword arguments passed to arviz plotting functions.

    Returns
    -------
        matplotlib Figure if arviz is available, None otherwise.
    """
    if az is None:
        return None

    # Convert chain to arviz format
    if chain.ndim == 3:
        # arviz expects (chain, draw, param) format
        # emcee provides (walker, step, param), which is compatible
        pass
    elif chain.ndim == 2:
        # Reshape to 3D: assume single chain
        chain = chain.reshape(1, *chain.shape)

    # Create InferenceData object
    try:
        idata = az.convert_to_inference_data(chain)  # type: ignore
        # Plot summary
        axes = az.plot_trace(idata, **kwargs)  # type: ignore
        return axes.ravel()[0].figure  # type: ignore
    except Exception:
        return None
