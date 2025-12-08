"""Utilities for generating example data for tutorials and documentation.

This module provides functions to generate synthetic experimental data
with various levels of complexity for demonstrating different fitting
scenarios.
"""

import numpy as np
import pandas as pd


def generate_linear_data(
    n_points: int = 50,
    slope: float = 2.0,
    intercept: float = 1.0,
    noise_level: float = 0.5,
    x_range: tuple[float, float] = (0, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic linear data with noise.

    Perfect for beginner tutorials demonstrating basic least-squares fitting.

    Args:
        n_points: Number of data points to generate.
        slope: True slope of the line.
        intercept: True y-intercept.
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = slope * x + intercept
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_polynomial_data(
    n_points: int = 50,
    coefficients: list[float] | None = None,
    noise_level: float = 0.5,
    x_range: tuple[float, float] = (-5, 5),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic polynomial data with noise.

    Args:
        n_points: Number of data points to generate.
        coefficients: Polynomial coefficients [a0, a1, a2, ...]
            for a0 + a1*x + a2*x^2 + ...
                     If None, uses [1, -2, 0.5] (quadratic).
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    if coefficients is None:
        coefficients = [1.0, -2.0, 0.5]

    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = np.polyval(coefficients[::-1], x)  # polyval expects highest order first
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_gaussian_data(
    n_points: int = 100,
    amplitude: float = 10.0,
    center: float = 5.0,
    fwhm: float = 2.0,
    baseline: float = 1.0,
    noise_level: float = 0.3,
    x_range: tuple[float, float] = (0, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic Gaussian peak data with noise.

    Args:
        n_points: Number of data points to generate.
        amplitude: Peak amplitude.
        center: Peak center position.
        fwhm: Full width at half maximum.
        baseline: Baseline offset.
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    # Gaussian: A * exp(-4*ln(2)*((x-center)/fwhm)^2)
    c = 4.0 * np.log(2.0)
    y_true = amplitude * np.exp(-c * ((x - center) / fwhm) ** 2) + baseline
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_multi_peak_data(
    n_points: int = 200,
    peaks: list[dict[str, float]] | None = None,
    baseline: float = 0.5,
    noise_level: float = 0.2,
    x_range: tuple[float, float] = (0, 20),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic data with multiple Gaussian peaks.

    Useful for demonstrating complex fitting scenarios and MCMC.

    Args:
        n_points: Number of data points to generate.
        peaks: List of peak dictionaries with keys 'amplitude', 'center', 'fwhm'.
               If None, generates two overlapping peaks.
        baseline: Baseline offset.
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    if peaks is None:
        peaks = [
            {"amplitude": 8.0, "center": 7.0, "fwhm": 2.0},
            {"amplitude": 6.0, "center": 12.0, "fwhm": 3.0},
        ]

    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = np.full_like(x, baseline)

    c = 4.0 * np.log(2.0)
    for peak in peaks:
        y_true += peak["amplitude"] * np.exp(
            -c * ((x - peak["center"]) / peak["fwhm"]) ** 2
        )

    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_rugged_surface_data(
    n_points: int = 100,
    noise_level: float = 0.5,
    x_range: tuple[float, float] = (0, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate data with a rugged, multi-modal objective function surface.

    This creates data that is difficult to fit with simple optimizers,
    demonstrating the need for global optimization methods like
    differential_evolution or MCMC.

    The function is: y = sin(x) * exp(-x/5) + 0.5*sin(3*x) + noise

    Args:
        n_points: Number of data points to generate.
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    # Complex function with multiple local minima
    y_true = np.sin(x) * np.exp(-x / 5) + 0.5 * np.sin(3 * x) + 2.0
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_exponential_decay_data(
    n_points: int = 50,
    amplitude: float = 10.0,
    decay_rate: float = 0.5,
    baseline: float = 1.0,
    noise_level: float = 0.3,
    x_range: tuple[float, float] = (0, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic exponential decay data.

    Args:
        n_points: Number of data points to generate.
        amplitude: Initial amplitude.
        decay_rate: Decay rate (positive for decay).
        baseline: Baseline offset.
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = amplitude * np.exp(-decay_rate * x) + baseline
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_oscillatory_data(
    n_points: int = 100,
    amplitude: float = 5.0,
    frequency: float = 2.0,
    phase: float = 0.0,
    decay: float = 0.1,
    baseline: float = 2.0,
    noise_level: float = 0.4,
    x_range: tuple[float, float] = (0, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic damped oscillatory data.

    Useful for demonstrating fitting of periodic functions with decay.

    Args:
        n_points: Number of data points to generate.
        amplitude: Oscillation amplitude.
        frequency: Oscillation frequency.
        phase: Phase offset.
        decay: Exponential decay rate.
        baseline: Baseline offset.
        noise_level: Standard deviation of Gaussian noise.
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = (
        amplitude * np.exp(-decay * x) * np.sin(2 * np.pi * frequency * x + phase)
        + baseline
    )
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})
