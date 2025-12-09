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

def rugged_noise(y_true: np.ndarray, noise_level: float = 0.3, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add rugged, non-Gaussian noise to a model (mixture of Gaussian bulk and exponential outliers).

    Parameters
    ----------
    y_true : np.ndarray
        The true model values (shape [n_points]).
    noise_level : float
        Standard deviation for the Gaussian component (base noise level).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    y : np.ndarray
        Model with added rugged noise (same shape as y_true).
    y_true : np.ndarray
        The true model values (shape [n_points]).
    noise : np.ndarray
        The noise values (shape [n_points]).
    """
    if seed is not None:
        np.random.seed(seed)
    n_points = y_true.size

    n_gaussian = int(0.85 * n_points)  # 85% Gaussian noise
    n_outliers = n_points - n_gaussian  # 15% outliers

    gaussian_noise = np.random.normal(0, noise_level, n_gaussian)
    # Exponential noise for outliers (skewed distribution)
    outlier_noise = np.random.exponential(scale=2.0 * noise_level, size=n_outliers)
    outlier_noise *= np.random.choice([-1, 1], size=n_outliers)  # Random sign

    # Combine and shuffle
    noise = np.concatenate([gaussian_noise, outlier_noise])
    np.random.shuffle(noise)

    y = y_true + noise
    return y, y_true, noise

def generate_rugged_surface_data(
    n_points: int = 200,
    noise_level: float = 0.3,
    x_range: tuple[float, float] = (0, 20),
    seed: int | None = None,
    peaks: list[dict[str, float]] | None = None,
    small_errorbars: bool = False,
) -> pd.DataFrame:
    """Generate data with a rugged, multi-modal objective function surface.

    This creates data with multiple peaks on an exponential background with
    non-Gaussian experimental errors, making it extremely difficult to fit
    with simple optimizers. Demonstrates the need for global optimization
    methods like differential_evolution or MCMC.

    The function is: y = A*exp(-x/tau) + sum(peaks) + noise
    where peaks are Gaussian functions and noise has non-Gaussian distribution.

    Args:
        n_points: Number of data points to generate.
        noise_level: Base noise level (actual noise is non-Gaussian).
        x_range: Tuple of (x_min, x_max) for data range.
        seed: Random seed for reproducibility.
        peaks: List of peak dictionaries with 'amplitude', 'center', 'fwhm'.
            If None, uses default peaks.

    Returns
    -------
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    if peaks is None:
        peaks = [
            {"amplitude": 5.0, "center": 3.0, "fwhm": 2.5},
            {"amplitude": 4.0, "center": 7.0, "fwhm": 3.2},
            {"amplitude": 6.0, "center": 12.0, "fwhm": 4.0},
            {"amplitude": 3.5, "center": 16.0, "fwhm": 5.8},
        ]

    x = np.linspace(x_range[0], x_range[1], n_points)

    # Exponential background
    A_bg = 10
    tau = 8.0
    y_true = A_bg * np.exp(-x / tau)

    # Add multiple Gaussian peaks
    c = 4.0 * np.log(2.0)
    for peak in peaks:
        y_true += peak["amplitude"] * np.exp(
            -c * ((x - peak["center"]) / peak["fwhm"]) ** 2
        )

    y, y_true, noise = rugged_noise(y_true, noise_level, seed)

    # Add Gaussian background
    A_bg_gauss = np.mean(y_true)
    center_bg = np.mean(x)
    fwhm_bg = np.std(x)
    y_true += A_bg_gauss * np.exp(
        -((x - center_bg) / fwhm_bg) ** 2
    )

    # Add linear background based on the xrange to add a slight slope to the data
    B_bg = 2* (y_true[x_range[1]] - y_true[x_range[0]]) / (x_range[1] - x_range[0])
    C_bg = y_true[x_range[0]] - B_bg * x_range[0]
    y_true += B_bg * x + C_bg

    # Error bars: larger for outliers, smaller for normal points
    # Use absolute value of noise as base, with some variation
    yerr = noise_level * (1.0 + 0.5 * np.abs(noise) / noise_level)
    yerr = np.clip(yerr, 0.1 * noise_level, 5.0 * noise_level)
    # If the small_errorbars is True, underestimate the errorbars by a factor of 10
    if small_errorbars:
        yerr = yerr / 10.0

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

def step_edge(x, c, H):
    return H * (x > c)

def generate_rugged_step_edge_data(c, H, x_range, n_points, noise_level):
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = step_edge(x, c, H)
    y, _, noise = rugged_noise(y, noise_level)
    y_err = noise_level * (1.0 + 0.5 * np.abs(noise) / noise_level)
    return pd.DataFrame({'x': x, 'y': y, 'yerr': y_err})

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
