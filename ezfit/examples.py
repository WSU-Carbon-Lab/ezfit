"""Example data generation module for ezfit.

This module provides utilities for generating synthetic experimental data
with various levels of complexity for demonstrating different fitting
scenarios. These functions are designed for tutorials, documentation examples,
and testing fitting algorithms under controlled conditions.

The generated data includes:
- Simple linear and polynomial relationships
- Gaussian peaks and multi-peak spectra
- Exponential decays and oscillatory functions
- Complex rugged surfaces with non-Gaussian noise
- Step edge functions

All functions return pandas DataFrames with standardized column names ('x', 'y', 'yerr')
that are compatible with the ezfit fitting interface.
"""

import numpy as np
import pandas as pd

# ========= / Functional Generators / =========


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

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 50.
    slope : float, optional
        True slope of the line, by default 2.0.
    intercept : float, optional
        True y-intercept, by default 1.0.
    noise_level : float, optional
        Standard deviation of Gaussian noise, by default 0.5.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (0, 10).
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
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

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 50.
    coefficients : list[float] | None, optional
        Polynomial coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
        If None, uses [1, -2, 0.5] (quadratic), by default None.
    noise_level : float, optional
        Standard deviation of Gaussian noise, by default 0.5.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (-5, 5).
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
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

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 100.
    amplitude : float, optional
        Peak amplitude, by default 10.0.
    center : float, optional
        Peak center position, by default 5.0.
    fwhm : float, optional
        Full width at half maximum, by default 2.0.
    baseline : float, optional
        Baseline offset, by default 1.0.
    noise_level : float, optional
        Standard deviation of Gaussian noise, by default 0.3.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (0, 10).
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
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

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 200.
    peaks : list[dict[str, float]] | None, optional
        List of peak dictionaries with keys 'amplitude', 'center', 'fwhm'.
        If None, generates two overlapping peaks, by default None.
    baseline : float, optional
        Baseline offset, by default 0.5.
    noise_level : float, optional
        Standard deviation of Gaussian noise, by default 0.2.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (0, 20).
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
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


def rugged_noise(
    y_true: np.ndarray, noise_level: float = 0.3, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add rugged, non-Gaussian noise to a model.

    Creates a mixture of Gaussian noise (85%) and exponential outliers (15%)
    to simulate realistic experimental noise with occasional large deviations.

    Parameters
    ----------
    y_true : np.ndarray
        The true model values (shape [n_points]).
    noise_level : float, optional
        Standard deviation for the Gaussian component (base noise level),
        by default 0.3.
    seed : int | None, optional
        Random seed for reproducibility, by default None.

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
    *,
    small_errorbars: bool = False,
) -> pd.DataFrame:
    """Generate data with a rugged, multi-modal objective function surface.

    This creates data with multiple peaks on an exponential background with
    non-Gaussian experimental errors, making it extremely difficult to fit
    with simple optimizers. Demonstrates the need for global optimization
    methods like differential_evolution or MCMC.

    The function is: y = A*exp(-x/tau) + sum(peaks) + noise
    where peaks are Gaussian functions and noise has non-Gaussian distribution.

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 200.
    noise_level : float, optional
        Base noise level (actual noise is non-Gaussian), by default 0.3.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (0, 20).
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    peaks : list[dict[str, float]] | None, optional
        List of peak dictionaries with 'amplitude', 'center', 'fwhm'.
        If None, uses default peaks, by default None.
    small_errorbars : bool, optional
        If True, underestimates error bars by a factor of 10,
        by default False.

    Returns
    -------
    pd.DataFrame
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
    y_true += A_bg_gauss * np.exp(-(((x - center_bg) / fwhm_bg) ** 2))

    # Add linear background based on the xrange to add a slight slope to the data
    B_bg = 2 * (y_true[-1] - y_true[0]) / (x_range[1] - x_range[0])
    C_bg = y_true[0] - B_bg * x_range[0]
    y_true += B_bg * x + C_bg

    # Error bars: larger for outliers, smaller for normal points
    # Use absolute value of noise as base, with some variation
    yerr = noise_level * (1.0 + 0.5 * np.abs(noise) / noise_level)
    yerr = np.clip(yerr, 0.1 * noise_level, 5.0 * noise_level)
    # If the small_errorbars is True, underestimate the errorbars by a factor of 10
    if small_errorbars:
        yerr = yerr / 10.0

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def generate_exponential_data(
    n_points: int = 50,
    amplitude: float = 10.0,
    decay_rate: float = 0.5,
    baseline: float = 1.0,
    noise_level: float = 0.3,
    x_range: tuple[float, float] = (0, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic exponential decay data.

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 50.
    amplitude : float, optional
        Initial amplitude, by default 10.0.
    decay_rate : float, optional
        Decay rate (positive for decay), by default 0.5.
    baseline : float, optional
        Baseline offset, by default 1.0.
    noise_level : float, optional
        Standard deviation of Gaussian noise, by default 0.3.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (0, 10).
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)
    # Compute as a
    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = amplitude * np.exp(-decay_rate * x) + baseline
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    yerr = np.full_like(y, noise_level)

    return pd.DataFrame({"x": x, "y": y, "yerr": yerr})


def step_edge(x: np.ndarray, c: float, H: float) -> np.ndarray:
    """Step edge function.

    Mathematical definition: f(x) = H * (x > c)

    At x = c, the function reaches its maximum value of H.
    The half maximum occurs at ``|x-c| = H / 2``.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    c : float
        Threshold value.
    H : float
        Height of the step.

    Returns
    -------
    np.ndarray
        Array of the step edge function.
    """
    return H * (x > c)


def generate_rugged_step_edge_data(
    c: float,
    H: float,
    x_range: tuple[float, float],
    n_points: int,
    noise_level: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate data with a rugged, step edge function.

    Parameters
    ----------
    c : float
        Threshold value.
    H : float
        Height of the step.
    x_range : tuple[float, float]
        Range of the independent variable.
    n_points : int
        Number of data points to generate.
    noise_level : float
        Standard deviation of Gaussian noise.
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'x', 'y', and 'yerr'.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = step_edge(x, c, H)
    y, _, noise = rugged_noise(y, noise_level, seed)
    y_err = noise_level * (1.0 + 0.5 * np.abs(noise) / noise_level)
    return pd.DataFrame({"x": x, "y": y, "yerr": y_err})


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

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 100.
    amplitude : float, optional
        Oscillation amplitude, by default 5.0.
    frequency : float, optional
        Oscillation frequency, by default 2.0.
    phase : float, optional
        Phase offset, by default 0.0.
    decay : float, optional
        Exponential decay rate, by default 0.1.
    baseline : float, optional
        Baseline offset, by default 2.0.
    noise_level : float, optional
        Standard deviation of Gaussian noise, by default 0.4.
    x_range : tuple[float, float], optional
        Tuple of (x_min, x_max) for data range, by default (0, 10).
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    pd.DataFrame
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


# ========= / Grab Data From File / =========


def get_dataset_names() -> list[str]:
    """Return a list of the names of the pre generated datasets.

    Returns
    -------
    list[str]
        List of available dataset names (without .csv extension).
    """
    import importlib.resources
    from pathlib import Path

    from ezfit import data as data_module

    datasets: list[str] = []
    for f in importlib.resources.files(data_module).iterdir():
        f_path = Path(str(f))
        if f_path.suffix == ".csv":
            datasets.append(f_path.stem)
    return sorted(datasets)


def load_dataset(name: str) -> pd.DataFrame:
    """Load a pre generated dataset from the data folder.

    Parameters
    ----------
    name : str
        Name of the dataset to load. Available datasets:
        - 'current_voltage_data': Current-voltage measurement data
        - 'powerlaw': Power law relationship data

    Returns
    -------
    pd.DataFrame
        DataFrame with the dataset.

    Raises
    ------
    FileNotFoundError
        If the specified dataset name is not found.
    """
    import importlib.resources

    from ezfit import data as data_module

    filename = f"{name}.csv" if not name.endswith(".csv") else name

    try:
        with (
            importlib.resources.files(data_module)
            .joinpath(filename)
            .open(
                encoding="utf-8",
            ) as f
        ):
            return pd.read_csv(f)
    except FileNotFoundError as e:
        available = get_dataset_names()
        msg = f"Dataset '{name}' not found. Available datasets: {available}"
        raise FileNotFoundError(msg) from e
