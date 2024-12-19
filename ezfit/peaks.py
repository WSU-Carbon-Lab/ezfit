from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gaussian(x, x0, sigma, A):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def chi2(y, y_fit, sigma = 1):
    return np.sum((y - y_fit) ** 2 / sigma ** 2)

def generate_guesses(data: pd.DataFrame, column: str,  plot=False, ax=None):
    """Generate initial guesses for peak fitting based on a gaussian model."""
    peak_guesses, info = find_peaks(data[column], width=1, height=[data[column].values[0], data[column].max()])
    heights = info["peak_heights"]
    centers = data.index[peak_guesses]
    widths = info["widths"]

    tol = 0.1
    heights = np.random.uniform(heights*(1-tol), heights*(1+tol), 10)
    centers = np.random.uniform(heights*(1-tol), heights*(1+tol), 10)
    widths = np.random.uniform(heights*(1-tol), heights*(1+tol), 10)
    # form a meshgrid of guesses
    guesses = np.meshgrid(centers, widths, heights)
    model = gaussian(data.index, *guesses)
    chi2_val = chi2(data[column], model)
    idx = np.argmin(chi2_val)
    best_guess = guesses[:, idx]
