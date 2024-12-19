"""Mulitpeak fitting analysis module."""

import numpy as np
from scipy.signal import find_peaks


def peak_guess(signal: np.ndarray):
    peaks, _ = find_peaks(signal)
    return peaks


if __name__ == "__main__":
    signal = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    print(peak_guess(signal))
