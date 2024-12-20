"""MultiPeak Fitting Analysis Module."""

from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional


@pd.api.extensions.register_dataframe_accessor("peak")
class PeakAccessor:
    def __init__(self, df):
        self._df = df

    def finder(
        self,
        x: str,
        y: str,
        num_peaks: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Find peaks in the data.

        Parameters
        ----------
        x : str
            The column name of the independent variable.
        y : str
            The column name of the dependent variable.
        number_of_peaks : int, optional
            The number of peaks to find. If None, all peaks are found.

        Returns
        -------
        DataFrame
            A DataFrame containing the peaks.
        """
        xlim, ylim, distance = self._setup_peak_finder(x, y, num_peaks, ax)
        peaks, info = find_peaks(
            self._df[y],
            height=ylim,
            distance=distance,
            wlen=distance,
        )
        print(info.keys())
        # filter the peaks by the axes limits
        peaks = peaks[(self._df[x][peaks] > xlim[0]) & (self._df[x][peaks] < xlim[1])][
            :num_peaks
        ]
        if ax is not None:
            # plot the peaks
            ax.plot(self._df[x][peaks], self._df[y][peaks], "x", **kwargs)

    def _setup_peak_finder(self, x, y, num_peaks, ax):
        if ax is not None:
            # get the axes limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        else:
            xlim = (self._df[x].min(), self._df[x].max())
            ylim = (self._df[y].min(), self._df[y].max())
        distance = (
            len(self._df) / (num_peaks + 1) / 2 if num_peaks is not None else None
        )
        return xlim, ylim, distance


if __name__ == "__main__":
    df = pd.read_csv("HW5data.txt", sep=r"\s+")
    ax = df.plot(x="qVal", y="intensity", loglog=True)
    ax.set_ylim(1e3, 1e5)
    ax.set_xlim(0.2, 2.5)
    df.peak.finder("qVal", "intensity", num_peaks=5, ax=ax)
    plt.show()
