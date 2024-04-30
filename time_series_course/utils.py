from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from torch import Tensor


def plot_series(
    time: Tensor,
    series: Tensor,
    format: str = "-",
    start: int = 0,
    end: int = None,
    label: Optional[str] = None,
    show: bool = True,
    use_new_figure: bool = True,
) -> None:
    """
    Visualizes time series data

    Args:
      time (array of int) - contains the time steps
      series (array of int) - contains the measurements for each time step
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      label (list of strings)- tag for the line
      show (bool) - whether to show the plot
      use_new_figure (bool) - whether to generate a new figure
    """

    if use_new_figure:
        plt.figure(figsize=(10, 6))

    # Plot the time series data
    plt.plot(time[start:end], series[start:end], format)

    # Label the x-axis
    plt.xlabel("Time")

    # Label the y-axis
    plt.ylabel("Value")

    if label:
        plt.legend(fontsize=14, labels=label)

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    if show:
        plt.show()
