from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module


# def naive_fcast(
#     times: Tensor, values: Tensor, prv_values: Tensor | None = None, **kwargs
# ) -> tuple[Tensor, ...]:
#     """Forecasting in a naive way using the previous value as forecast value

#     Args:
#         times (Tensor): tensor with the time data points
#         values (Tensor): tensor with the time series values

#     Returns:
#         tuple[Tensor, ...]: a 3-element tuple with the times,
#             the values and the forecasted values
#     """
#     if prv_values is None:
#         fcast_data = values
#     else:
#         fcast_data = torch.hstack((prv_values[-1], values))
#     return times, values, fcast_data[:-2]


def naive_fcast(series: Tensor, fcast_times: Tensor) -> tuple[Tensor, Tensor]:
    """Forecasting in a naive way using the previous value as forecast value

    Args:
        series (Tensor): tensor with the time series values
        fcast_times (Tensor): tensor with the time data points to forecast

    Returns:
        tuple[Tensor, ...]: a 2-element tuple with the forecast
            times and values
    """
    assert (
        fcast_times.size(0) < series.size(0) - 1
    ), f"Cannot forecast with given times: {fcast_times}"
    fcast_series = series[fcast_times - 1]
    return fcast_times, fcast_series


def mva_fcast(
    series: Tensor,
    fcast_times: Tensor,
    window_size: int = 20,
) -> tuple[Tensor, Tensor]:
    """Forecasting using simple moving average predictions

    Args:
        series (Tensor): the full time series
        fcast_times (Tensor): the times to perform the forecasting on
        window_size (int, optional): moving average window size

    Returns:
        tuple[Tensor, ...]: a 2-element tuple with the times
            and the forecasted values
    """
    assert (
        fcast_times.size(0) < series.size(0) - window_size
    ), f"Cannot forecast with given times: {fcast_times}"

    mva_times = fcast_times - window_size
    mva_series = list(map(torch.mean, [series[t : t + window_size] for t in mva_times]))
    return fcast_times, torch.tensor(mva_series).float()


def nn_fcast(
    series: Tensor,
    fcast_times: Tensor,
    model: Module,
    window_size: int = 20,
) -> tuple[Tensor, Tensor]:
    """Forecasting using a previously trained NN model

    Args:
        series (Tensor): the full time series
        fcast_times (Tensor): the times to perform the forecasting on
        model (Module): a trained model as a PyTorch module
        window_size (int, optional): moving average window size

    Returns:
        tuple[Tensor, ...]: a 2-element tuple with the times
            and the forecasted values
    """
    assert (
        fcast_times.size(0) < series.size(0) - window_size
    ), f"Cannot forecast with given times: {fcast_times}"

    times = fcast_times - window_size
    forecast = []
    for t in times:
        fcast_point = model(series[t : t + window_size]).item()
        forecast.append(fcast_point)

    return fcast_times, torch.tensor(forecast).float()


# def nn_model_fcast(
#     times: Tensor,
#     values: Tensor,
#     model: Module | None = None,
#     window: int = 20,
#     prv_values: Tensor | None = None,
#     **kwargs,
# ) -> tuple[Tensor, ...]:
#     """Forecasting using a previously trained NN model

#     Args:
#         times (Tensor): _description_
#         values (Tensor): _description_
#         model (Module): _description_
#         window (_type_): _description_
#         prv_values (Tensor | None, optional): _description_. Defaults to None.

#     Returns:
#         tuple[Tensor, ...]: a 3-element tuple with the times,
#             the values and the forecasted values
#     """
#     if model is None:
#         raise ValueError("You need to provide a trained model")

#     if prv_values is None:
#         fcast_data = values
#         n = len(values) - window
#     else:
#         fcast_data = torch.hstack((prv_values[-window:], values))
#         n = len(values)

#     forecast = []
#     for i in range(n):
#         fcast_point = model(fcast_data[i : i + window]).item()
#         forecast.append(fcast_point)
#     forecast = torch.tensor(forecast)
#     return times, values, forecast
