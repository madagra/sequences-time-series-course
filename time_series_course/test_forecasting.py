import pytest
import torch
from torch import Tensor

from time_series_course.forecasting import mva_fcast, naive_fcast, nn_fcast
from time_series_course.preprocessing import (
    fixed_partitioning,
)
from time_series_course.models import Linear, RNN, LSTM
from time_series_course.synthetic_data import noise, seasonality, trend

window_size = 20
valid_size = 0.25


@pytest.fixture
def ts_data() -> tuple[Tensor, Tensor]:
    n_years = 4
    times = torch.arange(n_years * 365 + 1)

    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5
    seas = 365

    series = (
        baseline
        + trend(times, slope)
        + seasonality(times, period=seas, amplitude=amplitude)
    )
    series += noise(times, noise_level, seed=0)
    return times, series


def test_mva_fcast(ts_data: tuple[Tensor, Tensor]) -> None:

    times, series = ts_data[0], ts_data[1]

    _, _, t_valid, _ = fixed_partitioning(times, series, valid_size=valid_size)

    tfcast, vfcast = mva_fcast(series, t_valid, window_size=window_size)
    assert tfcast.size(0) == vfcast.size(0)
    assert vfcast[0] == torch.mean(series[tfcast[0] - window_size : tfcast[0]])


def test_naive_fcast(ts_data: tuple[Tensor, Tensor]) -> None:

    times, series = ts_data[0], ts_data[1]

    _, _, t_valid, _ = fixed_partitioning(times, series, valid_size=valid_size)

    tfcast, vfcast = naive_fcast(series, t_valid)
    assert tfcast.size(0) == vfcast.size(0)
    assert vfcast[0] == series[tfcast[0] - 1]


@pytest.mark.parametrize(
    "model",
    [
        Linear(seq_length=20, num_layers=5, hidden_size=10, n_dim=1),
        RNN(seq_length=20, num_layers=5, hidden_size=10, n_dim=1),
        LSTM(seq_length=20, num_layers=5, hidden_size=10, n_dim=1, bidirectional=True),
    ],
)
def test_nn_fcast(ts_data: tuple[Tensor, Tensor], model: torch.nn.Module) -> None:

    times, series = ts_data[0], ts_data[1]

    _, _, t_valid, _ = fixed_partitioning(times, series, valid_size=valid_size)

    tfcast, vfcast = nn_fcast(
        series,
        t_valid,
        model,
        window_size=20,
    )
    assert tfcast.size(0) == vfcast.size(0)
