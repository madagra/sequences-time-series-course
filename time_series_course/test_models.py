import pytest
import torch
from torch import Tensor

from time_series_course.models import Linear, RNN, LSTM, LSTMConv
from time_series_course.preprocessing import windowed_dataloader
from time_series_course.synthetic_data import noise, seasonality, trend

WINDOW_SIZE = 20


@pytest.fixture
def ts_data() -> Tensor:
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
    return series


@pytest.mark.parametrize(
    "model",
    [
        Linear(seq_length=WINDOW_SIZE, num_layers=5, hidden_size=5),
        RNN(n_dim=1, seq_length=WINDOW_SIZE, hidden_size=5),
        LSTM(n_dim=1, seq_length=WINDOW_SIZE, hidden_size=5, bidirectional=True),
        LSTMConv(n_dim=1, seq_length=WINDOW_SIZE, hidden_size=5, bidirectional=True),
    ],
)
def test_models_output(ts_data: Tensor, model: torch.nn.Module) -> None:

    window_size = WINDOW_SIZE
    batch_size = 32
    dataloader = windowed_dataloader(
        ts_data, window=window_size + 1, batch_size=batch_size
    )

    x, _ = next(iter(dataloader))
    y_pred = model(x)
    assert y_pred.shape[0] == batch_size
