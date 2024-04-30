import pytest
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LinearLR

from time_series_course.models import Linear, RNN, LSTM
from time_series_course.train import train_nn
from time_series_course.preprocessing import windowed_dataloader
from time_series_course.synthetic_data import noise, seasonality, trend

torch.manual_seed(42)

WINDOW_SIZE = 20


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


@pytest.mark.parametrize(
    "model",
    [
        Linear(seq_length=WINDOW_SIZE, num_layers=5, hidden_size=10, n_dim=1),
        RNN(seq_length=WINDOW_SIZE, num_layers=5, hidden_size=10, n_dim=1),
        LSTM(
            seq_length=WINDOW_SIZE,
            num_layers=5,
            hidden_size=5,
            n_dim=1,
            bidirectional=True,
        ),
    ],
)
def test_train_model(ts_data: tuple[Tensor, Tensor], model: Module) -> None:

    window_size = WINDOW_SIZE
    batch_size = 16
    n_epochs = 30

    dataloader = windowed_dataloader(
        ts_data[1], window=window_size + 1, batch_size=batch_size
    )
    test_x, test_y = next(iter(dataloader))

    loss_fn = torch.nn.HuberLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10)

    before = model(test_x)

    trained_model, loss_history = train_nn(
        model,
        ts_data[1],
        loss_fn=loss_fn,
        window_size=WINDOW_SIZE,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    assert isinstance(trained_model, Module)
    assert len(loss_history) == n_epochs + 1

    after = trained_model(test_x)

    loss_before = loss_fn()(before, test_y)
    loss_after = loss_fn()(after, test_y)
    assert loss_after.item() < loss_before.item()
