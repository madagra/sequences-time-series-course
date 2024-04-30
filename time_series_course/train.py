from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from time_series_course.preprocessing import (
    windowed_dataloader,
)


def train_nn(
    model: nn.Module,
    series: torch.Tensor,
    loss_fn: nn.Module = torch.nn.L1Loss,
    window_size: int = 20,
    n_epochs: int = 1000,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.01,
    lr_scheduler: Optional[LRScheduler] = None,
    n_minibatch: Optional[int] = None,
    batch_size: int = 32,
    printout: bool = False,
) -> tuple[nn.Module, torch.Tensor]:

    # create windowed dataset where the last value is the target and
    # the first (window_size-1) values are the features
    dataloader = windowed_dataloader(
        series, window=window_size + 1, batch_size=batch_size, shuffle=False
    )

    # train on the train set
    _loss_fn = loss_fn()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print_every = n_epochs // (n_epochs * 0.01)
    loss_history = []

    for ne in range(n_epochs + 1):

        # optimizer step closure
        def _optimizer_step(current_loss: torch.Tensor) -> None:
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

        total_loss = torch.zeros(batch_size)

        # batch gradient descent with a single model update
        if n_minibatch is None:

            running_loss = torch.zeros(batch_size)
            for data in dataloader:
                x, y = data
                running_loss += _loss_fn(model(x), y)

            total_loss = running_loss
            _optimizer_step(running_loss.mean())

        # mini-batch gradient descent with multiple
        # model updates for each mini-batch
        else:

            assert n_minibatch > 0 and n_minibatch < len(
                dataloader
            ), f"Wrong number of minibatches: {n_minibatch}"

            # create a list of required sizes for each minibatch
            # in order to span the whole dataset. The last element
            # is the reminder in case the size of the dataloader is
            # not divisible for the number of minibatches
            _complete = len(dataloader) // n_minibatch
            _rest = len(dataloader) % n_minibatch
            minibatches = [_complete] * n_minibatch + [_rest]

            dl_it = iter(dataloader)
            for minibatch in minibatches:
                running_loss = torch.zeros(batch_size)
                for _ in range(minibatch):
                    x, y = next(dl_it)
                    running_loss += _loss_fn(model(x), y)

                total_loss += running_loss
                _optimizer_step(running_loss.mean())

        loss_history.append(total_loss.mean().item())
        if ne % print_every == 0 and printout:
            print(f"Loss at epoch {ne}: {loss_history[-1]}")

        # learning rate scheduler step
        if lr_scheduler is not None:
            lr_scheduler.step()

    return model, torch.tensor(loss_history, dtype=torch.float32)
