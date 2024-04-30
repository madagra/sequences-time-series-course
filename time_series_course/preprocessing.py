from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def fixed_partitioning(
    times: Tensor,
    series: Tensor,
    valid_size: float = 0.2,
    test_size: Optional[float] = None,
) -> tuple[Tensor, ...]:
    """Perform a fixed partitioning of the time series dataset

    Args:
        times (Tensor): a tensor containing the times
        series (Tensor): a tensor containing the time series values
        valid_size (float, optional): _description_. Defaults to 0.2.
        test_size (Optional[float], optional): _description_. Defaults to None.

    Returns:
        tuple[Tensor, ...]: _description_
    """
    total = len(series)

    if test_size is not None:
        assert valid_size + test_size < 1.0, "Test dataset size is too large"
        split_time_training = total - int(total * (valid_size + test_size))
        split_time_valid = int(split_time_training + total * valid_size)
        return (
            times[:split_time_training],
            series[:split_time_training],
            times[split_time_training:split_time_valid],
            series[split_time_training:split_time_valid],
            times[split_time_valid:],
            series[split_time_valid:],
        )
    else:
        split_time = total - int(total * valid_size)
        return (
            times[:split_time],
            series[:split_time],
            times[split_time:],
            series[split_time:],
        )


class WindowedDataset(Dataset):
    def __init__(self, data: Tensor, window: int):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window + 1


def windowed_dataloader(
    data: Tensor, window: int = 1, batch_size: int = 1, shuffle: bool = False
) -> DataLoader:
    """Sliding window dataloader on a 1D tensor

    The size of the dataset is n_batch x n_window x n_dim

    n_batch = the number of batches
    n_window = the size of the sliding window
    n_dim = the dimensionality of the time series (= 1 for univariate)
    """

    def _collate_fn(batch: Tensor):
        features = torch.vstack([b[:-1] for b in batch])
        targets = torch.tensor([b[-1] for b in batch])
        return features, targets

    dataset = WindowedDataset(data, window)
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=_collate_fn
    )
