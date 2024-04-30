from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        seq_length: int = 1,
        n_dim: int = 1,
        num_layers: int = 1,
        hidden_size: int = 5,
        act: nn.Module = nn.ReLU(),
    ) -> None:
        """Basic feedforward neural network architecture

        This model is a simple stack of linear layers with non-linear
        activation functions in-between

        The forward pass expects an input vector of dimensions:
            n_batch x seq_length

        Args:
            seq_length (int, optional): the number of elements in the sequence (e.g. window dimension)
            n_dim (int, optional): the dimensionality of the time series data
            num_layers (int, optional): the number of hidden layers
            hidden_size (int, optional): the number of neurons for each hidden layer
            act (nn.Module, optional): the non-linear activation function to use for stitching
                linear layers togeter
        """
        super().__init__()

        self.seq_length = seq_length
        self.num_neurons = hidden_size
        self.num_layers = num_layers
        self.n_dim = n_dim

        layers = []

        # input layer
        layers.append(nn.Linear(self.seq_length, hidden_size))

        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), act])

        # output layer
        layers.append(nn.Linear(hidden_size, n_dim))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze()


class RNN(nn.Module):
    def __init__(
        self,
        seq_length: int = 1,
        n_dim: int = 1,
        num_layers: int = 5,
        hidden_size: int = 5,
        nonlinearity: str = "tanh",
        scale_output: float = 100.0,
    ) -> None:
        """Simple recurrent NN for time series forecasting

        The model consists of multiple RNN layers (`num_layers` argument)
        of a `hidden_size` size implemented using the available PyTorch module.
        The final layer is a dense linear layer for compressing the output into
        a dimension of n_batch.

        The implementation is a sequence-to-vector RNN since only the output
        of the last RNN cell is considered.

        The forward pass expects an input vector of dimensions:
            n_batch x seq_length

        Args:
            seq_length (int, optional): the number of elements in the sequence (e.g. window dimension)
            n_dim (int, optional): The dimensionality of the time series. Usually is 1, for univariate
            num_layers (int, optional): number of layers in the RNN section
            hidden_size (int, optional): number of neurons in the hidden RNN layers
            nonlinearity (str, optional): RNN activation function
            scale_output (float, optional): a factor to scale the output of the network

        Returns:
            A vector of dimension n_batch with the model prediction
        """

        super().__init__()

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_dim = n_dim
        self.scale_output = scale_output

        self._rnn_layer = torch.nn.RNN(
            n_dim,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
        )
        self._out_layer = torch.nn.Linear(self.hidden_size, self.n_dim)

    def forward(self, x: Tensor, hidden0: Tensor | None = None) -> Tensor:

        if len(x.shape) == 2:
            n_batch = x.shape[0]
            seq_length = x.shape[1]
        elif len(x.shape) == 1:
            n_batch = 1
            seq_length = x.shape[0]
        else:
            raise ValueError(f"Wrong dimensionality of the input vector: {x.shape}")

        if hidden0 is None:
            hidden0 = torch.zeros(self.num_layers, n_batch, self.hidden_size)

        # make sure it respects the right shape
        x_reshaped = torch.reshape(x, (n_batch, seq_length, self.n_dim))

        # take the output of the RNN execution
        # consider only the last timestamp, basically
        # doing a sequence-to-vector RNN
        out, _ = self._rnn_layer(x_reshaped, hidden0)
        out = out[:, -1, :]

        # output a single result for each batch using
        # a linear layer for compression
        return self._out_layer(out).squeeze() * self.scale_output


class LSTM(nn.Module):
    def __init__(
        self,
        seq_length: int = 1,
        n_dim: int = 1,
        num_layers: int = 32,
        hidden_size: int = 5,
        scale_output: float = 100.0,
        bidirectional: bool = True,
    ) -> None:
        """LSTM recurrent NN for time series forecasting

        The model consists of multiple RNN layers with long-short term memory
        (`num_layers` argument) of a `hidden_size` size implemented using
        the available PyTorch module. The final layer is a dense linear layer
        for compressing the output into a dimension of n_batch.

        The implementation is a sequence-to-vector network since only the output
        of the last LSTM cell is considered when outputting the result to the
        dense layer

        The forward pass expects an input vector of dimensions:
            n_batch x seq_length

        Args:
            seq_length (int, optional): the number of elements in the sequence (e.g. window dimension)
            n_dim (int, optional): The dimensionality of the time series. Usually is 1, for univariate
            num_layers (int, optional): number of LSTM layers
            hidden_size (int, optional): number of neurons in the hidden LSTM layers
            scale_output (float, optional): a factor to scale the output of the network
            bidirectional (bool, optional): whether to make the network bidirectional or not

        Returns:
            A vector of dimension n_batch with the model prediction
        """

        super().__init__()

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_dim = n_dim
        self.scale_output = scale_output
        self.bidirectional = bidirectional

        # LSTM layer
        self._hdim = 1 if not self.bidirectional else 2
        self._lstm_layer = torch.nn.LSTM(
            n_dim,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
        )

        # dense layers at the end of the network
        self._dense_layer1 = torch.nn.Linear(
            self._hdim * self.hidden_size, self._hdim * self.hidden_size
        )
        self._dense_layer2 = torch.nn.Linear(
            self._hdim * self.hidden_size, self._hdim * self.hidden_size
        )

        # dense layer for compression, to return the data with right dimensionality
        self._out_layer = torch.nn.Linear(self._hdim * self.hidden_size, self.n_dim)

    def forward(
        self, x: Tensor, h0: Tensor | None = None, c0: Tensor | None = None
    ) -> Tensor:

        if len(x.shape) == 2:
            n_batch = x.size(0)
            seq_length = x.size(1)
        elif len(x.shape) == 1:
            n_batch = 1
            seq_length = x.size(0)
        else:
            raise ValueError(f"Wrong dimensionality of the input vector: {x.shape}")

        if h0 is None:
            h0 = torch.zeros(self._hdim * self.num_layers, n_batch, self.hidden_size)

        if c0 is None:
            c0 = torch.zeros(self._hdim * self.num_layers, n_batch, self.hidden_size)

        # make sure it respects the right shape
        x_reshaped = torch.reshape(x, (n_batch, seq_length, self.n_dim))

        out, _ = self._lstm_layer(x_reshaped, (h0, c0))
        out = out[:, -1, :]

        # output a single result for each batch using
        # a few linear layer for compression
        return self._out_layer(out).squeeze() * self.scale_output


class LSTMConv(LSTM):

    def __init__(
        self,
        seq_length: int = 1,
        n_dim: int = 1,
        num_layers: int = 32,
        hidden_size: int = 5,
        scale_output: float = 100,
        conv_channels: int = 16,
        bidirectional: bool = True,
    ) -> None:
        """Similar to the simple LSTM but with an initial Conv1D layer

        Additionally, two fully connected layers in addition to the final
        output one are also introduced. This network reproduces the most
        complex network used in the course

        Args:
            seq_length (int, optional): the number of elements in the sequence (e.g. window dimension)
            n_dim (int, optional): The dimensionality of the time series. Usually is 1, for univariate
            num_layers (int, optional): number of LSTM layers
            hidden_size (int, optional): number of neurons in the hidden LSTM layers
            scale_output (float, optional): a factor to scale the output of the network
            conv_channels (int, optional): number of output channels of the Conv1D layer. This
                corresponds then to the input size for the LSTM. Defaults to 16.
            bidirectional (bool, optional): whether to make the network bidirectional or not
        """
        super().__init__(
            seq_length, n_dim, num_layers, hidden_size, scale_output, bidirectional
        )

        self.conv_channels = conv_channels

        self._conv_layer = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=3,
            padding="same",
            stride=1,
        )

        self._lstm_layer = torch.nn.LSTM(
            input_size=self.conv_channels,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        # dense layers at the end of the network
        self._dense_layer1 = torch.nn.Linear(
            self._hdim * self.hidden_size, self._hdim * self.hidden_size
        )
        self._dense_layer2 = torch.nn.Linear(
            self._hdim * self.hidden_size, self._hdim * self.hidden_size
        )

    def forward(
        self, x: Tensor, h0: Tensor | None = None, c0: Tensor | None = None
    ) -> Tensor:

        if len(x.shape) == 2:
            n_batch = x.size(0)
            seq_length = x.size(1)
        elif len(x.shape) == 1:
            n_batch = 1
            seq_length = x.size(0)
        else:
            raise ValueError(f"Wrong dimensionality of the input vector: {x.shape}")

        if h0 is None:
            h0 = torch.zeros(self._hdim * self.num_layers, n_batch, self.hidden_size)

        if c0 is None:
            c0 = torch.zeros(self._hdim * self.num_layers, n_batch, self.hidden_size)

        # make sure it respects the right shape for the Convolution
        x_reshaped = torch.reshape(x, (n_batch, self.n_dim, seq_length))
        out = self._conv_layer(x_reshaped)
        out = out.permute(0, 2, 1)
        out, _ = self._lstm_layer(out, (h0, c0))
        out = out[:, -1, :]

        # output a single result for each batch using
        # a few fully connected layers for compression
        out = self._dense_layer2(self._dense_layer1(out))
        return self._out_layer(out).squeeze() * self.scale_output
