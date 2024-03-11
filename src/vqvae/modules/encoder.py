from typing import Sequence

import torch
import torch.nn as nn
from vqvae.modules.convolution import ConvLayer
from vqvae.modules.residual import ResidualStackedLayer


class VQVAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        num_residual_layers: int,
        num_residual_channels: int,
        strides: Sequence[int] | None = None,
        batchnorm: bool | Sequence[bool] = False,
    ):
        super().__init__()

        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError("Number of hidden channels and kernel sizes must match.")

        self.conv = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=in_channels if i == 0 else hidden_channels[i - 1],
                    out_channels=hidden_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i] if strides else 1,
                    batchnorm=batchnorm[i]
                    if isinstance(batchnorm, Sequence)
                    else batchnorm,
                )
                for i in range(len(hidden_channels))
            ]
        )

        self.residual_layers = ResidualStackedLayer(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1],
            hidden_channels=num_residual_channels,
            num_layers=num_residual_layers,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.residual_layers(x)
        return x
