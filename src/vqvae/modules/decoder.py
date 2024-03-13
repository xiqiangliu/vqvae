from typing import Sequence

import torch
import torch.nn as nn
from vqvae.modules.deconvolution import DeConvLayer
from vqvae.modules.residual import ResidualStackedLayer


class VQVAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
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

        self.deconv = nn.Sequential(
            *[
                DeConvLayer(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i] if i > 0 else out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i] if strides else 1,
                    batchnorm=batchnorm[i]
                    if isinstance(batchnorm, Sequence)
                    else batchnorm,
                    activation="relu",
                )
                for i in range(len(hidden_channels) - 1, 0, -1)
            ]
            + [
                DeConvLayer(
                    in_channels=hidden_channels[0],
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[0],
                    stride=strides[0] if strides else 1,
                    batchnorm=batchnorm[0]
                    if isinstance(batchnorm, Sequence)
                    else batchnorm,
                    activation="sigmoid",  # Use tanh activation for the last layer so output will be in [0, 1]
                )
            ]
        )

        self.residual_layers = ResidualStackedLayer(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1],
            hidden_channels=num_residual_channels,
            num_layers=num_residual_layers,
        )

    def forward(self, x: torch.Tensor):
        x = self.residual_layers(x)
        x = self.deconv(x)
        return x
