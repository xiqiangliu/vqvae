import torch.nn as nn


class DeConvLayer(nn.Sequential):
    """Deconvolutional layer with optional batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        batchnorm (bool): Whether to use batch normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batchnorm: bool = False,
        activation: str = "relu",
    ):
        if activation not in {"relu", "tanh"}:
            raise ValueError(f"Activation {activation} not supported for DeConvLayer.")
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True) if activation == "relu" else nn.Sigmoid(),
        )
