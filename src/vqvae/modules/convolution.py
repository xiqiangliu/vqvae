import torch.nn as nn


class ConvLayer(nn.Sequential):
    """Convolutional layer with optional batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        batchnorm (bool): Whether to use batch normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batchnorm: bool = False,
        padding: int = 0,
    ):
        super().__init__(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
        )
