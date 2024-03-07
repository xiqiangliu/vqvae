import torch.nn as nn


class ResidualLayer(nn.Module):
    """Layer with two convolutional layers and a residual connection.
    Format is ReLU -> Conv -> ReLU -> Conv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of channels in the hidden layer.
        batchnorm (bool): Whether to use batch normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
        )

        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x):
        return self.residual_connection(x) + self.res_block(x)


class ResidualStackedLayer(nn.Sequential):
    """Stack of residual layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of channels in the hidden layer.
        num_layers (int): Number of residual layers.
    """

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int
    ):
        super().__init__(
            *[ResidualLayer(in_channels, out_channels, hidden_channels)] * num_layers
            + [nn.ReLU(inplace=True)]
        )


if __name__ == "__main__":
    import torch

    # Test ResidualLayer
    x = torch.rand(5, 3, 32, 32)
    layer = ResidualLayer(
        in_channels=3, out_channels=64, hidden_channels=64, batchnorm=True
    )
    out = layer(x)
    assert out.shape[2:] == x.shape[2:]

    # Test ResidualStackedLayer
    x = torch.rand(5, 3, 32, 32)
    layer = ResidualStackedLayer(3, 3, 32, 2)
    out = layer(x)
    assert out.shape[2:] == x.shape[2:]
    print("ResidualLayer and ResidualStackedLayer are working.")
