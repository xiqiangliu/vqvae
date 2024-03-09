import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedActivation(nn.Module):
    """Gated activation function.

    See https://arxiv.org/pdf/1606.05328.pdf for more details.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor):
        """Forward pass of the gated activation function.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: y = Tanh(W_f * x) * Sigmoid(W_g * x)
        """
        if inputs.shape[1] % 2 != 0:
            raise ValueError("The input tensor must have an even number of channels.")

        # Split the input tensor into two halves
        a, b = inputs.chunk(2, dim=1)

        return a.tanh_() * F.sigmoid(b)
