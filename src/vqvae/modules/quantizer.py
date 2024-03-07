import torch
import torch.nn as nn

from einops import rearrange


class Quantizer(nn.Module):
    """Implementation of Vector Quantization Layer.

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, inputs: torch.Tensor):
        """Forward pass of the quantizer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Quantized tensor.
            torch.Tensor: Quantization loss.
            torch.Tensor: Indices of the closest embeddings.
        """
        # Flatten the input tensor to (B*H*W, C)
        b, c, h, w = inputs.shape
        inputs = rearrange(inputs, "b c h w -> (b h w) c")

        # Calculate the distance between the input and the embeddings: (x - e)^2 = x^2 - 2*x*e + e^2
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            - 2 * inputs @ self.embedding.weight.T  #
            + torch.sum(self.embedding.weight**2, dim=1)
        )

        # Find the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2)

        # Calculate the quantization loss and commitment loss
        inputs = rearrange(inputs, "(b h w) c -> b c h w", b=b, h=h, w=w)
        quantization_loss = torch.mean((quantized.detach() - inputs) ** 2)
        commitment_loss = torch.mean((quantized - inputs.detach()) ** 2)

        return quantized, quantization_loss, commitment_loss
