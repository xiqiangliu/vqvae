import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Quantizer(nn.Module):
    """Implementation of Vector Quantization Layer with EMA.

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    cluster_size: nn.Parameter
    embed_avg: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

        # EMA Parameters
        self.decay = decay
        self.eps = eps
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())

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
        inputs = rearrange(inputs, "b c h w -> (b h w) c").contiguous()

        # Calculate the distance between the input and the embeddings: (x - e)^2 = x^2 - 2*x*e + e^2
        distances = (
            (inputs**2).sum(dim=1, keepdim=True)
            - 2 * inputs @ self.embedding.weight.T
            + (self.embedding.weight**2).sum(dim=1)[None, ...]
        ).contiguous()

        # Find the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).view(b, h, w)
        quantized: torch.Tensor = rearrange(
            self.embedding(encoding_indices), "b h w c -> b c h w"
        ).contiguous()

        # Update embeddings usign EMA. Using it to avoid embedding collapse.
        # Refer to the appendix of the VQ-VAE paper for more details.
        # Use in-place operations as much as possible to avoid OOM error.
        if self.training:
            # Update the cluster size
            one_hot = (
                F.one_hot(encoding_indices, self.num_embeddings)
                .type(inputs.dtype)
                .view(-1, self.num_embeddings)
            )
            cluster_size = one_hot.sum(dim=0)
            self.cluster_size.data *= self.decay
            self.cluster_size.data.add_(cluster_size, alpha=1 - self.decay)

            # Update the average embedding.
            self.embed_avg.data *= self.decay
            self.embed_avg.add_((inputs.T @ one_hot).T, alpha=1 - self.decay)

            # Update the embeddings
            _cluster_size = self.cluster_size + self.eps
            _cluster_size *= self.cluster_size.sum()
            _cluster_size /= self.cluster_size.sum() + self.num_embeddings * self.eps
            self.embedding.weight.data.copy_(
                self.embed_avg / _cluster_size.unsqueeze(1)
            )

        # Calculate the quantization loss and commitment loss
        inputs = rearrange(inputs, "(b h w) c -> b c h w", b=b, h=h, w=w).contiguous()
        quantization_loss = torch.mean((quantized.detach() - inputs) ** 2)
        commitment_loss = torch.mean((quantized - inputs.detach()) ** 2)

        # Ensure straight through gradient for the quantized tensor
        quantized = inputs + (quantized - inputs).detach()

        return quantized, quantization_loss, commitment_loss
