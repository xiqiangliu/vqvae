from typing import Sequence

import lightning as L
import torch
import torch.nn.functional as F

from vqvae.modules.decoder import VQVAEDecoder
from vqvae.modules.encoder import VQVAEEncoder
from vqvae.modules.quantizer import Quantizer


class VQVAE(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        num_residual_layers: int,
        num_residual_channels: int,
        num_layers: int,
        kernel_sizes: Sequence[int] | int,
        strides: Sequence[int] | int | None = None,
        batchnorm: bool = False,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "in_channels": in_channels,
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "commitment_cost": commitment_cost,
                "num_residual_layers": num_residual_layers,
                "num_residual_channels": num_residual_channels,
                "num_layers": num_layers,
                "kernel_sizes": kernel_sizes,
                "strides": strides,
                "batchnorm": batchnorm,
                "learning_rate": learning_rate,
            }
        )

        hidden_channels = [embedding_dim] * num_layers
        kernel_sizes = (
            [kernel_sizes] * num_layers
            if isinstance(kernel_sizes, int)
            else kernel_sizes
        )

        if strides is None:
            strides = [1] * num_layers
        elif isinstance(strides, int):
            strides = [strides] * num_layers

        self.encoder = VQVAEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_residual_layers=num_residual_layers,
            num_residual_channels=num_residual_channels,
            kernel_sizes=kernel_sizes,
            batchnorm=batchnorm,
            strides=strides,
        )

        self.decoder = VQVAEDecoder(
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            num_residual_layers=num_residual_layers,
            num_residual_channels=num_residual_channels,
            batchnorm=batchnorm,
            strides=strides,
        )

        self.embedding = Quantizer(num_embeddings, embedding_dim)

        self.lr = learning_rate
        self.commitment_cost = commitment_cost

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        z_continous = self.encoder(x)
        z_discrete, quantization_loss, commitment_loss = self.embedding(z_continous)
        x_recon = self.decoder(z_discrete)
        return x_recon, quantization_loss, commitment_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, _ = batch
        x_recon, quantization_loss, commitment_loss = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + quantization_loss + self.commitment_cost * commitment_loss
        self.log("loss/recon", recon_loss, prog_bar=True)
        self.log("loss/quantization", quantization_loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, _ = batch
        x_recon, quantization_loss, commitment_loss = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + quantization_loss + self.commitment_cost * commitment_loss
        self.log("loss/recon_val", recon_loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import torch

    # Test VQVAE
    batch_size = 32
    in_channels = 3
    num_embeddings = 512
    commitment_cost = 0.25
    num_residual_layers = 2
    num_residual_channels = embedding_dim = 256
    num_layers = 2
    kernel_sizes = [4, 4]
    strides = [2, 2]
    batchnorm = True
    learning_rate = 1e-3

    model = VQVAE(
        in_channels=in_channels,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        num_residual_layers=num_residual_layers,
        num_residual_channels=num_residual_channels,
        num_layers=num_layers,
        kernel_sizes=kernel_sizes,
        strides=strides,
        batchnorm=batchnorm,
        learning_rate=learning_rate,
    )

    x = torch.randn(batch_size, in_channels, 64, 64)
    z = model(x)
    print(z.shape)
