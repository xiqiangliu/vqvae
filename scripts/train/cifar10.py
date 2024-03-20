import logging

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from vqvae.data.cifar10 import CIFAR10DataModule
from vqvae.modules.vqvae_model import VQVAE

logger = logging.getLogger("vqvae.train")

model = VQVAE(
    in_channels=3,
    num_embeddings=512,
    embedding_dim=640,  # 8 * 8 * 10
    commitment_cost=0.25,
    num_residual_layers=2,
    num_residual_channels=256,
    num_layers=2,
    kernel_sizes=[4, 4],
    strides=[2, 2],
    batchnorm=True,
    learning_rate=2e-4,
)

if __name__ == "__main__":
    L.seed_everything(42)
    dm = CIFAR10DataModule(train_batch_size=128)
    dm.prepare_data()
    dm.setup("train")
    dm.setup("validation")

    trainer = L.Trainer(
        max_steps=250_000,
        callbacks=[
            EarlyStopping(
                monitor="loss/recon_val", patience=10, mode="min", verbose=True
            )
        ],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
