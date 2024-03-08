import pytorch_lightning as pl
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader

from vqvae import PROJECT_ROOT
from .transforms import standard_transform as fashion_mnist_transform


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        tv_datasets.FashionMNIST(root=str(PROJECT_ROOT / "datasets"), download=True)

    def setup(self, stage: str):
        match stage:
            case "train" | "validation":
                self.train_dataset = tv_datasets.FashionMNIST(
                    root=str(PROJECT_ROOT / "datasets"),
                    train=True,
                    transform=fashion_mnist_transform,
                )
            case "test":
                self.test_dataset = tv_datasets.FashionMNIST(
                    root=str(PROJECT_ROOT / "datasets"),
                    train=False,
                    transform=fashion_mnist_transform,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
