import lightning as L
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader

from vqvae import PROJECT_ROOT
from .transforms import standard_transform as cifar_transform


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, train_batch_size: int = 256, val_batch_size: int = 1024):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        tv_datasets.CIFAR10(root=str(PROJECT_ROOT / "datasets"), download=True)

    def setup(self, stage: str):
        match stage:
            case "train" | "validation":
                self.train_dataset = tv_datasets.CIFAR10(
                    root=str(PROJECT_ROOT / "datasets"),
                    train=True,
                    transform=cifar_transform,
                )
            case "test" | "validation":
                self.test_dataset = tv_datasets.CIFAR10(
                    root=str(PROJECT_ROOT / "datasets"),
                    train=False,
                    transform=cifar_transform,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.val_batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
        )
