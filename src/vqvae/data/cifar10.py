import numpy as np
import pytorch_lightning as pl
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from vqvae import PROJECT_ROOT

cifar_transform = tv_transforms.Compose(
    [
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256, validation_size: float = 0.1):
        super().__init__()
        self.batch_size = batch_size
        self.validation_size = validation_size

    def prepare_data(self):
        train = tv_datasets.CIFAR10(
            root=str(PROJECT_ROOT / "datasets"), train=True, download=True
        )

        self.num_train = len(train)
        self.train_split = int(np.floor(self.validation_size * self.num_train))
        self.train_indices = np.arange(self.num_train)

    def setup(self, stage: str):
        match stage:
            case "train" | "validation":
                self.train_dataset = tv_datasets.CIFAR10(
                    root=str(PROJECT_ROOT / "datasets"),
                    train=True,
                    transform=cifar_transform,
                )
            case "test":
                self.test_dataset = tv_datasets.CIFAR10(
                    root=str(PROJECT_ROOT / "datasets"),
                    train=False,
                    transform=cifar_transform,
                )

    def train_dataloader(self):
        train_sampler = SubsetRandomSampler(
            self.train_indices[self.train_split :].tolist()
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=0,
        )

    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(
            self.train_indices[: self.train_split].tolist()
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
