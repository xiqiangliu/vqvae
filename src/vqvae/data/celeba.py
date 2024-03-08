import lightning as L
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader

from vqvae import PROJECT_ROOT
from .transforms import standard_transform as celeba_transform


class CelebADataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        tv_datasets.CelebA(root=str(PROJECT_ROOT / "datasets"), download=True)

    def setup(self, stage: str):
        match stage:
            case "train":
                self.train_dataset = tv_datasets.CelebA(
                    root=str(PROJECT_ROOT / "datasets"),
                    split="train",
                    target_type="attr",
                    transform=celeba_transform,
                    download=False,
                )
            case "validation":
                self.train_dataset = tv_datasets.CelebA(
                    root=str(PROJECT_ROOT / "datasets"),
                    split="valid",
                    target_type="attr",
                    transform=celeba_transform,
                    download=False,
                )
            case "test":
                self.test_dataset = tv_datasets.CelebA(
                    root=str(PROJECT_ROOT / "datasets"),
                    split="test",
                    target_type="attr",
                    transform=celeba_transform,
                    download=False,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
