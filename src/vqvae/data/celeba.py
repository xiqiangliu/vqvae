import multiprocessing as mp
import lightning as L
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader

from vqvae import PROJECT_ROOT
from .transforms import standard_transform

celeba_transform = tv_transforms.Compose(
    [standard_transform, tv_transforms.Resize(128)]
)


class CelebADataModule(L.LightningDataModule):
    def __init__(self, train_batch_size: int = 256, val_batch_size: int = 1024):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

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
                    transform=standard_transform,
                    download=False,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=mp.cpu_count(),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=mp.cpu_count(),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
        )
