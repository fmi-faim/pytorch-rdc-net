from pathlib import Path

import numpy as np
import torch
import torchstain
from pytorch_lightning import LightningDataModule
from skimage import measure
from torch.utils.data import Dataset


class MoNuSegDataset(Dataset):
    def __init__(self, dataset_path: Path, key: str = "Train"):
        data = torch.load(dataset_path, weights_only=False)[key]
        self.key = key
        self.images = []
        self.labels = []
        he_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        for item in data:
            norm, H, E = he_normalizer.normalize(item["image"], stains=True)
            img = np.moveaxis(H, -1, 0).copy().astype(np.float32) / 255.0
            self.images.append(img)
            self.labels.append(measure.label(item["nucleus_masks"][np.newaxis]))

    def __len__(self):
        if self.key == "Train":
            return 3 * 100
        else:
            return len(self.images)

    def __getitem__(self, item):
        idx = item % len(self.images)

        if self.key == "Train":
            img, label = self.random_crop(self.images[idx], self.labels[idx])

            aug_index = np.random.randint(0, 8)
            if aug_index == 0:
                return img, label
            elif aug_index == 1:
                return np.flip(img, axis=-1).copy(), np.flip(label, axis=-1).copy()
            elif aug_index == 2:
                return np.rot90(img, 1, axes=(1, 2)).copy(), np.rot90(
                    label, 1, axes=(1, 2)
                ).copy()
            elif aug_index == 3:
                return np.rot90(img, 2, axes=(1, 2)).copy(), np.rot90(
                    label, 2, axes=(1, 2)
                ).copy()
            elif aug_index == 4:
                return np.rot90(img, 3, axes=(1, 2)).copy(), np.rot90(
                    label, 3, axes=(1, 2)
                ).copy()
            elif aug_index == 5:
                return np.flip(np.rot90(img, 1, axes=(1, 2)), axis=-1).copy(), np.flip(
                    np.rot90(label, 1, axes=(1, 2)), axis=-1
                ).copy()
            elif aug_index == 6:
                return np.flip(np.rot90(img, 2, axes=(1, 2)), axis=-1).copy(), np.flip(
                    np.rot90(label, 2, axes=(1, 2)), axis=-1
                ).copy()
            elif aug_index == 7:
                return np.flip(np.rot90(img, 3, axes=(1, 2)), axis=-1).copy(), np.flip(
                    np.rot90(label, 3, axes=(1, 2)), axis=-1
                ).copy()
        else:
            return self.images[idx], self.labels[idx]

    def random_crop(self, img, label):
        shape = (256, 256)

        y = np.random.randint(0, img.shape[-2] - shape[-2])
        x = np.random.randint(0, img.shape[-1] - shape[-1])

        return img[..., y : y + shape[-2], x : x + shape[-1]], measure.label(
            label[..., y : y + shape[-2], x : x + shape[-1]]
        )


class MoNuSeg(LightningDataModule):
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = dataset_path

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            MoNuSegDataset(self.dataset_path, key="Train"),
            batch_size=3,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            MoNuSegDataset(self.dataset_path, key="Validation"),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            MoNuSegDataset(self.dataset_path, key="Test"),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
