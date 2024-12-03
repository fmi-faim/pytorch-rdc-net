from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule
from skimage.measure import label
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset


class BBBC039Dataset(Dataset):
    def __init__(self, img_files, seg_files):
        self.img_files = img_files
        self.seg_files = seg_files

    @staticmethod
    def _normalize(img):
        img = img.astype(np.float32)
        img = img - img.mean()
        return img / img.std()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = self._normalize(imread(self.img_files[idx]))[np.newaxis]
        seg = label(imread(self.seg_files[idx])[..., 0])[np.newaxis]

        return img, seg


class BBBC039(LightningDataModule):
    def __init__(self, img_dir: Path, label_dir: Path):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted(list(self.img_dir.glob("*.tif")))
        self.label_files = [self.label_dir / f"{f.stem}.png" for f in self.img_files]

        self.train_split = int(0.8 * len(self.img_files))
        self.val_split = int(0.9 * len(self.img_files))

        print(
            f"Train: {self.train_split}, Val: {self.val_split - self.train_split}, Test: {len(self.img_files) - self.val_split}"
        )

    def train_dataloader(self):
        return DataLoader(
            BBBC039Dataset(
                self.img_files[: self.train_split], self.label_files[: self.train_split]
            ),
            batch_size=2,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            BBBC039Dataset(
                self.img_files[self.train_split : self.val_split],
                self.label_files[self.train_split : self.val_split],
            ),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            BBBC039Dataset(
                self.img_files[self.val_split :], self.label_files[self.val_split :]
            ),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
