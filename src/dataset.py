import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# CIFAR-10 channel means and stds (computed over cars + trucks)
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)


class CifarCarTruckDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        # x: (N, C, H, W) float32 [0,1]   y: (N,) int64
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.augment = augment
        mean = torch.tensor(_MEAN).view(3, 1, 1)
        std  = torch.tensor(_STD).view(3, 1, 1)
        self.mean = mean
        self.std  = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx].clone()

        if self.augment:
            # random horizontal flip
            if torch.rand(1).item() < 0.5:
                img = img.flip(-1)
            # random crop: pad 4 then crop back to 32
            img = torch.nn.functional.pad(img, [4, 4, 4, 4], mode="reflect")
            i = torch.randint(0, 8, (1,)).item()
            j = torch.randint(0, 8, (1,)).item()
            img = img[:, i:i+32, j:j+32]

        img = (img - self.mean) / self.std
        return img, self.y[idx]


def get_loaders(batch_size, val_split=0.2, seed=42, num_workers=0,
                data_path=None):
    if data_path is None:
        data_path = os.path.join("data", "cifar-car-truck.npz")

    data = np.load(data_path)
    x_all = data["x_train"]   # (7500, 3, 32, 32) float32
    y_all = data["y_train"]   # (7500,) int64

    rng = np.random.default_rng(seed)
    n = len(y_all)
    idx = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = CifarCarTruckDataset(x_all[train_idx], y_all[train_idx], augment=True)
    val_ds   = CifarCarTruckDataset(x_all[val_idx],   y_all[val_idx],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
