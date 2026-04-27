import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Replace this with your actual dataset.

    Expected interface:
        __len__  → number of samples
        __getitem__(idx) → (input_tensor, label_tensor)
    """

    def __init__(self, split="train"):
        # TODO: load or generate your data here
        # Example: self.x, self.y = load_data(split)
        raise NotImplementedError("Implement CustomDataset for your data.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_loaders(batch_size, num_workers=0):
    """Return (train_loader, val_loader)."""
    train_dataset = CustomDataset(split="train")
    val_dataset = CustomDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
