import matplotlib.pyplot as plt
import torch
from dataset import get_loaders

# get a small batch
train_loader, _ = get_loaders(batch_size=8)
data_iter = iter(train_loader)
images, labels = next(data_iter)

# unnormalize if needed (adjust if you used different values)
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
images = images * std + mean

# plot
fig, axes = plt.subplots(1, 8, figsize=(16, 3))
for i in range(8):
    img = images[i].permute(1, 2, 0).numpy()
    axes[i].imshow(img)
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()