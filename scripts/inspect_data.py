import numpy as np
import matplotlib.pyplot as plt

LABEL_NAMES = {0: "car", 1: "truck"}
DATA_PATH = "data/cifar-car-truck.npz"

data = np.load(DATA_PATH)
x_train = data["x_train"]   # (N, C, H, W)
x_test  = data["x_test"]
y_train = data["y_train"]

print("Dimensions")
print(f"  x_train : {x_train.shape}  dtype={x_train.dtype}  range=[{x_train.min():.3f}, {x_train.max():.3f}]")
print(f"  x_test  : {x_test.shape}  dtype={x_test.dtype}")
print(f"  y_train : {y_train.shape}  dtype={y_train.dtype}")
print(f"  labels  : {dict(zip(*np.unique(y_train, return_counts=True)))}")

n_show = 16
fig, axes = plt.subplots(2, n_show // 2, figsize=(n_show, 5))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = x_train[i].transpose(1, 2, 0).clip(0, 1)   # CHW -> HWC
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"{LABEL_NAMES[y_train[i]]} ({y_train[i]})", fontsize=9)
    ax.axis("off")

plt.suptitle("x_train samples", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
