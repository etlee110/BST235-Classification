import numpy as np

def read_r_array(path, shape):
    """Read float32 binary written by R's writeBin (column-major) and reshape."""
    raw = np.fromfile(path, dtype=np.float32)
    # R writes in column-major order; reshape with Fortran order gives (N, H, W, C)
    arr = raw.reshape(shape, order='F')
    return arr

N_train, N_test = 7500, 2500
H, W, C = 32, 32, 3

x_train_nhwc = read_r_array("data/x_train_raw.bin", (N_train, H, W, C))
x_test_nhwc  = read_r_array("data/x_test_raw.bin",  (N_test,  H, W, C))

# (N, H, W, C) -> (N, C, H, W) for PyTorch
x_train = x_train_nhwc.transpose(0, 3, 1, 2).copy()
x_test  = x_test_nhwc.transpose(0, 3, 1, 2).copy()

# normalize to [0, 1]
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

y_train = np.fromfile("data/y_train_raw.bin", dtype=np.int32).astype(np.int64)

print(f"x_train: {x_train.shape}  min={x_train.min():.3f} max={x_train.max():.3f}")
print(f"x_test:  {x_test.shape}  min={x_test.min():.3f} max={x_test.max():.3f}")
print(f"y_train: {y_train.shape}  unique={np.unique(y_train)}")
print(f"channel means: {x_train.mean(axis=(0,2,3))}")

np.savez_compressed("data/cifar-car-truck.npz",
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train)
print("Saved data/cifar-car-truck.npz")
