"""One-time conversion: data/cifar-car-truck.RData → data/cifar-car-truck.npz

R layout: x (N, H, W, C) int [0,255], y numeric {0,1}
Output layout: x (N, C, H, W) float32 normalized to [0,1]

Run once from project root:
    python3 scripts/convert_rdata.py
"""

import subprocess
import tempfile
import os
import numpy as np

R_SCRIPT = """
load('data/cifar-car-truck.RData')

# x.train: (7500, 32, 32, 3)  x.test: (2500, 32, 32, 3)
# Write as raw binary row-major so numpy can read with np.frombuffer
writeBin(as.integer(x.train), '{xtrain_path}', size=4L)
writeBin(as.integer(x.test),  '{xtest_path}',  size=4L)
writeBin(as.double(y.train),  '{ytrain_path}', size=8L)
cat(dim(x.train), '\\n')
cat(dim(x.test),  '\\n')
cat(length(y.train), '\\n')
"""

def main():
    with tempfile.TemporaryDirectory() as tmp:
        xtrain_path = os.path.join(tmp, "xtrain.bin").replace("\\", "/")
        xtest_path  = os.path.join(tmp, "xtest.bin").replace("\\", "/")
        ytrain_path = os.path.join(tmp, "ytrain.bin").replace("\\", "/")

        r_code = R_SCRIPT.format(
            xtrain_path=xtrain_path,
            xtest_path=xtest_path,
            ytrain_path=ytrain_path,
        )
        result = subprocess.run(["R", "--no-save", "--quiet", "-e", r_code],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"R failed:\n{result.stderr}")

        x_train = np.frombuffer(open(xtrain_path, "rb").read(), dtype=np.int32)
        x_test  = np.frombuffer(open(xtest_path,  "rb").read(), dtype=np.int32)
        y_train = np.frombuffer(open(ytrain_path, "rb").read(), dtype=np.float64)

    x_train = x_train.reshape(7500, 32, 32, 3).astype(np.float32) / 255.0
    x_test  = x_test.reshape(2500,  32, 32, 3).astype(np.float32) / 255.0
    y_train = y_train.astype(np.int64)

    # NHWC → NCHW
    x_train = x_train.transpose(0, 3, 1, 2)
    x_test  = x_test.transpose(0, 3, 1, 2)

    out_path = os.path.join("data", "cifar-car-truck.npz")
    np.savez_compressed(out_path, x_train=x_train, x_test=x_test, y_train=y_train)
    print(f"Saved {out_path}")
    print(f"  x_train: {x_train.shape}  dtype={x_train.dtype}  range=[{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  x_test:  {x_test.shape}   dtype={x_test.dtype}")
    print(f"  y_train: {y_train.shape}  unique={np.unique(y_train)}")


if __name__ == "__main__":
    main()
