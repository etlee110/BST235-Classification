import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Optimizer: SGD, Adam, AdamW
OPTIMIZER = "Adam"

# Model architecture
IMAGE_SIZE = 32      # CIFAR images are 32×32
NUM_CHANNELS = 3     # RGB
NUM_CLASSES = 2      # binary: airplane vs truck

# Seed
SEED = 42

# Output directories
CHECKPOINTS_DIR = os.path.join("output", "checkpoints")
RESULTS_DIR = os.path.join("output", "results")
LOGS_DIR = os.path.join("output", "logs")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
