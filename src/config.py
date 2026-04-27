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
INPUT_DIM = 128
NUM_CLASSES = 10

# Seed
SEED = 42

# Output directories
CHECKPOINTS_DIR = os.path.join("output", "checkpoints")
RESULTS_DIR = os.path.join("output", "results")
LOGS_DIR = os.path.join("output", "logs")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
