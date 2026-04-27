import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, OPTIMIZER,
    NUM_CLASSES, IMAGE_SIZE, NUM_CHANNELS,
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, SEED,
)
from dataset import get_loaders
from model import CNN
from train import Trainer




train_loader, val_loader = get_loaders(batch_size=BATCH_SIZE)


for x, y in train_loader:
    print(x.shape, y.shape)
    print(y[:20])
    print(y.min().item(), y.max().item())
    break