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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"])
    args = parser.parse_args()

    hyperparams = {
        "DEVICE": DEVICE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": args.learning_rate if args.learning_rate else LEARNING_RATE,
        "OPTIMIZER": args.optimizer if args.optimizer else OPTIMIZER,
        "NUM_CLASSES": NUM_CLASSES,
        "IMAGE_SIZE": IMAGE_SIZE,
        "NUM_CHANNELS": NUM_CHANNELS,
        "CHECKPOINTS_DIR": CHECKPOINTS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "LOGS_DIR": LOGS_DIR,
        "SEED": SEED,
    }

    model = CNN(NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    opt_map = {
        "SGD": optim.SGD(model.parameters(), lr=hyperparams["LEARNING_RATE"], momentum=0.9),
        "Adam": optim.Adam(model.parameters(), lr=hyperparams["LEARNING_RATE"]),
        "AdamW": optim.AdamW(model.parameters(), lr=hyperparams["LEARNING_RATE"]),
    }
    optimizer = opt_map[hyperparams["OPTIMIZER"]]

    train_loader, val_loader = get_loaders(batch_size=BATCH_SIZE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        use_logger=True,
        save_frequency=10,
        hyperparams=hyperparams,
        git_commit=False,
    )
    trainer.train()


if __name__ == "__main__":
    main()
