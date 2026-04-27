import datetime
import io
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import torch

DEFAULT_CHECKPOINTS_DIR = "output/checkpoints"
DEFAULT_RESULTS_DIR = "output/results"
DEFAULT_LOGS_DIR = "output/logs"


def save_checkpoint(state, filename="checkpoint.pth.tar", directory=DEFAULT_CHECKPOINTS_DIR):
    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, filename))
    print(f"=> Saved checkpoint: {os.path.join(directory, filename)}")


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def plot_loss_curves(train_losses, val_losses, save_dir=DEFAULT_RESULTS_DIR, optimizer=None, learning_rate=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")

    markers = [i for i in epochs if i % 10 == 0 or i == 1 or i == len(epochs)]
    plt.plot(markers, [train_losses[i - 1] for i in markers], "bo")
    plt.plot(markers, [val_losses[i - 1] for i in markers], "ro")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loss_curves_{timestamp}"
    if optimizer:
        filename += f"_{optimizer}"
    if learning_rate:
        filename += f"_lr{str(learning_rate).replace('.', '-')}"
    filename += ".png"

    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Loss curves saved to '{path}'")

def plot_accuracy_curves(train_accuracies, val_accuracies, save_dir=DEFAULT_RESULTS_DIR, optimizer=None, learning_rate=None):
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, "b-", label="Training Accuracy")
    plt.plot(epochs, val_accuracies, "r-", label="Validation Accuracy")

    markers = [i for i in epochs if i % 10 == 0 or i == 1 or i == len(epochs)]
    plt.plot(markers, [train_accuracies[i - 1] for i in markers], "bo")
    plt.plot(markers, [val_accuracies[i - 1] for i in markers], "ro")

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"accuracy_curves_{timestamp}"
    if optimizer:
        filename += f"_{optimizer}"
    if learning_rate:
        filename += f"_lr{str(learning_rate).replace('.', '-')}"
    filename += ".png"

    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Accuracy curves saved to '{path}'")


class Logger:
    """Tees stdout to a timestamped log file and records git info + hyperparams."""

    def __init__(self, log_dir=DEFAULT_LOGS_DIR, hyperparams=None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_log_{timestamp}"
        if hyperparams:
            opt = hyperparams.get("OPTIMIZER", "")
            lr = hyperparams.get("LEARNING_RATE", "")
            if opt:
                filename += f"_{opt}"
            if lr:
                filename += f"_lr{str(lr).replace('.', '-')}"
        self.log_file_path = os.path.join(log_dir, f"{filename}.txt")
        self.log_file = open(self.log_file_path, "w")
        self.original_stdout = sys.stdout
        self.captured_output = io.StringIO()

    def start_capture(self):
        sys.stdout = self
        self._log_git_info()

    def stop_capture(self):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def write(self, text):
        self.original_stdout.write(text)
        self.log_file.write(text)
        self.captured_output.write(text)

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def _log_git_info(self):
        self.write("\n" + "=" * 50 + "\nGIT INFORMATION\n" + "=" * 50 + "\n")
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            self.write(f"Branch: {git_branch}\nCommit: {git_hash}\n")
            self.write(f"Recover with: git checkout {git_hash}\n")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.write("Git info unavailable.\n")
        self.write("=" * 50 + "\n\n")

    def log_hyperparameters(self, config_dict):
        self.write("\n" + "=" * 50 + "\nHYPERPARAMETERS\n" + "=" * 50 + "\n")
        for k, v in config_dict.items():
            self.write(f"{k}: {v}\n")
        self.write("=" * 50 + "\n\n")

    def log_final_metrics(self, train_loss, val_loss):
        self.write("\n" + "=" * 50 + "\nFINAL METRICS\n" + "=" * 50 + "\n")
        self.write(f"Final Training Loss: {train_loss:.6f}\n")
        self.write(f"Final Validation Loss: {val_loss:.6f}\n")
        self.write("=" * 50 + "\n\n")

    def get_log_file_path(self):
        return self.log_file_path
