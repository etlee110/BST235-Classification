import datetime
import os
import subprocess
import sys

import torch
from tqdm import tqdm

from utils import Logger, plot_loss_curves, plot_accuracy_curves, save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        num_epochs,
        use_logger=False,
        save_frequency=10,
        hyperparams=None,
        git_commit=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.use_logger = use_logger
        self.save_frequency = save_frequency
        self.hyperparams = hyperparams or {}
        self.git_commit = git_commit

        self.checkpoints_dir = self.hyperparams.get("CHECKPOINTS_DIR", "output/checkpoints")
        self.results_dir = self.hyperparams.get("RESULTS_DIR", "output/results")
        self.logs_dir = self.hyperparams.get("LOGS_DIR", "output/logs")

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.start_time = datetime.datetime.now()

        self.logger = None
        if self.use_logger:
            self.logger = Logger(log_dir=self.logs_dir, hyperparams=self.hyperparams)
            self.logger.start_capture()
            print(f"{'='*80}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.log_hyperparameters(self.hyperparams)

    def _train_epoch(self, epoch, pbar=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, targets in self.train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            correct += (predictions.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

            if pbar:
                pbar.update(1)
                pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader), correct / total

    def _evaluate_model(self, loader=None):
        loader = loader or self.val_loader
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                predictions = self.model(data)
                running_loss += self.loss_fn(predictions, targets).item()
                correct += (predictions.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)

        return running_loss / len(loader), correct / total

    def _save_checkpoint(self, epoch):
        opt_name = self.hyperparams.get("OPTIMIZER", type(self.optimizer).__name__)
        lr_val = self.hyperparams.get("LEARNING_RATE", "unk")
        checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        save_checkpoint(
            checkpoint,
            filename=f"checkpoint_{opt_name}_lr{lr_val}_epoch{epoch}.pth.tar",
            directory=self.checkpoints_dir,
        )

    def _make_git_commit(self, final_train_loss, final_val_loss):
        try:
            subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            h, rem = divmod(duration.seconds, 3600)
            m, s = divmod(rem, 60)
            msg = (
                f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Duration: {h:02d}:{m:02d}:{s:02d}\n"
                f"Final Training Loss: {final_train_loss:.6f}\n"
                f"Final Validation Loss: {final_val_loss:.6f}\n"
                f"Epochs: {self.num_epochs}\n"
                f"Learning Rate: {self.hyperparams.get('LEARNING_RATE', 'N/A')}"
            )
            subprocess.run(["git", "add", self.checkpoints_dir, self.results_dir, self.logs_dir])
            subprocess.run(["git", "commit", "-m", msg])
            print(f"\nCreated git commit:\n{msg}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\nGit commit failed: {e}")

    def train(self):
        print(f"Training on {self.device}")

        initial_train_loss, initial_train_acc = self._evaluate_model(loader=self.train_loader)
        initial_val_loss, initial_val_acc = self._evaluate_model()
        print(
            f"Epoch 0/{self.num_epochs}: "
            f"train_loss={initial_train_loss:.4f} train_acc={initial_train_acc:.4f} "
            f"val_loss={initial_val_loss:.4f} val_acc={initial_val_acc:.4f}"
        )
        self.train_losses.append(initial_train_loss)
        self.val_losses.append(initial_val_loss)
        self.train_accs.append(initial_train_acc)
        self.val_accs.append(initial_val_acc)

        for epoch in range(self.num_epochs):
            with tqdm(
                total=len(self.train_loader),
                file=sys.stdout,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                bar_format="{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:
                train_loss, train_acc = self._train_epoch(epoch, pbar)
                val_loss, val_acc = self._evaluate_model()
                pbar.set_postfix(
                    train_acc=f"{train_acc:.4f}",
                    val_acc=f"{val_acc:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"<<<<<< best val acc: {val_acc:.4f} >>>>>>")
                opt_name = self.hyperparams.get("OPTIMIZER", type(self.optimizer).__name__)
                lr_val = self.hyperparams.get("LEARNING_RATE", "unk")
                save_checkpoint(
                    {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
                    filename=f"best_model_{opt_name}_lr{lr_val}.pth.tar",
                    directory=self.checkpoints_dir,
                )

            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch + 1)

        if self.use_logger and self.logger:
            self.logger.log_final_metrics(
                self.train_losses[-1], self.val_losses[-1],
                self.train_accs[-1], self.val_accs[-1],
            )

        plot_loss_curves(
            self.train_losses, self.val_losses,
            save_dir=self.results_dir,
            optimizer=self.hyperparams.get("OPTIMIZER"),
            learning_rate=self.hyperparams.get("LEARNING_RATE"),
        )
        plot_accuracy_curves(
            self.train_accs, self.val_accs,
            save_dir=self.results_dir,
            optimizer=self.hyperparams.get("OPTIMIZER"),
            learning_rate=self.hyperparams.get("LEARNING_RATE"),
        )
        print(f"Training completed! Best val acc: {self.best_val_acc:.4f}")

        if self.use_logger and self.logger:
            print(f"Log file: {self.logger.get_log_file_path()}")
            self.logger.stop_capture()

        if self.git_commit:
            self._make_git_commit(self.train_losses[-1], self.val_losses[-1])

    def get_log_file_path(self):
        if self.use_logger and self.logger:
            return self.logger.get_log_file_path()
        return None
