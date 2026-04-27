import datetime
import os
import subprocess
import sys

import torch
from tqdm import tqdm

from utils import Logger, plot_loss_curves, save_checkpoint


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
        self.best_val_loss = float("inf")
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

        for data, targets in self.train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if pbar:
                pbar.update(1)
                pbar.set_postfix(train_loss=f"{loss.item():.6f}")

        return running_loss / len(self.train_loader)

    def _evaluate_model(self, loader=None):
        loader = loader or self.val_loader
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                predictions = self.model(data)
                running_loss += self.loss_fn(predictions, targets).item()

        return running_loss / len(loader)

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

        initial_train = self._evaluate_model(loader=self.train_loader)
        initial_val = self._evaluate_model()
        print(
            f"Epoch 0/{self.num_epochs}: 100%|██████████| "
            f"{len(self.train_loader)}/{len(self.train_loader)} "
            f"[00:00<00:00, 0.00it/s, train_loss={initial_train:.6f}, val_loss={initial_val:.6f}]"
        )
        self.train_losses.append(initial_train)
        self.val_losses.append(initial_val)

        for epoch in range(self.num_epochs):
            with tqdm(
                total=len(self.train_loader),
                file=sys.stdout,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                bar_format="{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ) as pbar:
                train_loss = self._train_epoch(epoch, pbar)
                self.train_losses.append(train_loss)
                val_loss = self._evaluate_model()
                self.val_losses.append(val_loss)
                pbar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.3f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"<<<<<< reach best val loss : {val_loss} >>>>>>")
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
            self.logger.log_final_metrics(self.train_losses[-1], self.val_losses[-1])

        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            save_dir=self.results_dir,
            optimizer=self.hyperparams.get("OPTIMIZER"),
            learning_rate=self.hyperparams.get("LEARNING_RATE"),
        )
        print("Training completed!")

        if self.use_logger and self.logger:
            print(f"Log file: {self.logger.get_log_file_path()}")
            self.logger.stop_capture()

        if self.git_commit:
            self._make_git_commit(self.train_losses[-1], self.val_losses[-1])

    def get_log_file_path(self):
        if self.use_logger and self.logger:
            return self.logger.get_log_file_path()
        return None
