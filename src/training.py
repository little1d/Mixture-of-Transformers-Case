"""
Training and validation logic for MoT experiments
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .utils import AverageMeter, accuracy, format_time, format_number


class Trainer:
    """Trainer class for MoT experiments"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        model_name: str = "model",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name

        # Device
        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * config.training.max_epochs
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_steps,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing
        )

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        # Metrics
        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch["image"].to(self.device, non_blocking=True)
            text_tokens = batch["text"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass
            outputs = self.model(images, text_tokens)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Logging
            if batch_idx % 50 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses.avg:.4f} Acc: {top1.avg:.2f}% "
                    f"Time: {batch_time.avg:.3f}s"
                )

        return {
            "loss": losses.avg,
            "accuracy": top1.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()

        # Metrics
        losses = AverageMeter()
        top1 = AverageMeter()

        # For detailed metrics
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch["image"].to(self.device, non_blocking=True)
                text_tokens = batch["text"].to(self.device, non_blocking=True)
                targets = batch["label"].to(self.device, non_blocking=True)

                batch_size = images.size(0)

                # Forward pass
                outputs = self.model(images, text_tokens)
                loss = self.criterion(outputs, targets)

                # Metrics
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                losses.update(loss.item(), batch_size)
                top1.update(acc1.item(), batch_size)

                # Store for detailed metrics
                all_outputs.append(F.softmax(outputs, dim=1).cpu())
                all_targets.append(targets.cpu())

        # Compute detailed metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Predictions
        predictions = all_outputs.argmax(dim=1).numpy()
        targets_np = all_targets.numpy()
        probabilities = all_outputs[:, 1].numpy()  # Probability of positive class

        # Detailed metrics
        accuracy_detailed = accuracy_score(targets_np, predictions) * 100
        f1 = f1_score(targets_np, predictions, average="weighted") * 100
        auc = roc_auc_score(targets_np, probabilities) * 100

        print(
            f"Val Epoch: {epoch} Loss: {losses.avg:.4f} "
            f"Acc: {top1.avg:.2f}% F1: {f1:.2f}% AUC: {auc:.2f}%"
        )

        return {
            "loss": losses.avg,
            "accuracy": top1.avg,
            "accuracy_detailed": accuracy_detailed,
            "f1_score": f1,
            "auc_score": auc,
        }

    def train(self) -> Dict[str, Any]:
        """Full training loop"""
        print(f"Starting training for {self.model_name}")
        print(
            f"Model parameters: {format_number(sum(p.numel() for p in self.model.parameters()))}"
        )
        print(f"Output directory: {self.output_dir}")

        start_time = time.time()

        for epoch in range(1, self.config.training.max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            if epoch % self.config.training.eval_every == 0:
                val_metrics = self.validate(epoch)

                # Track best model
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)

                # Store metrics
                train_metrics["epoch"] = epoch
                val_metrics["epoch"] = epoch
                self.train_metrics.append(train_metrics)
                self.val_metrics.append(val_metrics)

            # Save checkpoint
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch)

        total_time = time.time() - start_time

        print(f"Training completed in {format_time(total_time)}")
        print(
            f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}"
        )

        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy {self.best_val_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_metrics = checkpoint["train_metrics"]
        self.val_metrics = checkpoint["val_metrics"]

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    model_name: str,
) -> Trainer:
    """Create trainer instance"""
    return Trainer(model, train_loader, val_loader, config, model_name)
