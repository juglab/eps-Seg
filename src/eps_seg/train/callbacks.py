# mlproject/engine/callbacks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import torch
import numpy as np
import shutil
import time
from eps_seg.train.optimizers import LabelSizeScheduler

class Callback:
    # Override the hooks you need.
    def on_fit_start(self, trainer): ...
    def on_epoch_start(self, trainer, epoch: int): ...
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict[str, float]): ...
    def on_validation_end(self, trainer, logs: Dict[str, float]): ...
    def on_epoch_end(self, trainer, epoch: int): ...
    def on_fit_end(self, trainer): ...


class ModelCheckpoint(Callback):
    def __init__(
        self,
        dirpath: str,
        monitor="val_total",
        mode="min",
        top_k: int = 1,
        save_last: bool = True,
    ):
        os.makedirs(dirpath, exist_ok=True)
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.save_last = save_last
        self._best = float("inf") if mode == "min" else -float("inf")
        self._saved = []  # [(metric, path)]

    def _is_better(self, x, y):
        return x < y if self.mode == "min" else x > y

    def on_epoch_end(self, trainer, epoch: int):
        if self.save_last:
            path = os.path.join(self.dirpath, "last.net")
            torch.save(trainer.model, path)

    def on_validation_end(self, trainer, logs: Dict[str, float]):
        metric = float(logs.get(self.monitor, np.inf))
        if np.isnan(metric):
            return
        if self._is_better(metric, self._best):
            self._best = metric
            path = os.path.join(self.dirpath, "best.net")
            torch.save(trainer.model, path)
            self._saved.append((metric, path))
            self._saved = sorted(
                self._saved, key=lambda t: t[0], reverse=(self.mode == "max")
            )[: self.top_k]


class EarlyStopping(Callback):
    def __init__(self, monitor="val_total", mode="min", patience=50, min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad_epochs = 0

    def _improved(self, val):
        if self.mode == "min":
            return val < self.best - self.min_delta
        return val > self.best + self.min_delta

    def on_validation_end(self, trainer, logs: Dict[str, float]):
        val = float(logs.get(self.monitor, np.inf))
        if np.isnan(val):
            return
        if self._improved(val):
            self.best = val
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                trainer.should_stop = True


class ReduceLROnPlateauStep(Callback):
    """Steps a ReduceLROnPlateau scheduler on validation end."""

    def __init__(self, monitor="val_total"):
        self.monitor = monitor

    def on_validation_end(self, trainer, logs: Dict[str, float]):
        if trainer.scheduler is not None and hasattr(trainer.scheduler, "step"):
            trainer.scheduler.step(float(logs.get(self.monitor)))


class LabelMaskSizeScheduler(Callback):
    """Mirrors your LabelSizeScheduler + mask size evolution."""

    def __init__(
        self,
        initial_label: int,
        final_label: int,
        initial_mask: int,
        final_mask: int,
        step_interval: int,
    ):

        self._label_sched = LabelSizeScheduler(initial_label, final_label, step_interval)
        self._mask_sched = LabelSizeScheduler(initial_mask, final_mask, step_interval)
        self.enabled_label = initial_label != final_label
        self.enabled_mask = initial_mask != final_mask

    def on_epoch_start(self, trainer, epoch: int):
        step = trainer.bad_epochs  # reuse your "patience_" idea for stepping
        if self.enabled_label and trainer.train_loader is not None:
            size = self._label_sched.get_label_size(step)
            if hasattr(trainer.train_loader.dataset, "update_patches"):
                trainer.train_loader.dataset.update_patches(size)
            if trainer.val_loader is not None and hasattr(
                trainer.val_loader.dataset, "update_patches"
            ):
                trainer.val_loader.dataset.update_patches(size)
        if self.enabled_mask:
            size = self._mask_sched.get_label_size(step)
            trainer.model.mask_size = size


class ThresholdScheduler(Callback):
    """Implements your threshold warm-up for semisupervised mode."""

    def __init__(
        self, start=0.50, max_val=0.99, step=0.005, only_in_mode="semisupervised"
    ):
        self.th = start
        self.max_val = max_val
        self.step = step
        self.only_in_mode = only_in_mode

    def on_epoch_end(self, trainer, epoch: int):
        if (
            getattr(trainer.model, "training_mode", None) == self.only_in_mode
            and self.th < self.max_val
        ):
            self.th = min(self.max_val, self.th + self.step)
        trainer.extra_state["threshold"] = self.th


class PlateauActions(Callback):
    """
    Replicates your two plateau behaviors:
      - If patience hits 50 in supervised: switch dataset+model to semisupervised and restore best
      - If patience hits 50 in semisupervised: increase dataset.radius by 1 and restore best (until < 10)
    """

    def __init__(self, dirpath: str):
        self.dirpath = dirpath

    def on_epoch_end(self, trainer, epoch: int):
        # "bad epochs" counter lives on trainer
        if trainer.bad_epochs == 50:
            model_folder = self.dirpath
            # Load best to model
            best_path = os.path.join(model_folder, "best.net")
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, weights_only=False)
                trainer.model.load_state_dict(checkpoint.state_dict())

            # Behavior depends on dataset mode
            ds = trainer.train_loader.dataset
            if getattr(ds, "mode", None) == "supervised":
                print("Switching to semi-supervised mode")
                if hasattr(ds, "set_mode"):
                    ds.set_mode("semisupervised")
                if hasattr(trainer.model, "update_mode"):
                    trainer.model.update_mode("semisupervised")
                shutil.copy(best_path, os.path.join(model_folder, "best_supervised.net"))
                trainer.bad_epochs = 0  # reset plateau counter

            elif (
                getattr(ds, "mode", None) == "semisupervised"
                and getattr(ds, "radius", 0) < 10
            ):
                print(f"Increasing radius from {ds.radius} to {ds.radius + 1}")
                if hasattr(ds, "increase_radius"):
                    ds.increase_radius()
                trainer.bad_epochs = 0  # reset plateau counter


class NaNDetector(Callback):
    def on_batch_end(self, trainer, batch_idx: int, logs: Dict[str, float]):
        for k, v in logs.items():
            if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
                print(f"[NaNDetector] {k} is NaN/Inf at batch {batch_idx}.")
                trainer.should_stop = True
                break


class WandbLogger(Callback):
    """Minimal W&B adapter to log dicts (keeps your current behavior)."""

    def __init__(self, use_wandb: bool, project: str, config: Dict[str, Any]):
        self.use = use_wandb
        self.project = project
        self.config = config
        self.run = None

    def on_fit_start(self, trainer):
        if not self.use:
            return
        import wandb, os

        os.environ["WANDB_START_TIMEOUT"] = "600"
        wandb.login()
        self.run = wandb.init(project=self.project, config=self.config)
        # You can add wandb.run.log_code(...) here if you want

    def on_batch_end(self, trainer, batch_idx: int, logs: Dict[str, float]):
        if self.use and self.run:
            import wandb

            wandb.log(logs)

    def on_validation_end(self, trainer, logs: Dict[str, float]):
        if self.use and self.run:
            import wandb

            wandb.log(logs)

    def on_fit_end(self, trainer):
        if self.use and self.run:
            import wandb

            wandb.finish()
