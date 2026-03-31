import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import shutil
import os
from pathlib import Path
from typing import Optional
import torch

class EarlyStoppingWithPatiencePropagation(EarlyStopping):
    """
        Custom EarlyStopping that propagates the patience counter to the model.
    """
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if pl_module.current_training_mode == "semisupervised":
            if self.wait_count > 0:
                pl_module.current_radius_patience += 1
            else:
                pl_module.current_radius_patience = 0
                
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if pl_module.current_training_mode == "semisupervised":
            pl_module.log("val/radius_increase_patience", pl_module.current_radius_patience, prog_bar=True, on_epoch=True, sync_dist=True)
            pl_module.log("val/early_stopping_patience", self.wait_count, prog_bar=True, on_epoch=True, sync_dist=True)

        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


class SemiSupervisedModeCallback(L.Callback):
    """
        When training starts, switch the model to semi-supervised mode.
        This is to allow Lightning to load the complete model state (including optimizer states and global step)
        and change mode as soon as training starts.
    """
    def on_fit_start(self, trainer, pl_module):
        if pl_module.current_training_mode == "supervised":
            pl_module.update_mode("semisupervised")
            pl_module.trainer.datamodule.set_mode("semisupervised")
        return super().on_fit_start(trainer, pl_module)

class OptimizerStateTransferCallback(L.Callback):
    """
        Restore optimization state from a source checkpoint without restoring trainer loop/callback state.
    """

    def __init__(
        self,
        checkpoint_path: str,
        restore_optimizer: bool = True,
        restore_lr_scheduler: bool = True,
        restore_precision: bool = True,
        strict_counts: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.restore_optimizer = restore_optimizer
        self.restore_lr_scheduler = restore_lr_scheduler
        self.restore_precision = restore_precision
        self.strict_counts = strict_counts
        self._has_restored = False

    def _validate_counts(self, what: str, available: int, expected: int):
        if available != expected and self.strict_counts:
            raise RuntimeError(
                f"{what} count mismatch while restoring state from {self.checkpoint_path}: "
                f"checkpoint has {available}, trainer has {expected}."
            )

    def on_fit_start(self, trainer, pl_module):
        if self._has_restored:
            return super().on_fit_start(trainer, pl_module)

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for optimizer state transfer: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"OptimizerStateTransferCallback: loading training state from {ckpt_path}")

        if self.restore_optimizer:
            optimizer_states: Optional[list] = checkpoint.get("optimizer_states")
            if optimizer_states is None:
                if self.strict_counts:
                    raise RuntimeError("Checkpoint does not contain 'optimizer_states'.")
                print("OptimizerStateTransferCallback: no optimizer state found; skipping optimizer restore.")
            else:
                self._validate_counts("optimizer", len(optimizer_states), len(trainer.optimizers))
                for i, (optimizer, state_dict) in enumerate(zip(trainer.optimizers, optimizer_states)):
                    optimizer.load_state_dict(state_dict)
                    print(f"OptimizerStateTransferCallback: restored optimizer[{i}] state.")

        if self.restore_lr_scheduler:
            scheduler_states: Optional[list] = checkpoint.get("lr_schedulers")
            if scheduler_states is None:
                if self.strict_counts:
                    raise RuntimeError("Checkpoint does not contain 'lr_schedulers'.")
                print("OptimizerStateTransferCallback: no lr scheduler state found; skipping scheduler restore.")
            else:
                self._validate_counts("lr scheduler", len(scheduler_states), len(trainer.lr_scheduler_configs))
                for i, (scheduler_cfg, state_dict) in enumerate(zip(trainer.lr_scheduler_configs, scheduler_states)):
                    scheduler_cfg.scheduler.load_state_dict(state_dict)
                    print(f"OptimizerStateTransferCallback: restored lr_scheduler[{i}] state.")

        if self.restore_precision:
            precision_plugin = trainer.precision_plugin
            precision_plugin_name = type(precision_plugin).__name__
            precision_state = checkpoint.get(precision_plugin_name)
            if precision_state is None:
                print(
                    f"OptimizerStateTransferCallback: no precision state for key '{precision_plugin_name}'; "
                    "skipping precision restore."
                )
            elif hasattr(precision_plugin, "load_state_dict"):
                precision_plugin.load_state_dict(precision_state)
                print(f"OptimizerStateTransferCallback: restored precision state for '{precision_plugin_name}'.")
            else:
                print(
                    f"OptimizerStateTransferCallback: precision plugin '{precision_plugin_name}' has no "
                    "load_state_dict; skipping precision restore."
                )

        self._has_restored = True
        return super().on_fit_start(trainer, pl_module)

class ThresholdSchedulerCallback(L.Callback):
    """
        Move the training confidence threshold towards the configured max 
        threshold. This supports both increasing and decreasing schedules.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if pl_module.current_training_mode == "semisupervised":
            target_threshold = pl_module.train_cfg.max_threshold
            step = abs(pl_module.train_cfg.threshold_increment)

            if step > 0:
                if pl_module.current_threshold > target_threshold:
                    new_threshold = max(pl_module.current_threshold - step, target_threshold)
                else:
                    new_threshold = min(pl_module.current_threshold + step, target_threshold)

                if new_threshold != pl_module.current_threshold:
                    trainer.datamodule.set_confidence_threshold(new_threshold)
                    pl_module.current_threshold = new_threshold
        pl_module.log("train/threshold", pl_module.current_threshold, prog_bar=True, on_epoch=True)

class RadiusSchedulerCallback(L.Callback):
    """
        At the end of each epoch, if no improvement has been seen for radius_increment_patience epochs,
        increase the radius by 1, up to max_radius.

        Args:
            radius_increment_patience (int): Number of epochs with no improvement to wait before increasing radius
            best_ckpt_path (str): If provided, copies this checkpoint to a new file with the current radius in the filename whenever the radius is increased.
    """
    
    def __init__(self, radius_increment_patience: int, best_ckpt_path: str = None):
        super().__init__()
        self.radius_increment_patience = radius_increment_patience
        self.best_ckpt_path = best_ckpt_path
    
    def _backup_checkpoint(self, pl_module):
        if self.best_ckpt_path is not None:
            
            radius = pl_module.current_radius
            new_ckpt_path = f"{os.path.splitext(self.best_ckpt_path)[0]}_radius{radius}.ckpt"
            shutil.copyfile(self.best_ckpt_path, new_ckpt_path)
            print(f"Radius increased to {radius}. Backed up best checkpoint to {new_ckpt_path}.")

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if pl_module.current_training_mode == "semisupervised":
            if self.radius_increment_patience > 0 and pl_module.current_radius_patience >= self.radius_increment_patience:
                new_radius = min(pl_module.current_radius + 1, pl_module.train_cfg.max_radius)
                if new_radius > pl_module.current_radius:
                    self._backup_checkpoint(pl_module)
                    # Increase radius
                    pl_module.trainer.datamodule.set_radius(new_radius)
                    pl_module.current_radius = new_radius
                pl_module.current_radius_patience = 0
        pl_module.log("train/radius", pl_module.current_radius, prog_bar=True, on_epoch=True)


class PseudoLabelReevaluationCallback(L.Callback):
    """
        Re-evaluate neighbor confidences whenever validation improves during
        semisupervised training.
    """

    def __init__(self, early_stopping_callback: EarlyStoppingWithPatiencePropagation):
        super().__init__()
        self.early_stopping_callback = early_stopping_callback
        self._last_trigger_epoch: Optional[int] = None

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if trainer.sanity_checking or pl_module.current_training_mode != "semisupervised":
            return

        if self.early_stopping_callback.wait_count != 0:
            return

        if self._last_trigger_epoch == trainer.current_epoch:
            return

        self._last_trigger_epoch = trainer.current_epoch
        print("Validation improved. Re-evaluating pseudo-labels...")
        pl_module.re_evaluate_pseudo_labels()
