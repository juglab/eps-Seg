import lightning as L
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def extract_model_checkpoint_best_score(
    checkpoint_path: str | Path,
    monitor: str = "val/CE_epoch",
    mode: str = "min",
) -> float:
    """
        Read the best monitored score stored by Lightning's ModelCheckpoint.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found while reading best score: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    callbacks_state = checkpoint.get("callbacks", {})

    for callback_key, callback_state in callbacks_state.items():
        if not isinstance(callback_state, dict):
            continue
        if not callback_key.startswith("ModelCheckpoint"):
            continue
        if f"'monitor': '{monitor}'" not in callback_key:
            continue
        if f"'mode': '{mode}'" not in callback_key:
            continue

        best_score = callback_state.get("best_model_score")
        if best_score is None:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} contains ModelCheckpoint state for {monitor} "
                "but no best_model_score."
            )
        if hasattr(best_score, "item"):
            return float(best_score.item())
        return float(best_score)

    raise RuntimeError(
        f"Could not find ModelCheckpoint state for monitor={monitor!r}, mode={mode!r} "
        f"in checkpoint {ckpt_path}."
    )


class StageMetricCarryoverCallback(L.Callback):
    """
        Seed the current stage callbacks with the previous stage best score so
        checkpointing behaves like a continuation rather than a fresh run.
    """

    def __init__(
        self,
        reference_checkpoint_path: str | Path,
        carried_best_path: str | Path,
        monitor: str = "val/CE_epoch",
        mode: str = "min",
    ):
        super().__init__()
        self.reference_checkpoint_path = Path(reference_checkpoint_path)
        self.carried_best_path = Path(carried_best_path)
        self.monitor = monitor
        self.mode = mode
        self._has_seeded = False

    def _find_model_checkpoint(self, trainer) -> ModelCheckpoint:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.monitor == self.monitor and callback.mode == self.mode:
                return callback
        raise RuntimeError(
            f"Could not find ModelCheckpoint callback for monitor={self.monitor!r}, mode={self.mode!r}."
        )

    def _find_early_stopping(self, trainer) -> EarlyStopping | None:
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping) and callback.monitor == self.monitor and callback.mode == self.mode:
                return callback
        return None

    def on_fit_start(self, trainer, pl_module):
        if self._has_seeded:
            return super().on_fit_start(trainer, pl_module)

        reference_score = extract_model_checkpoint_best_score(
            checkpoint_path=self.reference_checkpoint_path,
            monitor=self.monitor,
            mode=self.mode,
        )
        score_tensor = torch.tensor(reference_score, dtype=torch.float32)

        checkpoint_callback = self._find_model_checkpoint(trainer)
        checkpoint_callback.best_model_score = score_tensor
        checkpoint_callback.best_model_path = str(self.carried_best_path)
        checkpoint_callback.current_score = score_tensor
        checkpoint_callback.kth_best_model_path = str(self.carried_best_path)
        checkpoint_callback.kth_value = score_tensor
        checkpoint_callback.best_k_models = {str(self.carried_best_path): score_tensor}

        early_stopping_callback = self._find_early_stopping(trainer)
        if early_stopping_callback is not None:
            early_stopping_callback.best_score = score_tensor
            early_stopping_callback.wait_count = 0

        print(
            f"StageMetricCarryoverCallback: seeded callbacks from {self.reference_checkpoint_path} "
            f"with best score {reference_score:.6f}."
        )
        self._has_seeded = True
        return super().on_fit_start(trainer, pl_module)


class OptimizerStateTransferCallback(L.Callback):
    """
        Restore the optimizer, LR scheduler, precision state, and selected
        module buffers from a previous stage checkpoint without restoring the
        entire Lightning loop state.
    """

    def __init__(
        self,
        checkpoint_path: str,
        restore_optimizer: bool = True,
        restore_lr_scheduler: bool = True,
        restore_precision: bool = True,
        restore_module_buffers: bool = True,
        strict_counts: bool = True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.restore_optimizer = restore_optimizer
        self.restore_lr_scheduler = restore_lr_scheduler
        self.restore_precision = restore_precision
        self.restore_module_buffers = restore_module_buffers
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

        if self.restore_module_buffers:
            state_dict = checkpoint.get("state_dict", {})
            for key in ("seen_samples",):
                if key in state_dict and hasattr(pl_module, key):
                    getattr(pl_module, key).copy_(state_dict[key].to(getattr(pl_module, key).device))
                    print(f"OptimizerStateTransferCallback: restored module buffer '{key}'.")

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
