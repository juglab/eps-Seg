import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import shutil
import os

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

class ThresholdSchedulerCallback(L.Callback):
    """
        At the end of each epoch, increase the threshold by max_threshold.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if pl_module.current_training_mode == "semisupervised":
            pl_module.current_threshold = min(pl_module.current_threshold + pl_module.train_cfg.threshold_increment, pl_module.train_cfg.max_threshold)
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
                self._backup_checkpoint(pl_module)
                # Increase radius
                pl_module.trainer.datamodule.increase_radius()
                pl_module.current_radius = min(pl_module.current_radius + 1, pl_module.train_cfg.max_radius)
                pl_module.current_radius_patience = 0
        pl_module.log("train/radius", pl_module.current_radius, prog_bar=True, on_epoch=True)
