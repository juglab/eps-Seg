import argparse
from eps_seg.config import LVAEConfig
from eps_seg.models import LVAEModel
from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.datamodules import BetaSegDataModule
from eps_seg.config.datasets import BetaSegDatasetConfig
import torch 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


class EarlyStoppingWithPatiencePropagation(EarlyStopping):
    """
        Custom EarlyStopping that propagates the patience counter to the model.
    """
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.wait_count > 0:
            pl_module.current_radius_patience += 1
        else:
            pl_module.current_radius_patience = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        pl_module.log("val/radius_increase_patience", pl_module.current_radius_patience, prog_bar=True, on_epoch=True)
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
    """
    
    def __init__(self, radius_increment_patience: int):
        super().__init__()
        self.radius_increment_patience = radius_increment_patience
    
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if pl_module.current_training_mode == "semisupervised":
            if self.radius_increment_patience > 0 and pl_module.current_radius_patience >= self.radius_increment_patience:
                pl_module.trainer.datamodule.increase_radius()
                pl_module.current_radius = min(pl_module.current_radius + 1, pl_module.train_cfg.max_radius)
                pl_module.current_radius_patience = 0
        pl_module.log("train/radius", pl_module.current_radius, prog_bar=True, on_epoch=True)


def train(train_config: TrainConfig, dataset_config: BetaSegDatasetConfig, model_config: LVAEConfig):
  
    dm = BetaSegDataModule(cfg=dataset_config, train_cfg=train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LVAEModel(model_cfg=model_config, train_cfg=train_config).to(device)

    supervised_modelcheckpoint = ModelCheckpoint(
        monitor="val/total_loss_epoch",
        dirpath=f"checkpoints/{train_config.model_name}/",
        filename="best_supervised",
        mode="min",
        save_last=True,
    )

    supervised_logger = TensorBoardLogger(
        name=train_config.model_name + "_supervised",
        save_dir=f"logs/{train_config.model_name}/",
    )

    #### SUPERVISED MODE ####
    supervised_trainer = L.Trainer(
        devices=1,
        logger=supervised_logger,
        max_epochs=train_config.max_epochs,
        callbacks=[
                    supervised_modelcheckpoint,
                    EarlyStoppingWithPatiencePropagation(
                        monitor="val/total_loss_epoch",
                        patience=train_config.early_stopping_patience,
                        mode="min",
                    )
                ],
        precision = "16-mixed" if train_config.amp else 32,
        gradient_clip_val=train_config.max_grad_norm, 
        log_every_n_steps=train_config.log_every_n_steps,
        )

    # First phase: Supervised training
    model.update_mode("supervised")

    supervised_trainer.fit(model, datamodule=dm)

    print("Supervised training complete. Best model at:", supervised_modelcheckpoint.best_model_path)

    semisupervised_modelcheckpoint = ModelCheckpoint(
        monitor="val/total_loss_epoch",
        dirpath=f"checkpoints/{train_config.model_name}/",
        filename="best_semisupervised",
        mode="min",
        save_last=True,
    )

    semisupervised_logger = TensorBoardLogger(
        name=train_config.model_name + "_semisupervised",
        save_dir=f"logs/{train_config.model_name}/",
    )

    semisupervised_trainer = L.Trainer(
        devices=1,
        logger=semisupervised_logger,
        max_epochs=train_config.max_epochs,
        callbacks=[
                SemiSupervisedModeCallback(),
                semisupervised_modelcheckpoint, 
                EarlyStoppingWithPatiencePropagation(
                        monitor="val/total_loss_epoch",
                        patience=train_config.early_stopping_patience,
                        mode="min",
                    ),
                LearningRateMonitor(logging_interval='step'),
                ThresholdSchedulerCallback(),
                RadiusSchedulerCallback(radius_increment_patience=train_config.radius_increment_patience),
                ],
        precision = "16-mixed" if train_config.amp else 32,
        gradient_clip_val=train_config.max_grad_norm, 
        log_every_n_steps=train_config.log_every_n_steps,
        )

    # TODO: This resets the optimizers and the global_step, but calling .fit(ckpt_path=...) gives weird behavior when used on another trainer.
    model = LVAEModel.load_from_checkpoint(supervised_modelcheckpoint.best_model_path,
                                        model_cfg=model_config,
                                        train_cfg=train_config).to(device)

    semisupervised_trainer.fit(model, datamodule=dm)

    print("Semisupervised training complete. Best model at:", semisupervised_modelcheckpoint.best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EPS-Seg Model")
    parser.add_argument("--train_config", type=str, required=True, help="Path to training configuration YAML file")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset configuration YAML file")
    parser.add_argument("--model_config", type=str, required=False, help="Path to model configuration YAML file")
    args = parser.parse_args()

    train_config = TrainConfig.from_yaml(args.train_config)
    dataset_config = BetaSegDatasetConfig.from_yaml(args.dataset_config)
    if args.model_config:
        model_config = LVAEConfig.from_yaml(args.model_config)
    else:
        model_config = LVAEConfig()
    
    train(train_config, dataset_config, model_config)