import argparse
import torch 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from eps_seg.models import LVAEModel
from eps_seg.dataloaders.datamodules import BetaSegDataModule
from eps_seg.train.callbacks import EarlyStoppingWithPatiencePropagation, SemiSupervisedModeCallback, ThresholdSchedulerCallback, RadiusSchedulerCallback
from eps_seg.config.train import ExperimentConfig
from dotenv import load_dotenv
import wandb

def train(exp_config: ExperimentConfig):
    """
        Train an EPS-Seg model based on the provided experiment configuration.
        Args:
            exp_config (ExperimentConfig): The experiment configuration object containing paths to training, dataset, and model configs.
    """


    train_config, dataset_config, model_config = exp_config.get_configs()

    # TODO: write a factory also for datamodules
    dm = BetaSegDataModule(cfg=dataset_config, train_cfg=train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LVAEModel(model_cfg=model_config, train_cfg=train_config).to(device)

    supervised_modelcheckpoint = ModelCheckpoint(
        monitor="val/total_loss_epoch",
        dirpath=exp_config.checkpoints_dir.resolve() / exp_config.experiment_name / train_config.model_name,
        filename="best_supervised",
        mode="min",
        save_last=True,
    )

    if train_config.use_wandb:
        supervised_logger = WandbLogger(
            name=f"{exp_config.experiment_name}_supervised",
            project=exp_config.project_name,
            save_dir=exp_config.logs_dir.resolve() / exp_config.experiment_name / train_config.model_name,
        )
    else:
        supervised_logger = TensorBoardLogger(
            name=f"{exp_config.experiment_name}_supervised",
            save_dir=exp_config.logs_dir.resolve() / exp_config.experiment_name / train_config.model_name,
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
    # Finish the wandb run to avoid next run to log into the same run
    if train_config.use_wandb:
        wandb.finish()

    semisupervised_modelcheckpoint = ModelCheckpoint(
        monitor="val/total_loss_epoch",
        dirpath=exp_config.checkpoints_dir.resolve() / exp_config.experiment_name / train_config.model_name,
        filename="best_semisupervised",
        mode="min",
        save_last=True,
    )

    if train_config.use_wandb:
        semisupervised_logger = WandbLogger(
            name=f"{exp_config.experiment_name}_semisupervised",
            project=exp_config.project_name,
            save_dir=exp_config.logs_dir.resolve() / exp_config.experiment_name / train_config.model_name,
        )
    else:
        semisupervised_logger = TensorBoardLogger(
            name=f"{exp_config.experiment_name}_semisupervised",
            save_dir=exp_config.logs_dir.resolve() / exp_config.experiment_name / train_config.model_name,
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
    if train_config.use_wandb:
            wandb.finish()

def main():
    # Allows to be run as: python -m eps_seg.train --exp_config path/to/exp_config.yaml --env_file path/to/.env
    parser = argparse.ArgumentParser(description="Train EPS-Seg Model")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to experiment configuration YAML file")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file with environment variables")
    args = parser.parse_args()
    print("Loading experiment config from:", args.exp_config)
    print("Loading environment variables from:", args.env_file)
    load_dotenv(args.env_file)
    exp_config = ExperimentConfig.from_yaml(args.exp_config)

    train(exp_config)

if __name__ == "__main__":
   main()