import argparse
import torch 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from eps_seg.models import LVAEModel
from eps_seg.dataloaders.datamodules import BetaSegTrainDataModule
from eps_seg.training.callbacks import EarlyStoppingWithPatiencePropagation, SemiSupervisedModeCallback, ThresholdSchedulerCallback, RadiusSchedulerCallback
from eps_seg.config.train import ExperimentConfig
from dotenv import load_dotenv
import wandb

def train(exp_config: ExperimentConfig, skip_supervised: bool = False):
    """
        Train an EPS-Seg model based on the provided experiment configuration.
        Args:
            exp_config (ExperimentConfig): The experiment configuration object containing paths to training, dataset, and model configs.
            skip_supervised (bool): If True, skip the supervised training phase and only perform semi-supervised training by loading the best supervised checkpoint.
    """
    train_config, dataset_config, model_config = exp_config.get_configs()

    # TODO: write a factory also for datamodules
    dm = BetaSegTrainDataModule(cfg=dataset_config, train_cfg=train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not skip_supervised:
        # Set random seed for reproducibility if provided
        if train_config.supervised_seed is not None:
            print(f"Setting random seed to {train_config.supervised_seed} for supervised training...")
            L.seed_everything(train_config.supervised_seed, workers=True)

        model = LVAEModel(model_cfg=model_config, train_cfg=train_config).to(device)

        supervised_best_ckpt_path = exp_config.best_checkpoint_path(mode="supervised")
        supervised_modelcheckpoint = ModelCheckpoint(
            monitor="val/total_loss_epoch",
            dirpath=supervised_best_ckpt_path.parent,
            filename=supervised_best_ckpt_path.stem,
            mode="min",
            save_last=True,
        )

        if train_config.use_wandb:
            supervised_logger = WandbLogger(
                name=f"{exp_config.experiment_name}_supervised",
                project=exp_config.project_name,
                save_dir=exp_config.get_log_dir(),
            )
        else:
            supervised_logger = TensorBoardLogger(
                name=f"{exp_config.experiment_name}_supervised",
                save_dir=exp_config.get_log_dir(),
            )

        #### SUPERVISED MODE ####
        supervised_trainer = L.Trainer(
            devices="auto",
            strategy="ddp",
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
            deterministic=train_config.deterministic,
            use_distributed_sampler=False, # We have our own distributed sampler
            accumulate_grad_batches=train_config.accumulate_grad_batches,
            # fast_dev_run=True,
            )

        model.update_mode("supervised")
        supervised_trainer.fit(model, datamodule=dm)

        print("Supervised training complete. Best model at:", supervised_modelcheckpoint.best_model_path)
        # Finish the wandb run to avoid next run to log into the same run
        if train_config.use_wandb:
            wandb.finish()
    else:
        print("Skipping supervised training as per the argument.")

    #### SEMISUPERVISED TRAINING ####

    # Set random seed for reproducibility if provided
    if train_config.semisupervised_seed is not None:
        print(f"Setting random seed to {train_config.semisupervised_seed} for semisupervised training...")
        L.seed_everything(train_config.semisupervised_seed, workers=True)


    semisupervised_best_ckpt_path = exp_config.best_checkpoint_path(mode="semisupervised")
    semisupervised_modelcheckpoint = ModelCheckpoint(
        monitor="val/total_loss_epoch",
        dirpath=semisupervised_best_ckpt_path.parent,
        filename=semisupervised_best_ckpt_path.stem,
        mode="min",
        save_last=True,
    )

    if train_config.use_wandb:
        semisupervised_logger = WandbLogger(
            name=f"{exp_config.experiment_name}_semisupervised",
            project=exp_config.project_name,
            save_dir=exp_config.get_log_dir(),
        )
    else:
        semisupervised_logger = TensorBoardLogger(
            name=f"{exp_config.experiment_name}_semisupervised",
            save_dir=exp_config.get_log_dir(),
        )

    semisupervised_trainer = L.Trainer(
        devices="auto",
        strategy="ddp",
        logger=semisupervised_logger,
        max_epochs=train_config.max_epochs,
        callbacks=[
                SemiSupervisedModeCallback(), # Switches model to semisupervised mode at the start of training
                semisupervised_modelcheckpoint, 
                EarlyStoppingWithPatiencePropagation(
                        monitor="val/total_loss_epoch",
                        patience=train_config.early_stopping_patience,
                        mode="min",
                    ),
                LearningRateMonitor(logging_interval='step'),
                ThresholdSchedulerCallback(),
                # RadiusSchedulerCallback(radius_increment_patience=train_config.radius_increment_patience),
                ],
        precision = "16-mixed" if train_config.amp else 32,
        gradient_clip_val=train_config.max_grad_norm, 
        log_every_n_steps=train_config.log_every_n_steps,
        deterministic=train_config.deterministic,
        use_distributed_sampler=False, # We have our own distributed sampler
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        # fast_dev_run=True,
        )

    # Initialize model from best supervised checkpoint
    best_supervised_modelcheckpoint = exp_config.best_checkpoint_path(mode="supervised") if skip_supervised else supervised_modelcheckpoint.best_model_path
    # TODO: This resets the optimizers and the global_step, but calling .fit(ckpt_path=...) gives weird behavior when used on another trainer.
    model = LVAEModel.load_from_checkpoint(best_supervised_modelcheckpoint,
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
    parser.add_argument("--skip_supervised", action="store_true", help="Skip supervised training phase")

    args = parser.parse_args()
    print("Loading experiment config from:", args.exp_config)
    print("Loading environment variables from:", args.env_file)
    load_dotenv(args.env_file)
    exp_config = ExperimentConfig.from_yaml(args.exp_config)

    train(exp_config, skip_supervised=args.skip_supervised)

if __name__ == "__main__":
   main()