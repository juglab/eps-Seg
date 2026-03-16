import argparse
import torch 
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from eps_seg.models import LVAEModel
from eps_seg.dataloaders.datamodules import EPSSegDataModule
from eps_seg.training.callbacks import EarlyStoppingWithPatiencePropagation, SemiSupervisedModeCallback, ThresholdSchedulerCallback, RadiusSchedulerCallback, OptimizerStateTransferCallback
from eps_seg.config.train import ExperimentConfig
from dotenv import load_dotenv
import wandb

def train(exp_config: ExperimentConfig, skip_supervised: bool = False, skip_semisupervised: bool = False, direct_ssl: bool = False):
    """
        Train an EPS-Seg model based on the provided experiment configuration.
        Args:
            exp_config (ExperimentConfig): The experiment configuration object containing paths to training, dataset, and model configs.
            skip_supervised (bool): If True, skip the supervised training phase and only perform semi-supervised training by loading the best supervised checkpoint.
    """
    train_config, dataset_config, model_config = exp_config.get_configs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    dm = EPSSegDataModule(cfg=dataset_config, train_cfg=train_config)
    if not skip_supervised and not direct_ssl:
        # Set random seed for reproducibility if provided
        if train_config.supervised_seed is not None:
            print(f"Setting random seed to {train_config.supervised_seed} for supervised training...")
            L.seed_everything(train_config.supervised_seed, workers=True)

        model = LVAEModel(model_cfg=model_config, train_cfg=train_config).to(device)

        # Dummy forward pass to initialize model parameters
        if strategy == "ddp":
            model.eval()
            model(torch.zeros(size=(1,) + (model_config.color_channels,) + tuple(model_config.img_shape), device=model.device), validation_mode=False)
            model.train()

        supervised_best_ckpt_path = exp_config.best_checkpoint_path(mode="supervised")
        supervised_modelcheckpoint = ModelCheckpoint(
            monitor="val/CE_epoch",
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
            strategy=strategy,
            logger=supervised_logger,
            max_epochs=train_config.max_epochs,
            callbacks=[
                        supervised_modelcheckpoint,
                        EarlyStoppingWithPatiencePropagation(
                            monitor="val/total_loss_epoch",
                            patience=train_config.early_stopping_patience,
                            mode="min",
                            check_on_train_epoch_end=False, # Avoid checking on train epoch end to prevent double increment of radius 
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
        # Example for 2D convs
        # batch_size = train_config.batch_size
        # C = dataset_config.n_channels
        # H, W = model_config.img_shape[-2], model_config.img_shape[-1]
        supervised_trainer.fit(model, datamodule=dm)

        print("Supervised training complete. Best model at:", supervised_modelcheckpoint.best_model_path)
        # Finish the wandb run to avoid next run to log into the same run
        if train_config.use_wandb:
            wandb.finish()
    else:
        print("Skipping supervised training as per the argument.")

    #### SEMISUPERVISED TRAINING ####
    if not skip_semisupervised:
        print("Starting semisupervised training...")

        # Set random seed for reproducibility if provided
        if train_config.semisupervised_seed is not None:
            print(f"Setting random seed to {train_config.semisupervised_seed} for semisupervised training...")
            L.seed_everything(train_config.semisupervised_seed, workers=True)


        semisupervised_best_ckpt_path = exp_config.best_checkpoint_path(mode="semisupervised")
        semisupervised_modelcheckpoint = ModelCheckpoint(
            monitor="val/CE_epoch",
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

        semisupervised_callbacks = []
        # Initialize model from best supervised checkpoint
        if not direct_ssl:
            best_supervised_modelcheckpoint = exp_config.best_checkpoint_path(mode="supervised") if skip_supervised else supervised_modelcheckpoint.best_model_path
            model = LVAEModel.load_from_checkpoint(best_supervised_modelcheckpoint,
                                                model_cfg=model_config,
                                                train_cfg=train_config).to(device)
        
            semisupervised_callbacks += [
                OptimizerStateTransferCallback(
                        checkpoint_path=str(best_supervised_modelcheckpoint),
                        restore_optimizer=True,
                        restore_lr_scheduler=True,
                        restore_precision=train_config.amp,
                        strict_counts=True,
                    ) # Restore optimizer state, learning rate scheduler state, and AMP from the best supervised checkpoint
            ]
        else:
            # Initialize a SSL model directly
            model = LVAEModel(model_cfg=model_config, train_cfg=train_config).to(device)
            # there is no optimizer state to transfer

        semisupervised_callbacks += [
                    SemiSupervisedModeCallback(), # Switches model to semisupervised mode at the start of training
                    semisupervised_modelcheckpoint, # Save best SSL checkpoint based on validation loss
                    EarlyStoppingWithPatiencePropagation(
                            monitor="val/total_loss_epoch",
                            patience=train_config.early_stopping_patience,
                            mode="min",
                            check_on_train_epoch_end=False, # Avoid checking on train epoch end to prevent double increment of radius 
                        ), # Early stopping based on validation loss, propagates patience to radius scheduler by writing it to the model's state dict
                    LearningRateMonitor(logging_interval='epoch'), # Log learning rate at the end of each epoch
                    ThresholdSchedulerCallback(), # Adjusts the threshold for pseudo-labeling based on the validation performance, by writing it to the model's state dict
                    RadiusSchedulerCallback(radius_increment_patience=train_config.radius_increment_patience), # Increments the radius for pseudo-labeling after a certain number of epochs without improvement
                    ],

        semisupervised_trainer = L.Trainer(
            devices="auto",
            strategy=strategy,
            logger=semisupervised_logger,
            max_epochs=train_config.max_epochs,
            callbacks= semisupervised_callbacks,
            precision = "16-mixed" if train_config.amp else 32,
            gradient_clip_val=train_config.max_grad_norm, 
            log_every_n_steps=train_config.log_every_n_steps,
            deterministic=train_config.deterministic,
            use_distributed_sampler=False, # We have our own distributed sampler
            accumulate_grad_batches=train_config.accumulate_grad_batches,
            # fast_dev_run=True,
            )
        
        semisupervised_trainer.fit(model, datamodule=dm)

        print("Semisupervised training complete. Best model at:", semisupervised_modelcheckpoint.best_model_path)
        if train_config.use_wandb:
                wandb.finish()
    else:
        print("Skipping semisupervised training as per the argument.")

def main():
    # Allows to be run as: python -m eps_seg.train --exp_config path/to/exp_config.yaml --env_file path/to/.env
    parser = argparse.ArgumentParser(description="Train EPS-Seg Model")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to experiment configuration YAML file")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file with environment variables")
    parser.add_argument("--skip_supervised", action="store_true", help="Skip supervised training phase")
    parser.add_argument("--skip_semisupervised", action="store_true", help="Skip semi-supervised training phase")
    parser.add_argument("--direct_ssl", action="store_true", help="Directly switch to semi-supervised training without supervised pretraining")
    
    args = parser.parse_args()
    print("Loading experiment config from:", args.exp_config)
    print("Loading environment variables from:", args.env_file)
    load_dotenv(args.env_file)
    exp_config = ExperimentConfig.from_yaml(args.exp_config)

    train(exp_config, skip_supervised=args.skip_supervised, skip_semisupervised=args.skip_semisupervised, direct_ssl=args.direct_ssl)

if __name__ == "__main__":
   main()
