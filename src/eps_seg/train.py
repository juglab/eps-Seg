import argparse
import gc
import shutil
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from eps_seg.config.train import ExperimentConfig, TrainConfig
from eps_seg.dataloaders.datamodules import EPSSegDataModule
from eps_seg.models import LVAEModel
from eps_seg.training.callbacks import (
    OptimizerStateTransferCallback,
    StageMetricCarryoverCallback,
)

"""
Training entrypoint for the staged EPS-Seg++ workflow.

The staged training loop works as follows:

1. Create or load the stage-0 scheduler.
2. Train one stage at a time, always saving ``best`` and ``last`` checkpoints.
3. Before stage K>0 starts, copy the previous stage best checkpoint to the new
   stage best path and seed the live checkpointing callbacks with the previous
   best validation score.
4. Extend the scheduler between stages by adding a fixed number of accepted
   pseudo-labels.

"""


def build_model_for_stage(
    model_cfg,
    train_cfg: TrainConfig,
    mode: Literal["supervised", "semisupervised"],
    weights_checkpoint_path: str | None = None,
) -> LVAEModel:
    """
        Build the model used for one training stage.

        If a checkpoint path is given, weights are loaded from that checkpoint.
        Otherwise a fresh model is created from the current config.

        The LVAE is then switched to the requested mode so the same staged loop
        can be used for both semisupervised training and supervised upper-bound
        runs.
    """
    if weights_checkpoint_path is not None:
        model = LVAEModel.load_from_checkpoint(
            weights_checkpoint_path,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            map_location="cpu",
        )
    else:
        model = LVAEModel(model_cfg=model_cfg, train_cfg=train_cfg)

    model.update_mode(mode)
    return model


def build_logger(exp_config: ExperimentConfig, train_cfg: TrainConfig, mode: str, stage_idx: int):
    """
        Build the experiment logger for one stage run.

        Each stage gets its own logger name so curves and checkpoints can be
        inspected stage by stage.
    """
    logger_name = f"{exp_config.experiment_name}_{mode}_K{stage_idx}"
    if train_cfg.use_wandb:
        return WandbLogger(
            name=logger_name,
            project=exp_config.project_name,
            save_dir=exp_config.get_log_dir(),
        )
    return TensorBoardLogger(
        name=logger_name,
        save_dir=exp_config.get_log_dir(),
    )


def create_initial_scheduler(exp_config: ExperimentConfig) -> Path:
    """
        Create the stage-0 scheduler on disk using the current training split.

        Stage 0 contains only the initial GT-labelled samples selected by the
        dataset configuration. This function is only called when the scheduler
        is missing, so resumed runs do not recreate it.
    """
    train_cfg, dataset_cfg, _ = exp_config.get_configs()
    scheduler_path = exp_config.stage_scheduler_path(0)
    dm = EPSSegDataModule(cfg=dataset_cfg, train_cfg=train_cfg, scheduler_path=None, scheduler_stage_index=0)
    dm.prepare_data()
    dm.setup("fit")
    dm.train_dataset.save_scheduler_npz(scheduler_path)
    return scheduler_path


def train_one_stage(
    exp_config: ExperimentConfig,
    mode: Literal["supervised", "semisupervised"],
    stage_idx: int,
    scheduler_path: Path,
    weights_checkpoint_path: str | None = None,
    training_state_checkpoint_path: str | None = None,
) -> tuple[Path, Path]:
    """
        Train exactly one stage and save stage-specific ``best`` and ``last``
        checkpoints.

        Important stage behavior:

        - Stage K0 starts from scratch.
        - Stage K>0 starts from the previous stage best checkpoint.
        - Before stage K>0 starts, the previous stage best checkpoint is copied
          to the current stage best path.
        - The checkpointing callbacks are then seeded with the previous best
          score, so the new stage behaves like a continuation instead of a
          fresh run.

        Returns:
            tuple[Path, Path]:
                ``(best_checkpoint_path, last_checkpoint_path)`` for this stage.
    """
    train_cfg, dataset_cfg, model_cfg = exp_config.get_configs()
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    seed = train_cfg.supervised_seed if mode == "supervised" else train_cfg.semisupervised_seed
    if seed is not None:
        print(f"Setting random seed to {seed} for {mode} stage K{stage_idx}...")
        L.seed_everything(seed, workers=True)

    dm = EPSSegDataModule(
        cfg=dataset_cfg,
        train_cfg=train_cfg,
        scheduler_path=scheduler_path,
        scheduler_stage_index=stage_idx,
    )
    model = build_model_for_stage(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        mode=mode,
        weights_checkpoint_path=weights_checkpoint_path,
    )

    best_ckpt_path = exp_config.stage_checkpoint_path(mode, stage_idx, "best")
    last_ckpt_path = exp_config.stage_checkpoint_path(mode, stage_idx, "last")
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if stage_idx > 0 and weights_checkpoint_path is not None:
        # Pre-create the current stage best checkpoint from the previous stage
        # so interrupted runs still have a valid carry-over checkpoint on disk.
        previous_best_path = Path(weights_checkpoint_path)
        if not previous_best_path.exists():
            raise FileNotFoundError(f"Previous stage best checkpoint not found: {previous_best_path}")
        shutil.copy2(previous_best_path, best_ckpt_path)

    best_checkpoint = ModelCheckpoint(
        monitor="val/CE_epoch",
        dirpath=best_ckpt_path.parent,
        filename=best_ckpt_path.stem,
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    callbacks = [
        best_checkpoint,
        EarlyStopping(
            monitor="val/CE_epoch",
            patience=train_cfg.early_stopping_patience,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if training_state_checkpoint_path is not None:
        callbacks.append(
            OptimizerStateTransferCallback(
                checkpoint_path=str(training_state_checkpoint_path),
                restore_optimizer=True,
                restore_lr_scheduler=True,
                restore_precision=train_cfg.amp,
                restore_module_buffers=True,
                strict_counts=True,
            )
        )
    if stage_idx > 0 and weights_checkpoint_path is not None:
        # Seed checkpointing and early stopping with the previous stage best
        # score so the first validation is compared against the carried model.
        callbacks.append(
            StageMetricCarryoverCallback(
                reference_checkpoint_path=str(weights_checkpoint_path),
                carried_best_path=str(best_ckpt_path),
                monitor="val/CE_epoch",
                mode="min",
            )
        )

    logger = build_logger(exp_config, train_cfg, mode, stage_idx)
    trainer = L.Trainer(
        devices="auto",
        strategy=strategy,
        logger=logger,
        max_epochs=train_cfg.max_epochs,
        callbacks=callbacks,
        precision="16-mixed" if train_cfg.amp else 32,
        gradient_clip_val=train_cfg.max_grad_norm,
        log_every_n_steps=train_cfg.log_every_n_steps,
        deterministic=train_cfg.deterministic,
        use_distributed_sampler=False,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(last_ckpt_path)
    print(f"Completed {mode} stage K{stage_idx}. Best: {best_checkpoint.best_model_path} | Last: {last_ckpt_path}")

    if train_cfg.use_wandb:
        wandb.finish()

    del trainer
    del dm
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Path(best_checkpoint.best_model_path), last_ckpt_path


def evaluate_scheduler_extension(
    exp_config: ExperimentConfig,
    scheduler_path: Path,
    weights_checkpoint_path: Path,
    next_stage_idx: int,
) -> Path | None:
    """
        Build the scheduler for the next stage by extending the current one.

        The current best semisupervised model is used to score the newly sampled
        candidate voxels. Only pseudo-labels above the configured confidence
        threshold are accepted.

        Returns:
            Path | None:
                Path to the next scheduler if enough pseudo-labels were added,
                otherwise ``None`` to signal that staged expansion should stop
                because the best model is not confident enough to extend the
                scheduler with the required number of pseudo-labels.
    """
    train_cfg, dataset_cfg, model_cfg = exp_config.get_configs()
    dm = EPSSegDataModule(
        cfg=dataset_cfg,
        train_cfg=train_cfg,
        scheduler_path=scheduler_path,
        scheduler_stage_index=next_stage_idx - 1,
    )
    dm.prepare_data()
    dm.setup("fit")

    model = build_model_for_stage(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        mode="semisupervised",
        weights_checkpoint_path=str(weights_checkpoint_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    if model.model.data_std.sum() == 0:
        mean, std = dm.get_data_statistics()
        model.model.data_mean = torch.as_tensor(mean, device=device)
        model.model.data_std = torch.as_tensor(std, device=device)

    accepted = dm.train_dataset.add_pseudolabels_for_stage(
        stage_index=next_stage_idx,
        n_new_samples=train_cfg.pseudolabels_per_extension,
        evaluator=model.evaluate_candidate_batch,
        evaluation_batch_size=train_cfg.test_batch_size,
    )

    if accepted < train_cfg.pseudolabels_per_extension:
        print(
            f"Scheduler extension stopped at stage K{next_stage_idx}: "
            f"accepted {accepted}/{train_cfg.pseudolabels_per_extension} pseudo-labels."
        )
        del dm
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    next_scheduler_path = exp_config.stage_scheduler_path(next_stage_idx)
    dm.train_dataset.save_scheduler_npz(next_scheduler_path)
    del dm
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return next_scheduler_path


def latest_completed_stage(exp_config: ExperimentConfig, mode: Literal["supervised", "semisupervised"]) -> int:
    """
        Return the highest stage index that looks complete on disk.

        A stage is considered complete when all three files exist:

        - ``scheduler_K*.npz``
        - ``best_*_K*.ckpt``
        - ``last_*_K*.ckpt``

        This function is used by auto-resume to decide which stage should be
        trained next.
    """
    train_cfg, _, _ = exp_config.get_configs()
    max_stage = train_cfg.max_extensions
    latest = -1
    for stage_idx in range(max_stage + 1):
        scheduler_path = exp_config.stage_scheduler_path(stage_idx)
        best_ckpt_path = exp_config.stage_checkpoint_path(mode, stage_idx, "best")
        last_ckpt_path = exp_config.stage_checkpoint_path(mode, stage_idx, "last")
        if scheduler_path.exists() and best_ckpt_path.exists() and last_ckpt_path.exists():
            latest = stage_idx
        else:
            break
    return latest


def run_semisupervised_staged_training(exp_config: ExperimentConfig):
    """
        Run the staged semisupervised training loop.

        The loop creates stage 0 if needed, resumes from the latest completed
        stage when enabled, extends the scheduler between stages, and trains one
        stage at a time until scheduler growth stops or the configured maximum
        stage is reached.
    """
    train_cfg, _, _ = exp_config.get_configs()
    if not exp_config.stage_scheduler_path(0).exists():
        create_initial_scheduler(exp_config)

    completed_stage = latest_completed_stage(exp_config, "semisupervised") if train_cfg.auto_resume else -1
    start_stage = completed_stage + 1
    if start_stage == 0:
        print("Starting semisupervised staged training from stage K0.")
    else:
        print(f"Resuming semisupervised staged training from stage K{start_stage}.")

    for stage_idx in range(start_stage, train_cfg.max_extensions + 1):
        scheduler_path = exp_config.stage_scheduler_path(stage_idx)
        if not scheduler_path.exists():
            scheduler_path = evaluate_scheduler_extension(
                exp_config=exp_config,
                scheduler_path=exp_config.stage_scheduler_path(stage_idx - 1),
                weights_checkpoint_path=exp_config.stage_checkpoint_path("semisupervised", stage_idx - 1, "best"),
                next_stage_idx=stage_idx,
            )
            if scheduler_path is None:
                break

        if stage_idx == 0:
            weights_checkpoint_path = None
            training_state_checkpoint_path = None
        else:
            weights_checkpoint_path = str(exp_config.stage_checkpoint_path("semisupervised", stage_idx - 1, "best"))
            training_state_checkpoint_path = str(exp_config.stage_checkpoint_path("semisupervised", stage_idx - 1, "best"))

        train_one_stage(
            exp_config=exp_config,
            mode="semisupervised",
            stage_idx=stage_idx,
            scheduler_path=scheduler_path,
            weights_checkpoint_path=weights_checkpoint_path,
            training_state_checkpoint_path=training_state_checkpoint_path,
        )


def run_fully_supervised_upper_bound(exp_config: ExperimentConfig):
    """
        Replay the saved staged schedulers in fully supervised mode.

        This produces an upper-bound run for each available scheduler stage
        using the same selected coordinates but training on ground-truth
        labels for all scheduled samples.
    """
    train_cfg, _, _ = exp_config.get_configs()
    available_stages = []
    for stage_idx in range(train_cfg.max_extensions + 1):
        if exp_config.stage_scheduler_path(stage_idx).exists():
            available_stages.append(stage_idx)
        else:
            break

    if not available_stages:
        raise FileNotFoundError("No staged scheduler files were found. Run semisupervised training first.")

    completed_stage = latest_completed_stage(exp_config, "supervised") if train_cfg.auto_resume else -1
    start_stage = completed_stage + 1
    for stage_idx in available_stages[start_stage:]:
        if stage_idx == 0:
            weights_checkpoint_path = None
            training_state_checkpoint_path = None
        else:
            weights_checkpoint_path = str(exp_config.stage_checkpoint_path("supervised", stage_idx - 1, "best"))
            training_state_checkpoint_path = str(exp_config.stage_checkpoint_path("supervised", stage_idx - 1, "best"))

        train_one_stage(
            exp_config=exp_config,
            mode="supervised",
            stage_idx=stage_idx,
            scheduler_path=exp_config.stage_scheduler_path(stage_idx),
            weights_checkpoint_path=weights_checkpoint_path,
            training_state_checkpoint_path=training_state_checkpoint_path,
        )


def train(exp_config: ExperimentConfig):
    """
        Main training method for the staged workflow.

        The same entrypoint is used for:

        - semisupervised staged training
        - supervised upper-bound staged training

        The ``train_fully_supervised`` config flag selects which staged loop is
        active. The ``skip_*`` flags are mainly useful for debugging.
    """
    train_cfg, _, _ = exp_config.get_configs()
    if not train_cfg.train_fully_supervised:
        run_semisupervised_staged_training(exp_config)
    else:
        run_fully_supervised_upper_bound(exp_config)


def main():
    """CLI entrypoint for staged EPS-Seg training."""
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
