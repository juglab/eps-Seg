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
Training entrypoint for EPS-Seg++ stage training.

The staged training loop works as follows:

1. Create or load the stage-0 scheduler.
2. Train one stage at a time, always saving ``best`` and ``last`` checkpoints.
3. Before stage K>0 starts, copy the previous stage best checkpoint to the new
   stage best path and seed the live checkpointing callbacks with the previous
   best validation score.
4. Extend the scheduler between stages by adding a fixed number of accepted
   pseudo-labels (semisupervised) or acquired GT labels (active learning) until the best
   model is not confident enough to extend the scheduler
   or the maximum number of stages is reached.

"""


def build_model_for_stage(
    model_cfg,
    train_cfg: TrainConfig,
    mode: Literal["supervised", "semisupervised", "active_learning"],
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


def build_logger(exp_config: ExperimentConfig, train_cfg: TrainConfig, mode: str, stage_idx: int | None):
    """
        Build the experiment logger for one stage run.

        Each stage gets its own logger name so curves and checkpoints can be
        inspected stage by stage.
    """
    logger_name = f"{exp_config.experiment_name}_{mode}" if stage_idx is None else f"{exp_config.experiment_name}_{mode}_K{stage_idx}"
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


def checkpoint_epoch(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1))


def create_initial_scheduler(exp_config: ExperimentConfig) -> Path:
    """
        Create the stage-0 scheduler on disk using the current training split.

        Stage 0 contains only the initial GT-labelled samples selected by the
        dataset configuration. This function is only called when the scheduler
        is missing, so resumed runs do not recreate it.
    """
    train_cfg, dataset_cfg, model_cfg = exp_config.get_configs()
    scheduler_path = exp_config.stage_scheduler_path(0)
    dm = EPSSegDataModule(cfg=dataset_cfg, train_cfg=train_cfg, model_cfg=model_cfg, scheduler_path=None, scheduler_stage_index=0,)
    dm.prepare_data()
    dm.setup("fit")
    dm.train_dataset.schedule.save_npz(scheduler_path)
    return scheduler_path


def train_one_stage(
    exp_config: ExperimentConfig,
    mode: Literal["supervised", "semisupervised", "active_learning"],
    stage_idx: int,
    scheduler_path: Path,
    weights_checkpoint_path: str | None = None,
    training_state_checkpoint_path: str | None = None,
    force_full_gt_labels: bool = False,
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
    if force_full_gt_labels and not train_cfg.train_fully_supervised:
        train_cfg = train_cfg.model_copy(update={"train_fully_supervised": True})
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    if mode == "semisupervised":
        seed = train_cfg.semisupervised_seed
    else:
        seed = train_cfg.supervised_seed
    if seed is not None:
        print(f"Setting random seed to {seed} for {mode} stage K{stage_idx}...")
        L.seed_everything(seed, workers=True)

    dm = EPSSegDataModule(
        cfg=dataset_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        scheduler_path=scheduler_path,
        scheduler_stage_index=stage_idx,
    )
    model = build_model_for_stage(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        mode=mode,
        weights_checkpoint_path=weights_checkpoint_path,
    )
    model.current_stage_idx = stage_idx

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
        monitor=train_cfg.monitored_metric,
        dirpath=best_ckpt_path.parent,
        filename=best_ckpt_path.stem,
        mode=train_cfg.monitored_metric_mode,
        save_top_k=1,
        save_last=False,
    )

    callbacks = [
        best_checkpoint,
        EarlyStopping(
            monitor=train_cfg.monitored_metric,
            patience=train_cfg.early_stopping_patience,
            mode=train_cfg.monitored_metric_mode,
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
                monitor=train_cfg.monitored_metric,
                mode=train_cfg.monitored_metric_mode,
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


def train_neighbor_phase(
    exp_config: ExperimentConfig,
    mode: Literal["supervised", "semisupervised"],
    fit_dataset_mode: Literal["supervised", "semisupervised"],
    max_epochs: int,
    ckpt_path: Path | None = None,
) -> tuple[Path, Path]:
    """
        Train one non-staged neighbor-SSL phase.

        When ``ckpt_path`` is provided, Lightning resumes the full training
        state from that checkpoint, including optimizer and LR scheduler state.
    """
    train_cfg, dataset_cfg, model_cfg = exp_config.get_configs()
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    seed = train_cfg.semisupervised_seed if mode == "semisupervised" else train_cfg.supervised_seed
    if seed is not None and ckpt_path is None:
        print(f"Setting random seed to {seed} for neighbor {mode} phase...")
        L.seed_everything(seed, workers=True)

    dm = EPSSegDataModule(
        cfg=dataset_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        scheduler_path=None,
        scheduler_stage_index=0,
        fit_dataset_kind="neighbor",
        fit_dataset_mode=fit_dataset_mode,
    )
    model = LVAEModel(model_cfg=model_cfg, train_cfg=train_cfg)
    model.update_mode(mode)

    best_ckpt_path = exp_config.checkpoint_path(mode, "best")
    last_ckpt_path = exp_config.checkpoint_path(mode, "last")
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_checkpoint = ModelCheckpoint(
        monitor=train_cfg.monitored_metric,
        dirpath=best_ckpt_path.parent,
        filename=best_ckpt_path.stem,
        mode=train_cfg.monitored_metric_mode,
        save_top_k=1,
        save_last=False,
    )

    callbacks = [
        best_checkpoint,
        EarlyStopping(
            monitor=train_cfg.monitored_metric,
            patience=train_cfg.early_stopping_patience,
            mode=train_cfg.monitored_metric_mode,
            check_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = build_logger(exp_config, train_cfg, mode, stage_idx=None)
    trainer = L.Trainer(
        devices="auto",
        strategy=strategy,
        logger=logger,
        max_epochs=max_epochs,
        callbacks=callbacks,
        precision="16-mixed" if train_cfg.amp else 32,
        gradient_clip_val=train_cfg.max_grad_norm,
        log_every_n_steps=train_cfg.log_every_n_steps,
        deterministic=train_cfg.deterministic,
        use_distributed_sampler=False,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
    )

    fit_ckpt_path = str(ckpt_path) if ckpt_path is not None else None
    if fit_ckpt_path is not None:
        print(f"Resuming neighbor {mode} phase from checkpoint: {fit_ckpt_path}")
    trainer.fit(model, datamodule=dm, ckpt_path=fit_ckpt_path)
    trainer.save_checkpoint(last_ckpt_path)
    print(f"Completed neighbor {mode} phase. Best: {best_checkpoint.best_model_path} | Last: {last_ckpt_path}")

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
    mode: Literal["semisupervised", "active_learning"],
) -> Path | None:
    """
        Build the scheduler for the next stage by optionally re-evaluating the
        active stage-extension rows and then extending the scheduler to the
        target size required for the next stage.

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
        model_cfg=model_cfg,
        scheduler_path=scheduler_path,
        scheduler_stage_index=next_stage_idx,
    )
    dm.prepare_data()
    dm.setup("fit")

    model = build_model_for_stage(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        mode=mode,
        weights_checkpoint_path=str(weights_checkpoint_path),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    if model.model.data_std.sum() == 0:
        mean, std = dm.get_data_statistics()
        model.model.data_mean = torch.as_tensor(mean, device=device)
        model.model.data_std = torch.as_tensor(std, device=device)

    schedule = dm.train_dataset.schedule
    schedule_engine = dm.train_dataset.schedule_engine

    extension_label_source = 1 if mode == "semisupervised" else 2
    active_rows_before = schedule.count_active_label_source_rows(extension_label_source)
    should_reevaluate = (
        train_cfg.pseudolabel_maintenance.name != "noop" and active_rows_before > 0
    )
    disabled_count = 0
    if should_reevaluate:
        disabled_count = schedule_engine.reevaluate_schedule(
            next_stage_idx=next_stage_idx,
            evaluator=model.evaluate_candidate_batch,
            evaluation_batch_size=train_cfg.test_batch_size,
            target_label_source=extension_label_source,
        )
    if disabled_count > 0 or active_rows_before > 0:
        schedule.bump_version()
        dm.train_dataset.sampling_version = schedule.sampling_version

    target_active_rows = active_rows_before + train_cfg.rows_per_extension
    accepted = schedule_engine.extend_schedule(
        stage_index=next_stage_idx,
        target_active_rows=target_active_rows,
        evaluator=model.evaluate_candidate_batch,
        evaluation_batch_size=train_cfg.test_batch_size,
        target_label_source=extension_label_source,
        candidate_evaluation_budget=train_cfg.candidate_evaluation_budget,
        candidate_order=train_cfg.schedule_admission.candidate_order,
    )
    if accepted > 0:
        schedule.bump_version()
        dm.train_dataset.sampling_version = schedule.sampling_version
        dm.train_dataset._print_schedule_report()

    active_rows_after = schedule.count_active_label_source_rows(extension_label_source)
    extension_desc = "pseudo-labels" if mode == "semisupervised" else "acquired GT labels"
    if active_rows_after < target_active_rows:
        print(
            f"Scheduler extension stopped at stage K{next_stage_idx}: "
            f"disabled {disabled_count} rows, accepted {accepted} new {extension_desc}, "
            f"and reached {active_rows_after}/{target_active_rows} active {extension_desc}."
        )
        del dm
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    next_scheduler_path = exp_config.stage_scheduler_path(next_stage_idx)
    schedule.save_npz(next_scheduler_path)
    del dm
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return next_scheduler_path


def latest_completed_stage(exp_config: ExperimentConfig, mode: Literal["supervised", "semisupervised", "active_learning"]) -> int:
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
                mode="semisupervised",
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


def run_active_learning_staged_training(exp_config: ExperimentConfig):
    """
        Run the staged active-learning loop.

        Stage K0 uses the initial GT-labelled scheduler, while every stage K>0
        adds newly acquired GT labels to the scheduler and then continues
        training from the previous stage best checkpoint and optimizer state.
    """
    train_cfg, _, _ = exp_config.get_configs()
    if not exp_config.stage_scheduler_path(0).exists():
        create_initial_scheduler(exp_config)

    completed_stage = latest_completed_stage(exp_config, "active_learning") if train_cfg.auto_resume else -1
    start_stage = completed_stage + 1
    if start_stage == 0:
        print("Starting active-learning staged training from stage K0.")
    else:
        print(f"Resuming active-learning staged training from stage K{start_stage}.")

    for stage_idx in range(start_stage, train_cfg.max_extensions + 1):
        scheduler_path = exp_config.stage_scheduler_path(stage_idx)
        if not scheduler_path.exists():
            scheduler_path = evaluate_scheduler_extension(
                exp_config=exp_config,
                scheduler_path=exp_config.stage_scheduler_path(stage_idx - 1),
                weights_checkpoint_path=exp_config.stage_checkpoint_path("active_learning", stage_idx - 1, "best"),
                next_stage_idx=stage_idx,
                mode="active_learning",
            )
            if scheduler_path is None:
                break

        if stage_idx == 0:
            weights_checkpoint_path = None
            training_state_checkpoint_path = None
        else:
            weights_checkpoint_path = str(exp_config.stage_checkpoint_path("active_learning", stage_idx - 1, "best"))
            training_state_checkpoint_path = str(exp_config.stage_checkpoint_path("active_learning", stage_idx - 1, "best"))

        train_one_stage(
            exp_config=exp_config,
            mode="active_learning",
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

        Each supervised stage trains from scratch and only reuses the
        corresponding scheduler as the dataset definition.
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
        train_one_stage(
            exp_config=exp_config,
            mode="supervised",
            stage_idx=stage_idx,
            scheduler_path=exp_config.stage_scheduler_path(stage_idx),
            weights_checkpoint_path=None,
            training_state_checkpoint_path=None,
            force_full_gt_labels=True,
        )


def run_fully_supervised_stage(exp_config: ExperimentConfig, stage_idx: int):
    """
        Train exactly one scheduler stage in fully supervised mode.

        Unlike the sequential replay helper, this path is intended for job
        arrays and stage-level parallelism. Each stage trains from scratch and
        only reuses the corresponding scheduler as the dataset definition, so
        all supervised jobs are independent from one another.
    """
    train_cfg, _, _ = exp_config.get_configs()
    if stage_idx < 0 or stage_idx > train_cfg.max_extensions:
        raise ValueError(
            f"Requested fully supervised stage K{stage_idx}, but max_extensions={train_cfg.max_extensions}."
        )

    scheduler_path = exp_config.stage_scheduler_path(stage_idx)
    if not scheduler_path.exists():
        raise FileNotFoundError(
            f"Scheduler for fully supervised stage K{stage_idx} not found: {scheduler_path}. "
            "Run semisupervised training for that stage first."
        )

    supervised_best_path = exp_config.stage_checkpoint_path("supervised", stage_idx, "best")
    supervised_last_path = exp_config.stage_checkpoint_path("supervised", stage_idx, "last")
    if train_cfg.auto_resume and supervised_best_path.exists() and supervised_last_path.exists():
        print(
            f"Skipping fully supervised stage K{stage_idx} because both outputs already exist: "
            f"{supervised_best_path} | {supervised_last_path}"
        )
        return

    train_one_stage(
        exp_config=exp_config,
        mode="supervised",
        stage_idx=stage_idx,
        scheduler_path=scheduler_path,
        weights_checkpoint_path=None,
        training_state_checkpoint_path=None,
        force_full_gt_labels=True,
    )


def run_neighbor_semisupervised_training(exp_config: ExperimentConfig):
    """
        Run the non-staged neighbor-based semisupervised training pipeline.

        Phase 1 trains labelled cached anchors in supervised mode. Phase 2
        resumes from the supervised best checkpoint, including optimizer and LR
        scheduler state, and continues in semisupervised mode with local
        unlabeled neighbors in each batch.
    """
    train_cfg, _, _ = exp_config.get_configs()
    supervised_best = exp_config.checkpoint_path("supervised", "best")
    supervised_last = exp_config.checkpoint_path("supervised", "last")
    semisupervised_best = exp_config.checkpoint_path("semisupervised", "best")
    semisupervised_last = exp_config.checkpoint_path("semisupervised", "last")

    if train_cfg.auto_resume and supervised_best.exists() and supervised_last.exists():
        print(f"Skipping neighbor supervised phase because outputs already exist: {supervised_best} | {supervised_last}")
    else:
        supervised_resume = supervised_last if train_cfg.auto_resume and supervised_last.exists() else None
        train_neighbor_phase(
            exp_config=exp_config,
            mode="supervised",
            fit_dataset_mode="supervised",
            max_epochs=train_cfg.resolved_neighbor_supervised_max_epochs,
            ckpt_path=supervised_resume,
        )

    if not supervised_best.exists():
        raise FileNotFoundError(
            f"Neighbor semisupervised phase requires supervised best checkpoint: {supervised_best}"
        )

    if train_cfg.auto_resume and semisupervised_best.exists() and semisupervised_last.exists():
        print(
            "Skipping neighbor semisupervised phase because outputs already exist: "
            f"{semisupervised_best} | {semisupervised_last}"
        )
        return

    ssl_resume = semisupervised_last if train_cfg.auto_resume and semisupervised_last.exists() else supervised_best
    supervised_best_epoch = checkpoint_epoch(supervised_best)
    ssl_max_epochs = (
        supervised_best_epoch
        + 1
        + train_cfg.resolved_neighbor_semisupervised_max_epochs
    )
    train_neighbor_phase(
        exp_config=exp_config,
        mode="semisupervised",
        fit_dataset_mode="semisupervised",
        max_epochs=ssl_max_epochs,
        ckpt_path=ssl_resume,
    )


def train(exp_config: ExperimentConfig, fully_supervised_stage: int | None = None):
    """
        Main training method for the staged workflow.

        The same entrypoint is used for:

        - semisupervised staged training
        - active-learning staged training
        - fully supervised replay

        The ``training_regime`` field controls which staged regime is used by
        default. Passing ``fully_supervised_stage`` keeps the compatibility path
        for one-stage fully supervised replay.
    """
    train_cfg, _, _ = exp_config.get_configs()
    if fully_supervised_stage is None:
        if train_cfg.training_regime == "semisupervised":
            run_semisupervised_staged_training(exp_config)
            return
        if train_cfg.training_regime == "active_learning":
            run_active_learning_staged_training(exp_config)
            return
        if train_cfg.training_regime == "upper_bound_replay":
            run_fully_supervised_upper_bound(exp_config)
            return
        if train_cfg.training_regime == "neighbor_semisupervised":
            run_neighbor_semisupervised_training(exp_config)
            return
        raise ValueError(f"Unknown training_regime: {train_cfg.training_regime}")
    run_fully_supervised_stage(exp_config, fully_supervised_stage)


def main():
    """CLI entrypoint for staged EPS-Seg training."""
    parser = argparse.ArgumentParser(description="Train EPS-Seg Model")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to experiment configuration YAML file")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file with environment variables")
    parser.add_argument(
        "--fully_supervised_stage",
        type=int,
        default=None,
        help=(
            "If set, train only the requested scheduler stage in fully supervised mode. "
            "This expects the stage scheduler and semisupervised best checkpoint to already exist."
        ),
    )
    
    args = parser.parse_args()
    print("Loading experiment config from:", args.exp_config)
    print("Loading environment variables from:", args.env_file)
    load_dotenv(args.env_file)
    exp_config = ExperimentConfig.from_yaml(args.exp_config)
    train(exp_config, fully_supervised_stage=args.fully_supervised_stage)


if __name__ == "__main__":
    main()
