import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def _get_model_checkpoint_callback(module, monitor: str = "val/CE_epoch"):
    for callback in module.trainer.callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.monitor == monitor:
            return callback
    return None


def _get_early_stopping_callback(module, monitor: str = "val/CE_epoch"):
    for callback in module.trainer.callbacks:
        if isinstance(callback, EarlyStopping) and callback.monitor == monitor:
            return callback
    return None


def log_lvae_step(module, outputs: dict, step: str, batch: dict):
    batch_size = batch["patch"].size(0)
    module.log(f"{step}/IP", outputs["inpainting_loss"] * module.train_cfg.alpha, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    module.log(f"{step}/IP_unweighted", outputs["inpainting_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    for loss_term_name, weight in zip(["kl", "cl", "ce"], [module.train_cfg.beta, module.train_cfg.gamma, 1.0]):
        module.log(f"{step}/{loss_term_name.upper()}", outputs[loss_term_name] * weight, prog_bar=loss_term_name == "ce", on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    for layer_idx, value in enumerate(outputs["kl_per_layer"]):
        module.log(f"{step}/KL_layer_{layer_idx}", value * module.train_cfg.beta, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        module.log(f"{step}/KL_layer_{layer_idx}_unweighted", value, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    module.log(f"{step}/total_loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    module.log("seen_samples", float(module.seen_samples), prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, reduce_fx="max")
    module.log("true_epoch", module.current_true_epoch, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="max")
    if step == "train" and module.current_training_mode == "semisupervised" and not module.train_cfg.train_fully_supervised:
        log_pseudo_label_stats(module, outputs, batch, batch_size)


def log_pseudo_label_stats(module, outputs: dict, batch: dict, batch_size: int):
    gt = batch.get("gt")
    is_initial_label = batch.get("is_initial_label")
    stage_index = batch.get("stage_index")
    probs = outputs.get("class_probabilities")
    pseudo_labels = outputs.get("pseudo_labels")
    if gt is None or is_initial_label is None or stage_index is None or probs is None or pseudo_labels is None:
        return

    is_initial_label = is_initial_label.bool()
    pseudo_mask = ~is_initial_label
    preds = torch.argmax(probs, dim=-1)
    module.log("pseudo_labels/gt_amount", is_initial_label.float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    module.log("pseudo_labels/pseudo_amount", pseudo_mask.float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

    if is_initial_label.any():
        module.log("pseudo_labels/gt_accuracy", (preds[is_initial_label] == gt[is_initial_label]).float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
    if pseudo_mask.any():
        pseudo_label_confidences = probs[pseudo_mask].gather(1, pseudo_labels[pseudo_mask].clamp(min=0).unsqueeze(1)).squeeze(1)
        module.log("pseudo_labels/accuracy", (pseudo_labels[pseudo_mask] == gt[pseudo_mask]).float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        module.log("pseudo_labels/avg_confidence", pseudo_label_confidences.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for stage in torch.unique(stage_index[pseudo_mask]).tolist():
            stage = int(stage)
            stage_mask = pseudo_mask & (stage_index == stage)
            if not stage_mask.any():
                continue
            stage_confidences = probs[stage_mask].gather(1, pseudo_labels[stage_mask].clamp(min=0).unsqueeze(1)).squeeze(1)
            module.log(f"pseudo_labels/stage_{stage}_accuracy", (pseudo_labels[stage_mask] == gt[stage_mask]).float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            module.log(f"pseudo_labels/stage_{stage}_amount", stage_mask.float().mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            module.log(f"pseudo_labels/stage_{stage}_avg_confidence", stage_confidences.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)


def log_scheduler_stats(module):
    datamodule = getattr(module.trainer, "datamodule", None)
    train_dataset = getattr(datamodule, "train_dataset", None)
    schedule = getattr(train_dataset, "schedule", None)
    if schedule is None or "name_id" not in schedule:
        return

    total_samples = int(len(schedule["name_id"]))
    if total_samples == 0:
        return

    enabled_mask = schedule["is_enabled"] if "is_enabled" in schedule else np.ones(total_samples, dtype=np.bool_)
    active_samples = int(enabled_mask.sum())
    disabled_samples = int(total_samples - active_samples)
    avg_active_confidence = float(schedule["confidence"][enabled_mask].mean()) if active_samples > 0 else 0.0
    module.log("scheduler/active_samples", float(active_samples), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    module.log("scheduler/disabled_samples", float(disabled_samples), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    module.log("scheduler/total_samples", float(total_samples), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    module.log("scheduler/avg_active_confidence", avg_active_confidence, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


def log_trainer_state(module):
    module.log("trainer/current_stage", float(module.current_stage_idx), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    early_stopping_callback = _get_early_stopping_callback(module)
    if early_stopping_callback is not None:
        module.log("trainer/early_stopping_wait_count", float(early_stopping_callback.wait_count), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        module.log("trainer/early_stopping_patience", float(early_stopping_callback.patience), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    if module.trainer.lr_scheduler_configs:
        scheduler = module.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, "num_bad_epochs"):
            module.log("trainer/lr_scheduler_wait_count", float(scheduler.num_bad_epochs), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if hasattr(scheduler, "patience"):
            module.log("trainer/lr_scheduler_patience", float(scheduler.patience), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    checkpoint_callback = _get_model_checkpoint_callback(module)
    if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None:
        best_score = checkpoint_callback.best_model_score
        if hasattr(best_score, "item"):
            best_score = float(best_score.item())
        module.log("trainer/best_val_CE_epoch", float(best_score), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)


def log_epoch_dice_scores(module, split: str, dice_metric, mean_prog_bar: bool):
    if not module.trainer.is_global_zero:
        dice_metric.reset()
        return

    dice_per_class = dice_metric.compute()
    for class_idx, dice_score in enumerate(dice_per_class):
        module.log(f"{split}/dice_score_class_{class_idx}", dice_score, prog_bar=False, sync_dist=False)
    dice_mean = dice_per_class.mean()
    module.log(f"{split}/dice_score_mean", dice_mean, prog_bar=mean_prog_bar, sync_dist=False)
    if split == "val":
        module.best_val_dice_score_mean = max(module.best_val_dice_score_mean, float(dice_mean))
        module.log("trainer/best_val_dice_score_mean", module.best_val_dice_score_mean, prog_bar=False, sync_dist=False)
    dice_metric.reset()
