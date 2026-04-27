import lightning as L
from eps_seg.modules.lvae import LadderVAE
from eps_seg.config import LVAEConfig
from eps_seg.config.train import TrainConfig
from eps_seg.training.logger import log_epoch_dice_scores, log_lvae_step, log_scheduler_stats, log_trainer_state
from typing import Literal
import torch 
from torchmetrics.classification import F1Score
import numpy as np


class LVAEModel(L.LightningModule):
    def __init__(self, model_cfg: LVAEConfig, train_cfg: TrainConfig = None):
        super().__init__()
        self.cfg = model_cfg
        self.train_cfg = train_cfg
        self.model: LadderVAE = LadderVAE(model_cfg)
        self.current_training_mode = "supervised"

        # Placeholders for data statistics
        self.model.register_buffer("data_mean", torch.tensor(0.0))
        self.model.register_buffer("data_std", torch.tensor(0.0))        
        self.register_buffer("seen_samples", torch.zeros(1, dtype=torch.long))

        self.save_hyperparameters({"model_config": model_cfg.model_dump(), 
                                   "train_config": train_cfg.model_dump() if train_cfg else None})
        
        # DiceScore implemented as F1Score.
        # Under DDP we want the final epoch metric to be computed from the joint
        # TP/FP/TN/FN statistics across ranks, so we let TorchMetrics sync on
        # compute and avoid per-step synchronization.
        self.train_dice_score = F1Score(
            num_classes=self.cfg.n_components,
            average=None,
            task="multiclass",
            ignore_index=-1,
            sync_on_compute=True,
            dist_sync_on_step=False,
        )
        self.validation_dice_score = F1Score(
            num_classes=self.cfg.n_components,
            average=None,
            task="multiclass",
            ignore_index=-1,
            sync_on_compute=True,
            dist_sync_on_step=False,
        )
        self.test_dice_score = F1Score(
            num_classes=self.cfg.n_components,
            average=None,
            task="multiclass",
            ignore_index=-1,
            sync_on_compute=True,
            dist_sync_on_step=False,
        )
        self.current_true_epoch = 0
        self.current_stage_idx = -1
        self.best_val_dice_score_mean = float("-inf")

    def forward(self, x, y=None, validation_mode: bool = False, confidence_threshold: float = 0.99):
        """
            Forward pass through the LVAE model.

            Args:
                inputs (torch.Tensor): Input tensor.
                labels (torch.Tensor, optional): Labels tensor. Defaults to None.
                validation_mode (bool, optional): Whether we are in validation mode 
                                                  (used to distinguish between validation and prediction). 
                                                  Controls whether to mask input or not and compute losses.
                                                  Defaults to False.
                confidence_threshold (float, optional): Confidence threshold for assigning pseudo-labels. 
                                                       Defaults to 0.99.
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("x has nan or inf")
        
        return self.model(x, y=y, validation_mode=validation_mode, confidence_threshold=confidence_threshold)

    def on_fit_start(self):
        # Add data statistics to the model before training or prediction (so that they are saved in checkpoints)
        # TODO: Fix this with a better method
        if self.model.data_std.sum() == 0:
            print("Data Statistics not found. Retrieving from datamodule...")
            mean, std = self.trainer.datamodule.get_data_statistics()
            self.model.data_mean = torch.as_tensor(mean, device=self.device)
            self.model.data_std = torch.as_tensor(std, device=self.device)
        else:
            print("Using existing data statistics from checkpoint.")
        print("Seen samples:", self.seen_samples.item())

    def log_step(self, outputs: dict, step: Literal["train", "val", "test"], batch: dict):
        log_lvae_step(self, outputs, step, batch)

    def compute_total_loss(self, outputs: dict):
        return (
            self.train_cfg.alpha * outputs["inpainting_loss"] +
            self.train_cfg.beta * outputs["kl"] +
            self.train_cfg.gamma * outputs["cl"] +
            outputs["ce"]
        )

    def training_step(self, batch, batch_idx):
        x = batch["patch"]
        if self.train_cfg.train_fully_supervised:
            y = batch["gt"]
        elif self.current_training_mode == "semisupervised":
            # Pseudo-labelled rows stay unlabeled during semisupervised training so
            # the LVAE can infer pseudo-labels internally from the stage-0 labelled
            # samples present in the same batch.
            y = batch["label"].clone()
            y[~batch["is_initial_label"].bool()] = -1
        else:
            y = batch["label"]
        
        batch_size = x.shape[0]

        outputs = self.model(
            x,
            y,
            validation_mode=False,
            confidence_threshold=self.train_cfg.pseudolabel_confidence_threshold,
        )
        outputs["loss"] = self.compute_total_loss(outputs)

        self.seen_samples += batch_size * self.trainer.world_size
        self.current_true_epoch = self.trainer.train_dataloader.batch_sampler.current_true_epoch
        self.log_step(outputs, "train", batch)

        # Accumulate metrics for dice loss (it is logged on epoch end)
        preds = torch.argmax(outputs["class_probabilities"], dim=-1)
        self.train_dice_score.update(preds, y)

        return outputs

    def validation_step(self, batch, batch_idx):
        # TODO: For now, validation still uses the old dataloader and batch format
        x, y, s, c = batch
        batch_size = x.shape[0]

        outputs = self.forward(x, 
                             y, 
                             validation_mode=True, 
                             confidence_threshold=self.train_cfg.pseudolabel_confidence_threshold,
                             )
        outputs["loss"] = self.compute_total_loss(outputs)   

        self.log_step(outputs, "val", {"patch": x, "label": y})
        
        # Accumulate metrics for dice loss (it is logged on epoch end)
        preds = torch.argmax(outputs["class_probabilities"], dim=-1)
        self.validation_dice_score.update(preds, y)

        return outputs

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_dice_score.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, s, c, key = batch
        x = (x - self.model.data_mean) / self.model.data_std
        outputs = self.forward(x, 
                             y=None, 
                             validation_mode=False, 
                             confidence_threshold=0.99,
                             )
        outputs["preds"] = torch.argmax(outputs["class_probabilities"], dim=-1)[:, None]  # Add channel dim for compatibility
        outputs["labels"] = y
        outputs["coords"] = c
        outputs["keys"] = key
        self.test_dice_score.update(outputs["preds"].to(y.device), y)
        return outputs

    def on_test_epoch_end(self):
        dice_loss_per_class = self.test_dice_score.compute()
        for class_idx, dice_score in enumerate(dice_loss_per_class):
            self.log(f'test/dice_score_class_{class_idx}', dice_score, prog_bar=True, sync_dist=False)
        self.log('test/dice_score_mean', dice_loss_per_class.mean(), prog_bar=True, sync_dist=False)
        self.test_dice_score.reset()
        super().on_test_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0, normalize=True):
        x, labels, _, coords, key = batch
        if normalize:
            x = (x - self.model.data_mean) / self.model.data_std
        outputs = self.forward(x, y=None, validation_mode=False)
        preds = torch.argmax(outputs["class_probabilities"], dim=-1)[:, None]  # Add channel dim for compatibility
        outputs["labels"] = labels
        outputs["coords"] = coords
        outputs["preds"] = preds
        outputs["keys"] = key
        return outputs

    def on_train_epoch_end(self):
        log_epoch_dice_scores(self, "train", self.train_dice_score, mean_prog_bar=False)

    def on_train_epoch_start(self):
        log_trainer_state(self)
        log_scheduler_stats(self)

    def on_validation_epoch_end(self):
        log_epoch_dice_scores(self, "val", self.validation_dice_score, mean_prog_bar=True)
        if self.trainer.is_global_zero:
            log_trainer_state(self)

    def evaluate_candidate_batch(self, batch: dict):
        """
            Evaluate scheduler extension candidates and return predicted labels
            together with their confidence scores.
        """
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            x = batch["patch"].to(self.device, non_blocking=True)
            amp_enabled = bool(self.train_cfg and self.train_cfg.amp and self.device.type == "cuda")
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
            with autocast_context:
                outputs = self.forward(
                    x,
                    y=None,
                    validation_mode=False,
                    confidence_threshold=self.train_cfg.pseudolabel_confidence_threshold,
                )
            probs = outputs["class_probabilities"]
            confidences, predicted_labels = probs.max(dim=-1)
        if was_training:
            self.train()
        return predicted_labels.cpu().numpy().astype(np.int32), confidences.cpu().numpy().astype(np.float32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(),
                                       lr=self.train_cfg.lr, 
                                       weight_decay=self.train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=self.train_cfg.lr_patience, factor=self.train_cfg.lr_factor, min_lr=self.train_cfg.lr_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
            },
        }

    def update_mode(self, mode: Literal["supervised", "semisupervised"]):
        print(f"Updating model training mode to: {mode}")
        self.current_training_mode = mode
        self.model.update_mode(mode)

    def configure_callbacks(self):
        return super().configure_callbacks()
