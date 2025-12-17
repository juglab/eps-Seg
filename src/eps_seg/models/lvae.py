import lightning as L
from eps_seg.modules.lvae import LadderVAE
from eps_seg.config import LVAEConfig
from eps_seg.config.train import TrainConfig
from typing import Literal
import torch 
from torchmetrics.classification import F1Score


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

        self.current_threshold = self.train_cfg.initial_threshold if self.train_cfg else 0.5
        self.current_radius = self.train_cfg.initial_radius if self.train_cfg else 5
        # Patience counter for radius increase
        self.current_radius_patience = 0
        self.save_hyperparameters({"model_config": model_cfg.model_dump(), 
                                   "train_config": train_cfg.model_dump() if train_cfg else None})
        
        # DiceScore implemented as F1Score
        # Index -1 is passed during selfsupervised mode for inpatinting loss on unlabeled regions
        # sync_on_compute=False because we want to accumulate stats across devices manually and then compute at epoch end only on rank 0
        # otherwise it will go deadlock because we end up with different class amounts on different devices
        self.train_dice_score = F1Score(num_classes=self.cfg.n_components, average=None, task="multiclass", ignore_index=-1, sync_on_compute=False, dist_sync_on_step=True) 
        self.validation_dice_score = F1Score(num_classes=self.cfg.n_components, average=None, task="multiclass", ignore_index=-1, sync_on_compute=False, dist_sync_on_step=True)
        self.test_dice_score = F1Score(num_classes=self.cfg.n_components, average=None, task="multiclass", ignore_index=-1, sync_on_compute=False, dist_sync_on_step=True)
        self.current_true_epoch = 0

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

    def training_step(self, batch, batch_idx):
        x, y, z, _ = batch
        batch_size = x.shape[0]

        outputs = self.model(x, y, validation_mode=False, confidence_threshold=self.current_threshold)

        inpainting_loss = outputs["inpainting_loss"]
        kld_loss = outputs["kl"]
        contrastive_loss = outputs["cl"]
        cross_entropy_loss = outputs["cross_entropy"]

        total_loss = (
            self.train_cfg.alpha * inpainting_loss +
            self.train_cfg.beta * kld_loss +
            self.train_cfg.gamma * contrastive_loss +
            cross_entropy_loss
        )

        self.seen_samples += batch_size * self.trainer.world_size

        self.log("train/IP", inpainting_loss * self.train_cfg.alpha, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/IP_unweighted", inpainting_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/KL", kld_loss * self.train_cfg.beta, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/KL_unweighted", kld_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/CL", contrastive_loss * self.train_cfg.gamma, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/CL_unweighted", contrastive_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/CE", cross_entropy_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("seen_samples", self.seen_samples, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, reduce_fx="max")
        
        self.current_true_epoch = self.trainer.train_dataloader.batch_sampler.current_true_epoch
        self.log("true_epoch", self.current_true_epoch, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="max")
        outputs["loss"] = total_loss # Needed for Lightning to work with optimizers
        # Accumulate metrics for dice loss (it is logged on epoch end)
        preds = torch.argmax(outputs["class_probabilities"], dim=-1)
        self.train_dice_score.update(preds, y)

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y, z, _ = batch

        outputs = self.forward(x, 
                             y, 
                             validation_mode=True, 
                             confidence_threshold=self.current_threshold,
                             )

        inpainting_loss = outputs["inpainting_loss"]
        kld_loss = outputs["kl"]
        cross_entropy_loss = outputs["cross_entropy"]
        contrastive_loss = (
            outputs["cl"] if not torch.isnan(outputs["cl"]) else torch.tensor(0.0, device=self.device)
        )

        total_loss = (
            self.train_cfg.alpha * inpainting_loss +
            self.train_cfg.beta * kld_loss +
            self.train_cfg.gamma * contrastive_loss +
            cross_entropy_loss
        )

        # Log losses
        self.log("val/IP", inpainting_loss * self.train_cfg.alpha, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/IP_unweighted", inpainting_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/KL", kld_loss * self.train_cfg.beta, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/KL_unweighted", kld_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/CL", contrastive_loss * self.train_cfg.gamma, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/CL_unweighted", contrastive_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/CE", cross_entropy_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("seen_samples", self.seen_samples, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, reduce_fx="max")
        self.log("true_epoch", self.current_true_epoch, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="max")

        outputs["loss"] = total_loss # Needed for Lightning to work with optimizers

        # Accumulate metrics for dice loss (it is logged on epoch end)
        preds = torch.argmax(outputs["class_probabilities"], dim=-1)
        self.validation_dice_score.update(preds, y)

        return outputs

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.current_test_outputs = []
        self.test_dice_score.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, s, c = batch
        x = (x - self.model.data_mean) / self.model.data_std
        outputs = self.forward(x, 
                             y=None, 
                             validation_mode=False, 
                             confidence_threshold=0.99,
                             )
        outputs["preds"] = torch.argmax(outputs["class_probabilities"], dim=-1)[:, None]  # Add channel dim for compatibility
        outputs["labels"] = y
        outputs["coords"] = c
        self.current_test_outputs.append(outputs)
        self.test_dice_score.update(outputs["preds"].to(y.device), y)
        return outputs

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            # We are node 0 device 0
            dice_loss_per_class = self.test_dice_score.compute()
            for class_idx, dice_score in enumerate(dice_loss_per_class):
                self.log(f'test/dice_score_class_{class_idx}', dice_score, prog_bar=True, sync_dist=False)
            self.log('test/dice_score_mean', dice_loss_per_class.mean(), prog_bar=True, sync_dist=False)
            self.test_dice_score.reset()
        super().on_test_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0, normalize=True):
        x, labels, _, coords = batch
        if normalize:
            x = (x - self.model.data_mean) / self.model.data_std
        outputs = self.forward(x, y=None, validation_mode=False)
        preds = torch.argmax(outputs["class_probabilities"], dim=-1)[:, None]  # Add channel dim for compatibility
        outputs["labels"] = labels
        outputs["coords"] = coords
        outputs["preds"] = preds
        return outputs

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            # We are node 0 device 0
            dice_loss_per_class = self.train_dice_score.compute()
            for class_idx, dice_score in enumerate(dice_loss_per_class):
                self.log(f'train/dice_score_class_{class_idx}', dice_score, prog_bar=True, sync_dist=False)
            self.log('train/dice_score_mean', dice_loss_per_class.mean(), prog_bar=True, sync_dist=False)
        self.train_dice_score.reset()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            # We are node 0 device 0
            dice_loss_per_class = self.validation_dice_score.compute()
            for class_idx, dice_score in enumerate(dice_loss_per_class):
                self.log(f'val/dice_score_class_{class_idx}', dice_score, prog_bar=True, sync_dist=False)
            self.log('val/dice_score_mean', dice_loss_per_class.mean(), prog_bar=True, sync_dist=False)
        self.validation_dice_score.reset()


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