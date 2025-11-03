import lightning as L
from eps_seg.modules.lvae import LadderVAE
from eps_seg.config import LVAEConfig
from eps_seg.config.train import TrainConfig
from typing import Literal
import torch 

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

        self.current_threshold = self.train_cfg.initial_threshold if self.train_cfg else 0.5
        self.current_radius = self.train_cfg.initial_radius if self.train_cfg else 5
        # Patience counter for radius increase
        self.current_radius_patience = 0
        self.save_hyperparameters({"model_config": model_cfg.model_dump(), 
                                   "train_config": train_cfg.model_dump() if train_cfg else None})

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
        
        x = x.squeeze(0)
        y = y.squeeze(0)

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

    def training_step(self, batch, batch_idx):
        x, y, z, _ = batch

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

        self.log("train/IP", inpainting_loss.item() * self.train_cfg.alpha, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/IP_unweighted", inpainting_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/KL", kld_loss.item() * self.train_cfg.beta, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/KL_unweighted", kld_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/CL", contrastive_loss.item() * self.train_cfg.gamma, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/CL_unweighted", contrastive_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/CE", cross_entropy_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        outputs["loss"] = total_loss # Needed for Lightning to work with optimizers
        return outputs

    def validation_step(self, batch, batch_idx):
        x, y, z, _ = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = z.squeeze(0)
        # FIXME: What threshold to use during validation?
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

        self.log("val/IP", inpainting_loss.item() * self.train_cfg.alpha, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/IP_unweighted", inpainting_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/KL", kld_loss.item() * self.train_cfg.beta, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/KL_unweighted", kld_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/CL", contrastive_loss.item() * self.train_cfg.gamma, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/CL_unweighted", contrastive_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/CE", cross_entropy_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("val/total_loss", total_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        outputs["loss"] = total_loss # Needed for Lightning to work with optimizers
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        return self.forward(x, y, validation_mode=False, confidence_threshold=0.99)

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