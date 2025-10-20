import pytorch_lightning as pl
from eps_seg.modules.lvae import LadderVAE
from eps_seg.config import LVAEConfig
from eps_seg.config.train import TrainConfig
from eps_seg.train.metrics import compute_unsupervised_metrics
from typing import Literal
import torch 

class LVAEModel(pl.LightningModule):
    def __init__(self, model_config: LVAEConfig, train_config: TrainConfig = None):
        super().__init__()
        self.cfg = model_config
        self.train_cfg = train_config
        self.model: LadderVAE = LadderVAE(model_config)
        self.current_threshold = self.train_cfg.initial_threshold if self.train_cfg else 0.99
        self.register_buffer("current_threshold", torch.tensor(self.current_threshold))
        self.current_label_size = self.train_cfg.initial_label_size if self.train_cfg else 1
        self.register_buffer("current_label_size", torch.tensor(self.current_label_size))
        self.current_mask_size = self.train_cfg.initial_mask_size if self.train_cfg else 1
        self.register_buffer("current_mask_size", torch.tensor(self.current_mask_size))
        # Number of "bad epochs" from EarlyStopping, used for callbacks that are based on patience
        self.current_wait_count = 0
        self.register_buffer("current_wait_count", torch.tensor(self.current_wait_count))

    def forward(self, x, y=None, mask_input=False, confidence_threshold: float = 0.99):
        """
            Forward pass through the LVAE model.

            Args:
                inputs (torch.Tensor): Input tensor.
                labels (torch.Tensor, optional): Labels tensor. Defaults to None.
                mask_input: (bool, optional): Whether to mask the input. Defaults to False. 
                                              Must be true during training/validation.
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("x has nan or inf")
        
        x = x.squeeze(0)
        y = y.squeeze(0)

        return self.model(x, y=y, mask_input=mask_input, confidence_threshold=confidence_threshold)
    
    def setup(self, stage=None):
        # Add data statistics to the model before training or prediction (so that they are saved in checkpoints)
        if stage in ("fit", "predict") and not hasattr(self.model, "data_mean"):
            print(f"Data Statistics not found in LadderVAE model. Retrieving from datamodule...")
            mean, std = self.trainer.datamodule.get_data_statistics()
            self.model.register_buffer("data_mean", torch.as_tensor(mean))
            self.model.register_buffer("data_std", torch.as_tensor(std))
        else:
            print(f"Using existing data statistics in LadderVAE checkpoint.")

    def training_step(self, batch, batch_idx):
        x, y, z, _ = batch
        
        outputs = self.model(x, y, mask_input=True, confidence_threshold=self.current_threshold)
        
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

        if self.model.training_mode == "unsupervised":
            confusion_metrics = compute_unsupervised_metrics(
                batch_size=x.shape[0],
                z=outputs["z"],
                outputs=outputs,
            )
            self.log_dict(confusion_metrics, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train/IP", inpainting_loss.item() * self.train_cfg.alpha, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/KL", kld_loss.item() * self.train_cfg.beta)
        self.log("train/CL", contrastive_loss.item() * self.train_cfg.gamma, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/CE", cross_entropy_loss.item())
        self.log("train/total_loss", total_loss.item(), prog_bar=True, on_step=True, on_epoch=True)

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y, z, _ = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = z.squeeze(0)
        # FIXME: What threshold to use during validation?
        outputs = self.model(x, y, mask_input=True, confidence_threshold=self.current_threshold)

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

        # FIXME: Continue

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        return self.forward(x, y, mask_input=False, confidence_threshold=0.99)  

    def configure_optimizers(self):
        super().configure_optimizers()
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
                "monitor": "val_total_loss",
            },
        }

    def update_mode(self, mode: Literal["supervised", "semisupervised"]):
        print(f"Updating model training mode to: {mode}")
        self.model.update_mode(mode)


    def configure_callbacks(self):
        return super().configure_callbacks()