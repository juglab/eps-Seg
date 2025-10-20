import pytorch_lightning as pl
from eps_seg.modules.lvae import LadderVAE
from eps_seg.config import LVAEConfig
from eps_seg.config.train import TrainConfig
import torch 

class LVAEModel(pl.LightningModule):
    def __init__(self, model_config: LVAEConfig, train_config: TrainConfig = None):
        super().__init__()
        self.cfg = model_config
        self.train_cfg = train_config
        self.model: LadderVAE = LadderVAE(model_config)

    def forward(self, x, y=None, mask_input=False, confidence_threshold: float = 0.99):
        """
            Forward pass through the LVAE model.

            Args:
                inputs (torch.Tensor): Input tensor.
                labels (torch.Tensor, optional): Labels tensor. Defaults to None.
                mask_input: (bool, optional): Whether to mask the input. Defaults to False. 
                                              Must be true during training/validation.
        """
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
        x, y = batch
        outputs = self.model(x, y, mask_input=True)
        # FIXME: Continue

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x, y, mask_input=True)
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

        


    def configure_callbacks(self):
        return super().configure_callbacks()