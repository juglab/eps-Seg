from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from eps_seg.config.base import BaseEPSConfig


class TrainConfig(BaseEPSConfig):
    model_name: str = Field(default="eps_seg_default", description="Name of the model"),
    lr: float = Field(default=3e-5, description="Learning rate")
    lr_patience: int = Field(default=10, description="Patience for learning rate scheduler")
    lr_factor: float = Field(default=0.9, description="Factor for learning rate scheduler")
    lr_min: float = Field(default=1e-12, description="Minimum learning rate")
    weight_decay: float = Field(default=0.0, description="Weight decay for optimizer")
    max_epochs: int = Field(default=1000, description="Maximum number of training epochs")
    batch_size: int = Field(default=256, description="Batch size for training")
    amp: bool = Field(default=True, description="Use mixed precision training")
    gradient_scale: int = Field(default=256, description="Gradient scaling factor")
    max_grad_norm: Optional[float] = Field(default=1.0, description="Maximum gradient norm")
    alpha: float = Field(default=1.0, description="Weight for the inpainting loss")
    beta: float = Field(default=1e-1, description="Weight for the KLD loss")
    gamma: float = Field(default=1.0, description="Weight for the contrastive loss")
    use_wandb: bool = Field(default=True, description="Use Weights and Biases for logging (if key is set in .env file)")
    initial_label_size: int = Field(default=1, description="Initial label size for label size scheduler")
    final_label_size: int = Field(default=10, description="Final label size for label size scheduler")
    initial_mask_size: int = Field(default=1, description="Initial mask size for mask size scheduler")
    final_mask_size: int = Field(default=10, description="Final mask size for mask size scheduler")
    step_interval: int = Field(default=20, description="Step interval for label/mask size scheduler")
