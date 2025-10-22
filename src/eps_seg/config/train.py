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
    early_stopping_patience: int = Field(default=50, description="Patience for early stopping")
    batch_size: int = Field(default=256, description="Batch size for training")
    amp: bool = Field(default=True, description="Use mixed precision training")
    gradient_scale: int = Field(default=256, description="Gradient scaling factor")
    max_grad_norm: Optional[float] = Field(default=1.0, description="Maximum gradient norm")
    alpha: float = Field(default=1.0, description="Weight for the inpainting loss")
    beta: float = Field(default=1e-1, description="Weight for the KLD loss")
    gamma: float = Field(default=1.0, description="Weight for the contrastive loss")
    use_wandb: bool = Field(default=True, description="Use Weights and Biases for logging (if key is set in .env file)")
    log_every_n_steps: int = Field(default=5, description="Logging frequency in steps")
    initial_threshold: float = Field(default=0.50, description="Initial confidence threshold for training in semisupervised mode")
    max_threshold: float = Field(default=0.99, description="Maximum confidence threshold for training in semisupervised mode")
    threshold_increment: float = Field(default=0.005, description="Step size for confidence threshold increase")
    max_radius: float = Field(default=10.0, description="Maximum radius for training in semisupervised mode")
    radius_increment_patience: int = Field(default=10, description="Number of epochs without improvement before increasing radius in semisupervised mode")