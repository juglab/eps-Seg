from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    lr: float = 3e-5
    max_epochs: int = 1000
    batch_size: int = 256
    amp: bool = True
    gradient_scale: int = 256
    max_grad_norm: Optional[float] = 1.0
    alpha: float = 1.0
    beta: float = 1e-1
    gamma: float = 1.0
    use_wandb: bool = True
