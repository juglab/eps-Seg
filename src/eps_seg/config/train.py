from dataclasses import dataclass
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field
from eps_seg.config.base import BaseEPSConfig
from eps_seg.config.datasets import BaseEPSDatasetConfig
from eps_seg.config.models import BaseEPSModelConfig, LVAEConfig
from pathlib import Path


class TrainConfig(BaseEPSConfig):
    model_name: str = Field(default="eps_seg_default", description="Name of the model")
    supervised_seed: Union[int, None] = Field(default=None, description="Random seed for supervised training. Does not affect data shuffling if a dataset seed is provided. See config.dataset.")
    semisupervised_seed: Union[int, None] = Field(default=None, description="Random seed for semisupervised training.")
    deterministic: bool = Field(default=False, description="Whether to use deterministic training (may slow down training but ensures reproducibility)")
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
    initial_radius: int = Field(default=5, description="Initial radius for training in semisupervised mode")
    max_radius: int = Field(default=10, description="Maximum radius for training in semisupervised mode")
    radius_increment_patience: int = Field(default=10, description="Number of epochs without improvement before increasing radius in semisupervised mode")


class ExperimentConfig(BaseEPSConfig):
    project_name: str = Field(default="eps-seg-default-project", description="Name of the project, e.g. used in WandB logging")
    train_cfg_path: str = Field(default=None, description="Path to the training configuration YAML file. Can be either absolute or relative to the experiment config file")
    dataset_cfg_path: str = Field(description="Path to the dataset configuration YAML file. Can be either absolute or relative to the experiment config file")
    model_cfg_path: str = Field(default=None, description="Path to the model configuration YAML file. Can be either absolute or relative to the experiment config file")

    def get_configs(self) -> tuple[TrainConfig, BaseEPSDatasetConfig, BaseEPSConfig]:
        """Load and return the training, dataset, and model configurations objects from the provided paths."""

        # Paths can be either absolute or relative to the experiment config file

        if self.train_cfg_path:
            train_cfg_path = Path(self.train_cfg_path)
            if not train_cfg_path.is_absolute():
                train_cfg_path = Path(self.config_yaml_path).parent / train_cfg_path

            train_cfg = TrainConfig.from_yaml(train_cfg_path)
            print(f"Loaded training config from {train_cfg_path}")
        else:
            train_cfg = TrainConfig()
            print("Using default training config")


        dataset_cfg_path = Path(self.dataset_cfg_path)
        if not dataset_cfg_path.is_absolute():
            dataset_cfg_path = Path(self.config_yaml_path).parent / dataset_cfg_path
        dataset_cfg = BaseEPSDatasetConfig.from_yaml(dataset_cfg_path)

        if self.model_cfg_path:
            model_cfg_path = Path(self.model_cfg_path)
            if not model_cfg_path.is_absolute():
                model_cfg_path = Path(self.config_yaml_path).parent / model_cfg_path
            model_cfg = BaseEPSModelConfig.from_yaml(model_cfg_path)
            print(f"Loaded model config from {model_cfg_path}")
        else:
            model_cfg = LVAEConfig()
            print("Using default LVAE config")

        return train_cfg, dataset_cfg, model_cfg

    @property
    def experiment_root(self) -> Path:
        """Return the root directory of the experiment based on the config YAML path."""
        if self.config_yaml_path is None:
            raise ValueError("config_yaml_path is not set.")
        return Path(self.config_yaml_path).parent
    
    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment based on the config YAML file name."""
        if self.config_yaml_path is None:
            raise ValueError("config_yaml_path is not set.")
        return Path(self.config_yaml_path).stem
    
    @property
    def checkpoints_dir(self) -> Path:
        """Return the directory path for saving checkpoints."""
        return self.experiment_root / "checkpoints"
    
    @property
    def logs_dir(self) -> Path:
        """Return the directory path for saving logs."""
        return self.experiment_root / "logs"

    def best_checkpoint_path(self, mode: Literal["supervised", "semisupervised"]) -> Path:
        """Return the path to the best model checkpoint based on the training mode."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()

        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"best_{mode}.ckpt"
    
    def get_log_dir(self) -> Path:
        """Return the directory path for saving logs."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()

        return self.logs_dir.resolve() / self.experiment_name / train_cfg.model_name