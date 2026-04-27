from typing import Any, Dict, Optional, Union, Literal
from pydantic import Field, model_validator
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
    max_epochs: int = Field(default=1000, description="Maximum number of training (pseudo) epochs")
    early_stopping_patience: int = Field(default=100, description="Patience for early stopping")
    batch_size: int = Field(default=128, description="Batch size for training")
    batches_per_pseudoepoch: Union[int, None] = Field(default=100, description="Number of batches per pseudo-epoch. If None, it is set to dataset_size / batch_size. Must be divisible by num_gpus.")
    test_batch_size: int = Field(default=512, description="Batch size for testing/inference")
    amp: bool = Field(default=True, description="Use mixed precision training")
    gradient_scale: int = Field(default=256, description="Gradient scaling factor")
    max_grad_norm: Optional[float] = Field(default=1.0, description="Maximum gradient norm")
    alpha: float = Field(default=1.0, description="Weight for the inpainting loss")
    beta: float = Field(default=1e-2, description="Weight for the KLD loss")
    gamma: float = Field(default=0.1, description="Weight for the contrastive loss")
    monitored_metric: str = Field(default="val/dice_score_mean",description="Validation metric used for model checkpointing, early stopping, and staged best-score carry-over.",)
    monitored_metric_mode: Literal["min", "max"] = Field(default="max", description="Optimization direction for the monitored metric. Use 'max' for scores such as Dice and 'min' for losses such as CE.",)
    use_wandb: bool = Field(default=True, description="Use Weights and Biases for logging (if key is set in .env file)")
    log_every_n_steps: int = Field(default=1, description="Logging frequency in steps")
    accumulate_grad_batches: int = Field(default=1, description="Number of batches to accumulate gradients over before performing an optimizer step. Useful for simulating larger batch sizes with limited GPU memory.")
    train_fully_supervised: bool = Field(default=False, description="Whether to use ground truth labels also for pseudo-labeled samples. Used to set the upper bound of the performances achievable by the model.")
    pseudolabel_confidence_threshold: float = Field(default=0.75, description="Minimum confidence required to accept a newly sampled pseudo-label.")
    pseudolabels_per_extension: int = Field(default=100000, description="Number of pseudo-labels to add whenever the scheduler is extended.")
    enable_pseudolabel_pruning: bool = Field(default=False, description="Whether to disable active pseudo-labels that repeatedly fail re-evaluation at stage transitions.")
    pseudolabel_keep_threshold: float = Field(default=0.75, description="Minimum confidence required to keep an active pseudo-label enabled during stage re-evaluation.")
    pseudolabel_pruning_patience: int = Field(default=1, description="Number of consecutive failed stage re-evaluations before a pseudo-label is disabled.")
    max_extensions: int = Field(default=0, description="Maximum number of scheduler extensions after the initial labelled stage.")
    min_initial_label_fraction: float = Field(default=0.25, description="Minimum fraction of each training batch that must come from the initial GT-labelled pool.")
    auto_resume: bool = Field(default=True, description="Automatically resume from the latest completed staged checkpoint if present.")
    initial_threshold: float = Field(default=1.0, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    max_threshold: float = Field(default=0.70, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    threshold_increment: float = Field(default=0.01, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    threshold_decrement_patience: int = Field(default=5, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    initial_radius: int = Field(default=3, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    max_radius: int = Field(default=7, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    radius_increment_patience: int = Field(default=20, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")
    pseudolabel_age_for_election: int = Field(default=10, description="Deprecated legacy option for the previous curriculum scheduler. Ignored by staged training.")

    @model_validator(mode="after")
    def validate_staged_training(self):
        if not self.monitored_metric:
            raise ValueError("monitored_metric must be a non-empty string.")
        if not 0.0 <= self.min_initial_label_fraction <= 1.0:
            raise ValueError("min_initial_label_fraction must be between 0 and 1.")
        if self.pseudolabel_confidence_threshold < 0.0 or self.pseudolabel_confidence_threshold > 1.0:
            raise ValueError("pseudolabel_confidence_threshold must be between 0 and 1.")
        if self.pseudolabel_keep_threshold < 0.0 or self.pseudolabel_keep_threshold > 1.0:
            raise ValueError("pseudolabel_keep_threshold must be between 0 and 1.")
        if self.pseudolabels_per_extension < 0:
            raise ValueError("pseudolabels_per_extension must be >= 0.")
        if self.pseudolabel_pruning_patience < 1:
            raise ValueError("pseudolabel_pruning_patience must be >= 1.")
        if self.max_extensions < 0:
            raise ValueError("max_extensions must be >= 0.")
        return self



class ExperimentConfig(BaseEPSConfig):
    project_name: str = Field(default="eps-seg-default-project", description="Name of the project, e.g. used in WandB logging")
    train_cfg_path: str = Field(default=None, description="Path to the training configuration YAML file. Can be either absolute or relative to the experiment config file")
    dataset_cfg_path: str = Field(description="Path to the dataset configuration YAML file. Can be either absolute or relative to the experiment config file")
    dataset_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional dataset configuration overrides applied after loading the dataset YAML. Useful for fold-specific experiment files.",
    )
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
        if self.dataset_overrides:
            merged_dataset_cfg = {
                **dataset_cfg.model_dump(exclude={"config_yaml_path"}),
                **self.dataset_overrides,
            }
            dataset_cfg = type(dataset_cfg)(**merged_dataset_cfg)
            dataset_cfg.config_yaml_path = dataset_cfg_path
            print(
                f"Loaded dataset config from {dataset_cfg_path} "
                f"with overrides {self.dataset_overrides}"
            )
        else:
            print(f"Loaded dataset config from {dataset_cfg_path}")

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
    def outputs_dir(self) -> Path:
        """Return the directory path for saving outputs."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()
        return self.experiment_root / "outputs" / self.experiment_name / train_cfg.model_name 
    
    @property
    def results_csv_path(self) -> Path:
        """Return the directory path for saving results."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()
        return self.experiment_root / "results" / self.experiment_name / train_cfg.model_name / "results.csv"

    @property
    def logs_dir(self) -> Path:
        """Return the directory path for saving logs."""
        return self.experiment_root / "logs"

    def best_checkpoint_path(self, mode: Literal["supervised", "semisupervised"]) -> Path:
        """Return the path to the best model checkpoint based on the training mode."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()

        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"best_{mode}.ckpt"

    def stage_checkpoint_path(self, mode: Literal["supervised", "semisupervised"], stage_idx: int, kind: Literal["best", "last"]) -> Path:
        train_cfg, _, _ = self.get_configs()
        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"{kind}_{mode}_K{stage_idx}.ckpt"

    def stage_scheduler_path(self, stage_idx: int) -> Path:
        train_cfg, _, _ = self.get_configs()
        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"scheduler_K{stage_idx}.npz"
    
    def get_log_dir(self) -> Path:
        """Return the directory path for saving logs."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()

        return self.logs_dir.resolve() / self.experiment_name / train_cfg.model_name
