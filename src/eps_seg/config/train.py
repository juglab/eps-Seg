from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from pydantic import Field, model_validator

from eps_seg.config.base import BaseEPSConfig
from eps_seg.config.datasets import BaseEPSDatasetConfig
from eps_seg.config.models import BaseEPSModelConfig, LVAEConfig


class CandidateSamplingConfig(BaseEPSConfig):
    """Configuration for candidate voxel sampling during scheduler extension.

    Args:
        name: Name of the candidate sampling strategy.

    Returns:
        CandidateSamplingConfig: Validated candidate sampling configuration.
    """

    name: Literal["uniform_coordinate_sampling", "class_balanced_substack"] = Field(
        default="uniform_coordinate_sampling",
        description="Candidate voxel sampling strategy used during staged scheduler extension.",
    )


class ScheduleAdmissionConfig(BaseEPSConfig):
    """Configuration for scheduler-row admission into the staged scheduler.

    Args:
        name: Name of the admission policy.
        confidence_min: Minimum confidence required to admit a candidate.
        confidence_max: Maximum confidence allowed to admit a candidate.

    Returns:
        ScheduleAdmissionConfig: Validated schedule admission configuration.
    """

    name: Literal[
        "confidence_threshold_with_pseudolabels",
        "admit_all_with_gt",
        "confidence_window_with_gt",
        "confidence_percentile_window_with_gt",
        "reconstruction_error_with_gt",
    ] = Field(
        default="confidence_threshold_with_pseudolabels",
        description="Policy used to admit evaluated stage-extension candidates into the scheduler.",
    )
    score_metric: Literal[
        "max_probability",
        "normalized_reciprocal_entropy",
        "margin",
        "reconstruction_error",
        "inpainting_error",
    ] = Field(
        default="max_probability",
        description="Model-derived score used by admission policies.",
    )
    candidate_order: Literal["fifo", "ascending", "descending"] = Field(
        default="fifo",
        description="Ordering applied to evaluated candidates before admission.",
    )
    confidence_min: float = Field(
        default=0.75,
        description="Minimum score required to accept a newly sampled candidate.",
    )
    confidence_max: float = Field(
        default=1.0,
        description="Maximum score allowed to accept a newly sampled candidate.",
    )
    confidence_percentile_min: float = Field(
        default=0.0,
        description="Minimum evaluated-pool score percentile allowed by percentile-window admission policies.",
    )
    confidence_percentile_max: float = Field(
        default=100.0,
        description="Maximum evaluated-pool score percentile allowed by percentile-window admission policies.",
    )

    @model_validator(mode="after")
    def validate_confidence_window(self) -> "ScheduleAdmissionConfig":
        """Validate the configured confidence window.

        Args:
            None

        Returns:
            ScheduleAdmissionConfig: The validated configuration.
        """

        if not 0.0 <= self.confidence_min <= 1.0:
            raise ValueError("schedule_admission.confidence_min must be between 0 and 1.")
        if not 0.0 <= self.confidence_max <= 1.0:
            raise ValueError("schedule_admission.confidence_max must be between 0 and 1.")
        if self.confidence_min > self.confidence_max:
            raise ValueError("schedule_admission.confidence_min must be <= confidence_max.")
        if not 0.0 <= self.confidence_percentile_min <= 100.0:
            raise ValueError("schedule_admission.confidence_percentile_min must be between 0 and 100.")
        if not 0.0 <= self.confidence_percentile_max <= 100.0:
            raise ValueError("schedule_admission.confidence_percentile_max must be between 0 and 100.")
        if self.confidence_percentile_min > self.confidence_percentile_max:
            raise ValueError("schedule_admission.confidence_percentile_min must be <= confidence_percentile_max.")
        return self


class PseudolabelMaintenanceConfig(BaseEPSConfig):
    """Configuration for pseudo-label reevaluation and pruning.

    Args:
        name: Name of the maintenance policy.
        enable_pruning: Whether failed pseudo-labels should be disabled.
        pruning_patience: Number of failed reevaluations tolerated before disabling a row.

    Returns:
        PseudolabelMaintenanceConfig: Validated pseudo-label maintenance configuration.
    """

    name: Literal["pruning", "noop"] = Field(
        default="pruning",
        description="Policy used to reevaluate and optionally disable active pseudo-labels between stages.",
    )
    enable_pruning: bool = Field(
        default=False,
        description="Whether to disable active pseudo-labels that repeatedly fail re-evaluation at stage transitions.",
    )
    pruning_patience: int = Field(
        default=1,
        description="Number of consecutive failed stage re-evaluations before a pseudo-label is disabled.",
    )

    @model_validator(mode="after")
    def validate_pruning(self) -> "PseudolabelMaintenanceConfig":
        """Validate the configured pruning options.

        Args:
            None

        Returns:
            PseudolabelMaintenanceConfig: The validated configuration.
        """

        if self.pruning_patience < 1:
            raise ValueError("pseudolabel_maintenance.pruning_patience must be >= 1.")
        return self


class InitialLabelSamplingConfig(BaseEPSConfig):
    """Configuration for stage-0 initial labelled voxel selection.

    Args:
        name: Name of the initial label sampling strategy.

    Returns:
        InitialLabelSamplingConfig: Validated initial label sampling configuration.
    """

    name: Literal["class_balanced_slice", "class_balanced_substack"] = Field(
        description="Strategy used to construct the initial labelled scheduler at stage K0.",
    )


class TrainConfig(BaseEPSConfig):
    training_regime: Literal[
        "semisupervised",
        "active_learning",
        "upper_bound_replay",
        "neighbor_semisupervised",
    ] = Field(
        default="semisupervised",
        description="High-level staged training regime controlling how scheduler rows are interpreted and extended.",
    )
    model_name: str = Field(default="eps_seg_default", description="Name of the model")
    supervised_seed: Union[int, None] = Field(default=None, description="Random seed for supervised training. Does not affect data shuffling if a dataset seed is provided. See config.dataset.")
    semisupervised_seed: Union[int, None] = Field(default=None, description="Random seed for semisupervised training.")
    deterministic: bool = Field(default=True, description="Whether to use deterministic training (may slow down training but ensures reproducibility)")
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
    monitored_metric: str = Field(default="val/CE_epoch", description="Validation metric used for model checkpointing, early stopping, and staged best-score carry-over.")
    monitored_metric_mode: Literal["min", "max"] = Field(default="min", description="Optimization direction for the monitored metric. Use 'max' for scores such as Dice and 'min' for losses such as CE.")
    use_wandb: bool = Field(default=True, description="Use Weights and Biases for logging (if key is set in .env file)")
    log_every_n_steps: int = Field(default=1, description="Logging frequency in steps")
    accumulate_grad_batches: int = Field(default=1, description="Number of batches to accumulate gradients over before performing an optimizer step. Useful for simulating larger batch sizes with limited GPU memory.")
    train_fully_supervised: bool = Field(default=False, description="Whether to use ground truth labels also for pseudo-labeled samples. Used to set the upper bound of the performances achievable by the model.")
    model_confidence_threshold: float = Field(
        default=0.75,
        description="Confidence threshold passed to the LVAE when it internally reasons over unlabeled samples.",
    )
    pseudo_label_confidence_direction: Literal["above", "below"] = Field(
        default="above",
        description=(
            "Direction used by internal pseudo-label confidence gating. "
            "'above' keeps high-confidence pseudo-labels; 'below' keeps low-confidence pseudo-labels."
        ),
    )
    model_confidence_threshold_end: Optional[float] = Field(
        default=None,
        description=(
            "Optional final confidence threshold for semisupervised threshold annealing. "
            "When omitted, model_confidence_threshold stays static."
        ),
    )
    model_confidence_threshold_step: float = Field(
        default=0.005,
        description="Absolute threshold step used when annealing toward model_confidence_threshold_end.",
    )
    model_confidence_threshold_step_epochs: int = Field(
        default=1,
        description="Number of semisupervised epochs between threshold annealing steps.",
    )
    validation_use_pseudolabel_neighbors: bool = Field(
        default=False,
        description=(
            "Whether neighbor-based semisupervised validation should include local "
            "unlabeled neighbors and treat the model's hard pseudo-labels as targets "
            "for validation losses."
        ),
    )
    validation_variant: Literal[
        "anchor_gt",
        "spatial_pl",
        "spatial_gt",
        "random_pl",
        "random_gt",
    ] = Field(
        default="anchor_gt",
        description=(
            "Validation loss target/source variant for neighbor semisupervised "
            "training. The legacy validation_use_pseudolabel_neighbors=True flag "
            "maps to spatial_pl when this is left at the default."
        ),
    )
    mask_input_during_prediction: bool = Field(
        default=False,
        description=(
            "Whether prediction should run the LVAE with the same center-input masking "
            "used during training/validation. This does not enable validation losses."
        ),
    )
    candidate_sampling: CandidateSamplingConfig = Field(
        default_factory=CandidateSamplingConfig,
        description="Candidate voxel sampling strategy used during staged scheduler extension.",
    )
    schedule_admission: ScheduleAdmissionConfig = Field(
        default_factory=ScheduleAdmissionConfig,
        description="Admission policy used during staged scheduler extension.",
    )
    pseudolabel_maintenance: PseudolabelMaintenanceConfig = Field(
        default_factory=PseudolabelMaintenanceConfig,
        description="Pseudo-label maintenance policy used between stages.",
    )
    initial_label_sampling: Optional[InitialLabelSamplingConfig] = Field(
        default=None,
        description="Optional stage-0 initial-label sampling strategy. When omitted, it is inferred from the available dataset metadata.",
    )
    extension_samples_per_class: Optional[Dict[int, int]] = Field(
        default=None,
        description="Optional per-class budget used by class-balanced scheduler extension strategies.",
    )
    rows_per_extension: int = Field(default=100000, description="Number of scheduler rows to add whenever the scheduler is extended.")
    candidate_evaluation_budget: Optional[int] = Field(
        default=None,
        description=(
            "Optional maximum number of newly sampled candidates to evaluate per scheduler extension. "
            "When set, candidate evaluation is capped separately from rows_per_extension, which remains "
            "the label-admission budget."
        ),
    )
    neighbor_radius: int = Field(
        default=5,
        description="Maximum radius used to sample local unlabeled neighbors for neighbor-based semisupervised training.",
    )
    neighbor_samples_per_anchor: int = Field(
        default=1,
        description=(
            "Number of precomputed local neighbors to randomly emit per anchor in each "
            "neighbor-semisupervised batch. A value of 1 gives 50% GT anchors and "
            "50% unlabeled neighbor patches."
        ),
    )
    deterministic_neighbor_selection: bool = Field(
        default=False,
        description=(
            "Whether neighbor-semisupervised training should emit a fixed contiguous "
            "window of stored neighbors instead of randomly sampling per anchor access."
        ),
    )
    neighbor_selection_start_index: int = Field(
        default=0,
        description=(
            "Zero-based first stored-neighbor index used when deterministic neighbor "
            "selection is enabled."
        ),
    )
    neighbor_unlabeled_curriculum_step_size: int = Field(
        default=0,
        description=(
            "If greater than zero, run neighbor semisupervised training as a cumulative "
            "unlabeled-pool curriculum with this many additional unlabeled neighbor "
            "patches per phase."
        ),
    )
    neighbor_unlabeled_curriculum_max_samples: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of unlabeled neighbor patches to expose in the cumulative "
            "neighbor curriculum. Required when neighbor_unlabeled_curriculum_step_size "
            "is greater than zero."
        ),
    )
    neighbor_unlabeled_curriculum_seed: Optional[int] = Field(
        default=None,
        description=(
            "Seed used to deterministically shuffle the full unlabeled neighbor pool "
            "before taking cumulative prefixes. Defaults to the semisupervised seed, "
            "then the dataset seed."
        ),
    )
    neighbor_supervised_max_epochs: Optional[int] = Field(
        default=None,
        description=(
            "Maximum epochs for the supervised warmup phase of neighbor-based semisupervised training. "
            "Defaults to max_epochs when omitted."
        ),
    )
    neighbor_semisupervised_max_epochs: Optional[int] = Field(
        default=None,
        description=(
            "Additional maximum epochs for the semisupervised phase of neighbor-based semisupervised training. "
            "Defaults to max_epochs when omitted."
        ),
    )
    max_extensions: int = Field(default=0, description="Maximum number of scheduler extensions after the initial labelled stage.")
    training_batch_sampler: Literal["stage_class_balanced", "class_balanced_schedule"] = Field(
        default="stage_class_balanced",
        description=(
            "Batch sampler used for scheduler-backed training datasets. "
            "'stage_class_balanced' balances on class and stage; useful for semisupervised training to keep sufficient GT in each batch to compute psuedo-labels."
            "'class_balanced_schedule' class-balances active schedule rows without stage stratification."
        ),
    )
    min_initial_label_fraction: float = Field(default=0.25, description="Minimum fraction of each training batch that must come from the initial GT-labelled pool.")
    auto_resume: bool = Field(default=True, description="Automatically resume from the latest completed staged checkpoint if present.")

    @model_validator(mode="after")
    def validate_staged_training(self) -> "TrainConfig":
        """Validate the staged-training configuration.

        Args:
            None

        Returns:
            TrainConfig: The validated training configuration.
        """

        if not self.monitored_metric:
            raise ValueError("monitored_metric must be a non-empty string.")
        if not 0.0 <= self.min_initial_label_fraction <= 1.0:
            raise ValueError("min_initial_label_fraction must be between 0 and 1.")
        if not 0.0 <= self.model_confidence_threshold <= 1.0:
            raise ValueError("model_confidence_threshold must be between 0 and 1.")
        if self.model_confidence_threshold_end is not None and not 0.0 <= self.model_confidence_threshold_end <= 1.0:
            raise ValueError("model_confidence_threshold_end must be between 0 and 1 when provided.")
        if self.model_confidence_threshold_step <= 0.0:
            raise ValueError("model_confidence_threshold_step must be > 0.")
        if self.model_confidence_threshold_step_epochs < 1:
            raise ValueError("model_confidence_threshold_step_epochs must be >= 1.")
        if self.rows_per_extension < 0:
            raise ValueError("rows_per_extension must be >= 0.")
        if self.candidate_evaluation_budget is not None and self.candidate_evaluation_budget < 0:
            raise ValueError("candidate_evaluation_budget must be >= 0 when provided.")
        if self.max_extensions < 0:
            raise ValueError("max_extensions must be >= 0.")
        if self.neighbor_radius < 1:
            raise ValueError("neighbor_radius must be >= 1.")
        if self.neighbor_samples_per_anchor not in {1, 3, 7}:
            raise ValueError(
                "neighbor_samples_per_anchor must be one of {1, 3, 7}, "
                "so each anchor group has 2, 4, or 8 patches."
            )
        if self.neighbor_selection_start_index < 0:
            raise ValueError("neighbor_selection_start_index must be >= 0.")
        if self.neighbor_unlabeled_curriculum_step_size < 0:
            raise ValueError("neighbor_unlabeled_curriculum_step_size must be >= 0.")
        if (
            self.neighbor_unlabeled_curriculum_max_samples is not None
            and self.neighbor_unlabeled_curriculum_max_samples < 1
        ):
            raise ValueError("neighbor_unlabeled_curriculum_max_samples must be >= 1 when provided.")
        if (
            self.neighbor_unlabeled_curriculum_step_size > 0
            and self.neighbor_unlabeled_curriculum_max_samples is None
        ):
            raise ValueError(
                "neighbor_unlabeled_curriculum_max_samples is required when "
                "neighbor_unlabeled_curriculum_step_size > 0."
            )
        if (
            self.neighbor_unlabeled_curriculum_step_size > 0
            and self.neighbor_samples_per_anchor != 1
        ):
            raise ValueError(
                "neighbor_unlabeled_curriculum_step_size > 0 requires "
                "neighbor_samples_per_anchor=1 to keep batches 50% anchors and 50% unlabeled."
            )
        if self.neighbor_supervised_max_epochs is not None and self.neighbor_supervised_max_epochs < 1:
            raise ValueError("neighbor_supervised_max_epochs must be >= 1 when provided.")
        if self.neighbor_semisupervised_max_epochs is not None and self.neighbor_semisupervised_max_epochs < 1:
            raise ValueError("neighbor_semisupervised_max_epochs must be >= 1 when provided.")
        if self.training_regime == "active_learning" and self.schedule_admission.name not in {
            "admit_all_with_gt",
            "confidence_window_with_gt",
            "confidence_percentile_window_with_gt",
            "reconstruction_error_with_gt",
        }:
            raise ValueError(
                "Active learning requires schedule_admission.name to be one of "
                "{'admit_all_with_gt', 'confidence_window_with_gt', "
                "'confidence_percentile_window_with_gt', 'reconstruction_error_with_gt'}."
            )
        if (
            self.training_regime == "active_learning"
            and self.candidate_sampling.name == "class_balanced_substack"
            and self.extension_samples_per_class is None
        ):
            raise ValueError(
                "Active learning with candidate_sampling.name='class_balanced_substack' "
                "requires extension_samples_per_class."
            )
        return self

    @property
    def resolved_neighbor_supervised_max_epochs(self) -> int:
        return int(self.neighbor_supervised_max_epochs or self.max_epochs)

    @property
    def resolved_neighbor_semisupervised_max_epochs(self) -> int:
        return int(self.neighbor_semisupervised_max_epochs or self.max_epochs)

    @property
    def resolved_validation_variant(self) -> str:
        if self.validation_use_pseudolabel_neighbors and self.validation_variant == "anchor_gt":
            return "spatial_pl"
        return self.validation_variant


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
            print(f"Loaded dataset config from {dataset_cfg_path} with overrides {self.dataset_overrides}")
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

    def checkpoint_path(
        self,
        mode: Literal["supervised", "semisupervised", "active_learning"],
        kind: Literal["best", "last"],
    ) -> Path:
        train_cfg, dataset_cfg, model_cfg = self.get_configs()
        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"{kind}_{mode}.ckpt"

    def best_checkpoint_path(self, mode: Literal["supervised", "semisupervised", "active_learning"]) -> Path:
        """Return the path to the best model checkpoint based on the training mode."""
        return self.checkpoint_path(mode, "best")

    def stage_checkpoint_path(
        self,
        mode: Literal["supervised", "semisupervised", "active_learning"],
        stage_idx: int,
        kind: Literal["best", "last"],
    ) -> Path:
        train_cfg, _, _ = self.get_configs()
        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"{kind}_{mode}_K{stage_idx}.ckpt"

    def stage_scheduler_path(self, stage_idx: int) -> Path:
        train_cfg, _, _ = self.get_configs()
        return self.checkpoints_dir.resolve() / self.experiment_name / train_cfg.model_name / f"scheduler_K{stage_idx}.npz"

    def get_log_dir(self) -> Path:
        """Return the directory path for saving logs."""
        train_cfg, dataset_cfg, model_cfg = self.get_configs()
        return self.logs_dir.resolve() / self.experiment_name / train_cfg.model_name
