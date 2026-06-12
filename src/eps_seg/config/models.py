from eps_seg.config.base import BaseEPSConfig
from pydantic import Field, model_validator
from typing import List, Literal, Tuple, Union
import yaml


class BaseEPSModelConfig(BaseEPSConfig):
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load model configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        model_type = config_dict.get("type")
        config_dict["config_yaml_path"] = yaml_path
        if model_type == "LVAEConfig":
            return LVAEConfig(**config_dict)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class LVAEConfig(BaseEPSModelConfig):
    architecture: Literal["eps_seg_plus", "eps_seg_vanilla"] = Field(
        default="eps_seg_plus",
        description="Architecture family to instantiate.",
    )
    n_components: int = Field(
        default=4, description="Number of components (classes) for the mixture model."
    )
    n_layers: int = Field(default=3, description="Number of layers in the LVAE.")
    z_dims: List[int] = Field(
        default_factory=lambda: [32, 32, 32],
        description="Latent variable dimensions for each layer.",
    )
    img_shape: List[int] = Field(
        default=[64, 64],
        description="Shape of the input images (height, width[, depth]).",
    )
    color_channels: int = Field(
        default=1, description="Number of color channels in the input images."
    )
    blocks_per_layer: int = Field(
        default=5, description="Number of blocks per layer in the LVAE."
    )
    conv_mult: Literal[2, 3] = Field(
        default=2, description="Dimensions of the Conv layers (2D or 3D)."
    )
    nonlin: Literal["ReLU", "LeakyReLU", "ELU", "SELU"] = Field(
        default="ELU", description="Non-linearity to use in the model."
    )
    enable_top_down_residuals: List[bool] = Field(
        default_factory=lambda: [True, True, True],
        description="Whether to skip the stochastic merger at each TopDown layer.",
    )
    skip_connections: List[bool] = Field(
        default_factory=lambda: [True, True, True],
        description="Whether to use skip connections at each layer (i.e., merge bu_values with top-down values).",
    )
    skip_connections_merge_type: Literal["residual", "linear"] = Field(
        default="residual", description="Type of merge to use for skip connections."
    )
    use_batchnorm: bool = Field(
        default=True, description="Whether to use batch normalization in the model."
    )
    training_mode: Literal["supervised", "semisupervised"] = Field(
        default="supervised",
        description="Training mode for the LVAE. Starts with 'supervised' and can be switched to 'semisupervised' during training.",
    )
    n_filters: Union[int, List[int], None] = Field(
        default=None,
        description=(
            "Deterministic feature channels. "
            "If int, the same value is used for all layers. "
            "If list, must have one value per layer. "
            "If None, defaults to z_dims."
        ),
    )
    dropout: float = Field(default=0.2, description="Dropout rate to use in the model.")
    kl_free_bits: float = Field(
        default=0.0, description="Free bits value for KL divergence regularization."
    )
    learn_top_prior: bool = Field(
        default=False, description="Whether to learn the top prior distribution."
    )
    res_block_type: str = Field(
        default="bacdbacd", description="Type of residual block to use in the model."
    )
    use_gated_convs: bool = Field(
        default=True, description="Whether to use gated convolutions in the model."
    )
    grad_checkpoint: bool = Field(
        default=True,
        description="Whether to use gradient checkpointing to save memory during training.",
    )
    no_initial_downscaling: bool = Field(
        default=True,
        description="Whether to avoid initial downscaling of the input images.",
    )
    mask_size: int = Field(default=1, description="Size of the mask used in the model")
    # Should be part of training config
    mask_strategy: int = Field(
        default=0,
        description="Masking strategy to use during training. If 0, mask is filled with zeros. If >0, mask is filled with average of an annulus of size mask_strategy around the masked region.",
    )
    use_contrastive_learning: bool = Field(
        default=True, description="Whether to use contrastive loss during training."
    )
    margin: float = Field(default=20.0, description="Margin value for contrastive loss.")
    learnable_thetas: bool = Field(
        default=True, description="Whether to use NeurIPS-paper contrastive learning."
    )
    aggregation_mode: Literal["SMV", "PoE"] = Field(
        default="SMV",
        description="Aggregation mode for the segmentation prediction. ( SMV: Softmax Majority Voting, PoE: Product of Experts)",
    )
    seg_features: Literal["mu", "bu"] = Field(
        default="mu",
        description="Which features to use for the vanilla segmentation head.",
    )
    feature_spatial_size: List[int] = Field(
        default_factory=lambda: [0, 0, 8],
        description="Spatial size of selected features at each hierarchy. 0 disables that level.",
    )
    top_prior_schedule: Literal["none", "static", "dynamic"] = Field(
        default="none",
        description="Policy controlling the top-prior mean evolution across epochs.",
    )
    top_prior_mu_init: float = Field(
        default=10.0,
        description="Initial mean value assigned to active top-prior mixture channels.",
    )
    top_prior_mu_supervised_max: float | None = Field(
        default=None,
        description=(
            "Target top-prior mean reached during supervised training when using "
            "the dynamic schedule. The scheduler increases or decreases toward this value."
        ),
    )
    top_prior_mu_semisupervised_min: float | None = Field(
        default=None,
        description=(
            "Lower target top-prior mean reached during semisupervised training when using "
            "the dynamic schedule. Semisupervised starts from the carried supervised "
            "checkpoint value and only decreases toward this value when it is lower."
        ),
    )
    top_prior_mu_step_epochs: int = Field(
        default=10,
        description="Number of epochs between unit changes in the dynamic top-prior schedule.",
    )

    @model_validator(mode="after")
    def check_z_dim_size(self):
        if len(self.z_dims) != self.n_layers:
            raise ValueError(
                f"z_dims must have length {self.n_layers}, got {len(self.z_dims)}"
            )
        return self

    @model_validator(mode="after")
    def check_skip_connections(self):
        if len(self.skip_connections) != self.n_layers:
            raise ValueError(
                f"skip_connections must have length {self.n_layers}, got {len(self.skip_connections)}"
            )
        return self

    @model_validator(mode="after")
    def check_enable_top_down_residuals(self):
        if len(self.enable_top_down_residuals) != self.n_layers:
            raise ValueError(
                f"enable_top_down_residuals must have length {self.n_layers}, got {len(self.enable_top_down_residuals)}"
            )
        return self

    @model_validator(mode="after")
    def normalize_n_filters(self):
        if self.n_filters is None:
            n_filters = list(self.z_dims)
        elif isinstance(self.n_filters, int):
            n_filters = [self.n_filters] * self.n_layers
        else:
            n_filters = list(self.n_filters)

        if len(n_filters) != self.n_layers:
            raise ValueError(
                f"n_filters must have length {self.n_layers}, got {len(n_filters)}"
            )

        if any(f <= 0 for f in n_filters):
            raise ValueError("All n_filters values must be > 0")

        self.n_filters = n_filters
        return self

    @model_validator(mode="after")
    def validate_architecture_specific_fields(self):
        natural_plus_feature_spatial_size = [
            int(self.img_shape[-1] / (2 ** (layer_idx + 1)))
            for layer_idx in range(self.n_layers)
        ]
        default_vanilla_feature_spatial_size = [0] * max(self.n_layers - 1, 0) + [
            natural_plus_feature_spatial_size[-1]
        ]
        feature_spatial_size_was_explicit = "feature_spatial_size" in self.model_fields_set

        if self.architecture == "eps_seg_plus" and not feature_spatial_size_was_explicit:
            self.feature_spatial_size = natural_plus_feature_spatial_size
        if self.architecture == "eps_seg_vanilla" and not feature_spatial_size_was_explicit:
            self.feature_spatial_size = default_vanilla_feature_spatial_size

        if len(self.feature_spatial_size) != self.n_layers:
            if self.feature_spatial_size == [0, 0, 8]:
                self.feature_spatial_size = (
                    natural_plus_feature_spatial_size
                    if self.architecture == "eps_seg_plus"
                    else default_vanilla_feature_spatial_size
                )
            else:
                raise ValueError(
                    f"feature_spatial_size must have length {self.n_layers}, got {len(self.feature_spatial_size)}"
                )
        if any(size < 0 for size in self.feature_spatial_size):
            raise ValueError("feature_spatial_size values must be >= 0.")

        if self.top_prior_mu_step_epochs < 1:
            raise ValueError("top_prior_mu_step_epochs must be >= 1.")

        if self.top_prior_schedule == "dynamic":
            if self.top_prior_mu_supervised_max is None:
                raise ValueError("top_prior_mu_supervised_max is required when top_prior_schedule='dynamic'.")
            if self.top_prior_mu_semisupervised_min is None:
                raise ValueError("top_prior_mu_semisupervised_min is required when top_prior_schedule='dynamic'.")

        if self.architecture == "eps_seg_vanilla":
            if len(set(self.n_filters)) != 1:
                raise ValueError("eps_seg_vanilla requires uniform n_filters across all layers.")
            if self.feature_spatial_size != default_vanilla_feature_spatial_size:
                raise ValueError(
                    "eps_seg_vanilla only supports the default feature_spatial_size "
                    f"{default_vanilla_feature_spatial_size}."
                )
        else:
            if self.seg_features != "mu":
                raise ValueError("seg_features is only supported by eps_seg_vanilla.")
            if any(size == 0 for size in self.feature_spatial_size):
                raise ValueError("eps_seg_plus requires all feature_spatial_size values to be > 0.")
            invalid_sizes = [
                (layer_idx, size, natural_size)
                for layer_idx, (size, natural_size) in enumerate(
                    zip(self.feature_spatial_size, natural_plus_feature_spatial_size)
                )
                if size > natural_size
            ]
            if invalid_sizes:
                raise ValueError(
                    "eps_seg_plus feature_spatial_size values must be <= the natural "
                    f"per-layer feature sizes {natural_plus_feature_spatial_size}. "
                    f"Invalid entries: {invalid_sizes}."
                )

        return self
