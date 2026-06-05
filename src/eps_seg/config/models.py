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
    aggregation_mode: str = Field(
        default="SMV",
        description="Aggregation mode for the segmentation prediction. ( SMV: Softmax Majority Voting, PoE: Product of Experts)",
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
