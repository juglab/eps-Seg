from eps_seg.config.base import BaseEPSConfig
from pydantic import Field, model_validator
from typing import List, Literal, Tuple
import yaml

class BaseEPSModelConfig(BaseEPSConfig):

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load model configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_type = config_dict.get("type")
        config_dict["config_yaml_path"] = yaml_path
        if model_type == "LVAEConfig":
            return LVAEConfig(**config_dict)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class LVAEConfig(BaseEPSModelConfig):
    n_components: int = Field(default=4, description="Number of components (classes) for the mixture model.")
    n_layers: int = Field(default=3, description="Number of layers in the LVAE.")
    z_dims: List[int] = Field(default=[32,]*3, description="Latent variable dimensions for each layer.")
    img_shape: List[int] = Field(default=[64, 64], description="Shape of the input images (height, width[, depth]).")
    color_channels: int = Field(default=1, description="Number of color channels in the input images.")
    blocks_per_layer: int = Field(default=5, description="Number of blocks per layer in the LVAE.")
    conv_mult: Literal[2, 3] = Field(default=2, description="Dimensions of the Conv layers (2D or 3D).")
    nonlin: Literal['ReLU', 'LeakyReLU', 'ELU', 'SELU'] = Field(default='ELU', description="Non-linearity to use in the model.")
    enable_top_down_residuals: List[bool] = Field(default=[True, True, True], description="Whether to skip the stochastic merger at each TopDown layer.")
    skip_connections: List[bool] = Field(default=[True, True, True], description="Whether to use skip connections at each layer (i.e., merge bu_values with top-down values).")
    skip_connections_merge_type: Literal["residual", "linear"] = Field(default="residual", description="Type of merge to use for skip connections.")
    use_batchnorm: bool = Field(default=True, description="Whether to use batch normalization in the model.")
    training_mode: Literal['supervised', 'semisupervised'] = Field(default='supervised', description="Training mode for the LVAE. Starts with 'supervised' and can be switched to 'semisupervised' during training.")
    n_filters: int = Field(default=64, description="Number of filters in all convolutional layers.")
    dropout: float = Field(default=0.2, description="Dropout rate to use in the model.")
    kl_free_bits: float = Field(default=0.0, description="Free bits value for KL divergence regularization.")
    learn_top_prior: bool = Field(default=False, description="Whether to learn the top prior distribution.")
    res_block_type: str = Field(default="bacdbacd", description="Type of residual block to use in the model.")
    use_gated_convs: bool = Field(default=True, description="Whether to use gated convolutions in the model.")
    grad_checkpoint: bool = Field(default=True, description="Whether to use gradient checkpointing to save memory during training.")
    no_initial_downscaling: bool = Field(default=True, description="Whether to avoid initial downscaling of the input images.")
    mask_size: int = Field(default=1, description="Size of the mask used in the model")
    # Should be part of training config
    mask_strategy: int = Field(default=0, description="Masking strategy to use during training. If 0, mask is filled with zeros. If >0, mask is filled with average of an annulus of size mask_strategy around the masked region.")
    use_contrastive_learning: bool = Field(default=True, description="Whether to use contrastive loss during training.")
    margin: float = Field(default=20.0, description="Margin value for contrastive loss.")
    learnable_thetas: bool = Field(default=True, description="Whether to use NeurIPS-paper contrastive learning.")
    seg_features: Literal['mu', 'bu'] = Field(default='mu', description="Which features to use for segmentation head ('mu' or 'bu').")
    feature_spatial_size: List[int] = Field(
        default_factory=lambda: [0, 0, 8],
        description="Spatial size of the features used for segmentation head at each hierarchy. 0 disables that level."
    )
    @model_validator(mode="after")
    def check_feature_spatial_size(self):
        if len(self.feature_spatial_size) != self.n_layers:
            raise ValueError(
                f"feature_spatial_size must have length {self.n_layers}, "
                f"got {len(self.feature_spatial_size)}"
            )
        return self
    def check_
