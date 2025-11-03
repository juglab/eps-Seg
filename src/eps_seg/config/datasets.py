from eps_seg.config.base import BaseEPSConfig
from pydantic import Field
from typing import List, Optional, Literal, Dict
from pathlib import Path
from pydantic import model_validator
import yaml

class BaseEPSDatasetConfig(BaseEPSConfig):
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load dataset configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config_dict["config_yaml_path"] = yaml_path
        dataset_type = config_dict.get("type")
        if dataset_type == "BetaSegDatasetConfig":
            return BetaSegDatasetConfig(**config_dict)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

class BetaSegDatasetConfig(BaseEPSDatasetConfig):
    dim: int = Field(2, description="Dimensionality of the data (2D or 3D)")
    data_dir: str = Field(..., description="Path to the dataset directory")
    cache_dir: Optional[str] = Field(None, description="Path to cache directory where normalized data and split indices are stored")
    enable_cache: bool = Field(True, description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.")
    train_keys: List[str] = Field(..., description="List of dataset keys to load for training")
    test_keys: List[str] = Field(..., description="List of dataset keys to load for testing")
    seed: int = Field(42, description="Random seed for shuffling the dataset")
    patch_size: int = Field(64, description="Size of the patches to extract from the images")
    n_classes: int = Field(4, description="Number of segmentation classes in the dataset")
    mode: Literal["supervised", "semisupervised"] = Field("supervised", description="Dataset mode: supervised or semisupervised")    
    initial_labeled_ratio: float = Field(1.0, description="Initial ratio of labeled data in semisupervised mode")


    @model_validator(mode="after")
    def check_dirs_not_equal(self) -> "BetaSegDatasetConfig":
        """Ensure data_dir and cache_dir are not the same resolved path."""
        if self.data_dir and self.cache_dir:
            d = Path(self.data_dir).expanduser().resolve(strict=False)
            c = Path(self.cache_dir).expanduser().resolve(strict=False)
            if d == c:
                raise ValueError("data_dir and cache_dir must not resolve to the same path")
        return self
