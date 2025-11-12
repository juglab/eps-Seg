from eps_seg.config.base import BaseEPSConfig
from pydantic import Field
from typing import List, Optional, Literal, Dict, Tuple
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

    def get_test_paths(self) -> Tuple[List[str], List[str]]:
        """Return the list of test image and label paths."""
        raise NotImplementedError("This method should be implemented in subclasses.")

class BetaSegDatasetConfig(BaseEPSDatasetConfig):
    dim: int = Field(2, description="Dimensionality of the data (2D or 3D)")
    data_dir: str = Field(..., description="Path to the dataset directory")
    cache_dir: Optional[str] = Field(None, description="Path to cache directory where normalized data and split indices are stored")
    enable_cache: bool = Field(True, description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.")
    train_keys: List[str] = Field(..., description="List of dataset keys to load for training")
    test_keys: List[str] = Field(..., description="List of dataset keys to load for testing")
    seed: int = Field(42, description="Random seed for shuffling the dataset")
    patch_size: int = Field(64, description="Size of the patches to extract from the images")
    n_channels: int = Field(1, description="Number of image channels in the dataset")
    n_classes: int = Field(4, description="Number of segmentation classes in the dataset")
    mode: Literal["supervised", "semisupervised"] = Field("supervised", description="Dataset mode: supervised or semisupervised")    
    samples_per_class_validation: Optional[Dict[int, int]] = Field({1: 2}, description="Number of samples per class for validation dataset. If None, defaults to 1 per class.")
    samples_per_class_training: Optional[Dict[int, int]] = Field({1: 2}, description="Number of samples per class for training dataset. If None, defaults to unlimited.")

    @model_validator(mode="after")
    def check_dirs_not_equal(self) -> "BetaSegDatasetConfig":
        """Ensure data_dir and cache_dir are not the same resolved path."""
        if self.data_dir and self.cache_dir:
            d = Path(self.data_dir).expanduser().resolve(strict=False)
            c = Path(self.cache_dir).expanduser().resolve(strict=False)
            if d == c:
                raise ValueError("data_dir and cache_dir must not resolve to the same path")
        return self
    
    def get_test_paths(self) -> Tuple[List[str], List[str]]:
        """Return the list of test keys."""
        img_paths = [str(Path(self.data_dir) / key / f"{key}_source.tif") for key in self.test_keys]
        lbl_paths = [str(Path(self.data_dir) / key / f"{key}_gt.tif") for key in self.test_keys]
        return img_paths, lbl_paths

class LiverFibsemDatasetConfig(BaseEPSDatasetConfig):
    dim: int = Field(2, description="Dimensionality of the data (2D or 3D)")
    data_dir: str = Field(..., description="Path to the dataset directory")
    cache_dir: Optional[str] = Field(None, description="Path to cache directory where normalized data and split indices are stored")
    enable_cache: bool = Field(True, description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.")
    train_keys: List[str] = Field(..., description="List of dataset keys to load for training")
    test_keys: List[str] = Field(..., description="List of dataset keys to load for testing")
    seed: int = Field(42, description="Random seed for shuffling the dataset")
    patch_size: int = Field(64, description="Size of the patches to extract from the images")
    n_classes: int = Field(7, description="Number of segmentation classes in the dataset")
    mode: Literal["supervised", "semisupervised"] = Field("supervised", description="Dataset mode: supervised or semisupervised")    


    @model_validator(mode="after")
    def check_dirs_not_equal(self) -> "LiverFibsemDatasetConfig":
        """Ensure data_dir and cache_dir are not the same resolved path."""
        if self.data_dir and self.cache_dir:
            d = Path(self.data_dir).expanduser().resolve(strict=False)
            c = Path(self.cache_dir).expanduser().resolve(strict=False)
            if d == c:
                raise ValueError("data_dir and cache_dir must not resolve to the same path")
        return self
