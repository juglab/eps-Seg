from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import Field, model_validator

from eps_seg.config.base import BaseEPSConfig


class BaseEPSDatasetConfig(BaseEPSConfig):
    name: str = Field(..., description="Name of the dataset, used for cache naming")
    dim: int = Field(..., description="Dimensionality of the data (2D or 3D)")
    data_dir: str = Field(..., description="Path to the dataset directory")
    cache_root: Optional[str] = Field(
        ...,
        description="Path to the cache root where normalized data and fold assignments are stored.",
    )
    enable_cache: bool = Field(
        ...,
        description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.",
    )
    train_keys: List[str] = Field(..., description="List of dataset keys to load for training")
    test_keys: List[str] = Field(..., description="List of dataset keys to load for testing")
    test_center_slices: List[Union[int, None]] = Field(
        ...,
        description="Center slice index for testing dataset. (one for each test_key). If None, use the whole volume",
    )
    test_steppings: List[int] = Field(
        ...,
        description="Stepping value for slicing the test dataset (each one applies in all dimensions, only for testing). (One for each test_key)",
    )
    test_half_depths: List[int] = Field(
        ...,
        description="Half depth of the test volume to consider around the center slice (applies in Z dimension for testing). (One for each test_key). Only has effect if test_center_slices is not None.",
    )
    predict_center_slices: List[Union[int, None]] = Field(
        ...,
        description="Center slice index for prediction dataset (one for each test_key). If None, use the whole volume",
    )
    predict_half_depths: List[int] = Field(
        ...,
        description="Half depth of the prediction volume to consider around the center slice (applies in Z dimension for prediction). (One for each test_key). Only has effect if predict_center_slices is not None.",
    )
    seed: int = Field(..., description="Random seed used for fold creation and anchor sampling")
    train_to_val_ratio: float = Field(
        ...,
        description="Ratio of training to validation substacks when max_folds == 1. Used only when building a fresh cache.",
    )
    patch_size: int = Field(..., description="Size of the patches to extract from the images")
    n_channels: int = Field(..., description="Number of image channels in the dataset")
    n_classes: int = Field(..., description="Number of segmentation classes in the dataset")
    mode: Literal["supervised", "semisupervised"] = Field(
        "supervised",
        description="Dataset mode: supervised or semisupervised",
    )
    samples_per_class: Optional[Dict[int, int]] = Field(
        None,
        description="Number of GT anchor centers to sample per class per valid z-slice before folding.",
    )
    samples_per_class_validation: Optional[Dict[int, int]] = Field(
        None,
        description="Deprecated. Validation-specific GT sampling is ignored by the canonical-anchor dataloader.",
    )
    samples_per_class_training: Optional[Dict[int, int]] = Field(
        None,
        description="Deprecated. If samples_per_class is omitted, these values are used as a fallback.",
    )
    n_neighbors: int = Field(
        7,
        description="Number of neighbors for neighbor sampling (only for semisupervised datasets).",
    )
    fold: int = Field(..., description="Fold number for cross-validation")
    max_folds: int = Field(..., description="Maximum number of folds for cross-validation")
    substacking: int = Field(
        1,
        description="Depth of contiguous z-substacks used as fold units. 1 corresponds to slice-wise folding.",
    )
    initial_label_sampling: Optional[Literal["from_csv", "class_balanced_slice", "class_balanced_substack"]] = Field(
        None,
        description="Optional strategy used to build the canonical labelled universe when creating a fresh cache.",
    )
    load_train_coords_from: Optional[str] = Field(
        None,
        description="Optional CSV file with externally sampled train coordinates. Used only when creating a fresh cache in 1-fold mode.",
    )
    load_val_coords_from: Optional[str] = Field(
        None,
        description="Optional CSV file with externally sampled validation coordinates. Used only when creating a fresh cache in 1-fold mode.",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load dataset configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config_dict["config_yaml_path"] = yaml_path
        dataset_type = config_dict.get("type")
        if dataset_type == "BetaSegDatasetConfig":
            return BetaSegDatasetConfig(**config_dict)
        if dataset_type == "ZStackedSlice2DDatasetConfig":
            return ZStacked2DDatasetConfig(**config_dict)
        if dataset_type == "LiverFibsemDatasetConfig":
            return LiverFibsemDatasetConfig(**config_dict)
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
        Returns the image and label file paths for the specified keys.

        Args:
            keys (List[str]): The dataset keys for which to get the paths.
        Returns:
            Dict[str, Tuple[Path, Path]]: A dictionary mapping dataset keys to
                their corresponding image and label file paths.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_cache_folder(self) -> Path:
        """
        Get the shared cache folder path for this dataset configuration.

        The cache folder is independent from the selected fold; fold-specific
        files live inside the shared directory and are resolved at runtime.
        """
        if self.cache_root is None:
            raise ValueError("cache_root must be specified to get the cache folder.")
        return Path(self.cache_root) / (
            f"{self.name}_seed_{self.seed}_substack_{self.substacking}_folds_{self.max_folds}"
        )

    @model_validator(mode="after")
    def _validate_sampling_fields(self):
        if self.fold < 0:
            raise ValueError("fold must be >= 0.")
        if self.max_folds < 1:
            raise ValueError("max_folds must be >= 1.")
        if self.fold >= self.max_folds:
            raise ValueError("fold must be < max_folds.")
        if self.substacking < 1:
            raise ValueError("substacking must be >= 1.")
        if not 0.0 < self.train_to_val_ratio < 1.0:
            raise ValueError("train_to_val_ratio must be between 0 and 1.")

        importing_coords = self.load_train_coords_from is not None or self.load_val_coords_from is not None
        if importing_coords:
            if self.load_train_coords_from is None or self.load_val_coords_from is None:
                raise ValueError(
                    "Both load_train_coords_from and load_val_coords_from must be set together."
                )
            if self.max_folds != 1 or self.fold != 0:
                raise ValueError(
                    "External coordinate import is only supported in 1-fold mode with fold=0."
                )
            if self.substacking != 1:
                raise ValueError(
                    "External coordinate import is only supported with substacking=1."
                )

        if self.samples_per_class is None:
            if self.samples_per_class_training is None:
                raise ValueError(
                    "samples_per_class must be provided for the canonical-anchor dataloader."
                )
            self.samples_per_class = dict(self.samples_per_class_training)

        if self.samples_per_class_validation is not None and (
            self.samples_per_class_validation != self.samples_per_class
        ):
            raise ValueError(
                "samples_per_class_validation is deprecated. Use one shared samples_per_class dictionary."
            )

        self.samples_per_class_training = None
        self.samples_per_class_validation = None
        return self


class ZStacked2DDatasetConfig(BaseEPSDatasetConfig):
    """
    Configuration for 2D slice-based datasets where patches are previously
    stacked in the Z dimension and then treated as independent 2D samples.

    Dataset structure:
    [data_dir]/[key]_source.tif
    [data_dir]/[key]_gt.tif
    where key is the identifier for each stack.
    """

    dim: int = 2
    seed: int = 42
    train_to_val_ratio: float = 0.85
    patch_size: int = 64

    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
        Return the image and label file paths for the specified keys.

        For ZStacked2DDatasetConfig, we have
        [data_dir]/[key]_source.tif and [data_dir]/[key]_gt.tif.
        """
        paths = {}
        for key in keys:
            img_path = Path(self.data_dir) / f"{key}_source.tif"
            lbl_path = Path(self.data_dir) / f"{key}_gt.tif"
            paths[key] = (img_path, lbl_path)
        return paths

    @model_validator(mode="after")
    def _populate_test_and_predict_areas(self):
        n_test = len(self.test_keys)
        self.test_center_slices = [None] * n_test
        self.test_steppings = [1] * n_test
        self.test_half_depths = [0] * n_test
        self.predict_center_slices = [None] * n_test
        self.predict_half_depths = [0] * n_test
        return self


class BetaSegDatasetConfig(BaseEPSDatasetConfig):
    dim: int = 2
    data_dir: str = Field(..., description="Path to the dataset directory")
    enable_cache: bool = Field(
        True,
        description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.",
    )
    train_keys: List[str] = ["high_c1", "high_c2", "high_c3"]
    test_keys: List[str] = ["high_c4"]
    test_center_slices: List[Union[int, None]] = [626]
    test_steppings: List[int] = [3]
    test_half_depths: List[int] = [6]
    predict_center_slices: List[Union[int, None]] = [626]
    predict_half_depths: List[int] = [0]
    seed: int = 42
    train_to_val_ratio: float = 0.85
    patch_size: int = 64
    n_channels: int = 1
    n_classes: int = 4
    samples_per_class: Optional[Dict[int, int]] = None
    max_folds: int = 5

    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
        Return the image and label file paths for the specified keys.

        For BetaSeg, we have
        [data_dir]/[key]/[key]_source.tif and [data_dir]/[key]/[key]_gt.tif.
        """
        paths = {}
        for key in keys:
            img_path = Path(self.data_dir) / key / f"{key}_source.tif"
            lbl_path = Path(self.data_dir) / key / f"{key}_gt.tif"
            paths[key] = (img_path, lbl_path)
        return paths


class LiverFibsemDatasetConfig(BaseEPSDatasetConfig):
    dim: int = 2
    data_dir: str = Field(..., description="Path to the dataset directory")
    enable_cache: bool = Field(
        True,
        description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.",
    )
    train_keys: List[str] = [
        "crop_01",
        "crop_02",
        "crop_03",
        "crop_04",
        "crop_05",
        "crop_06",
        "crop_07",
        "crop_08",
        "crop_09",
    ]
    test_keys: List[str] = ["crop_00", "crop_10"]
    test_center_slices: List[Union[int, None]] = [None, None]
    test_steppings: List[int] = [1, 1]
    test_half_depths: List[int] = [0, 0]
    predict_center_slices: List[Union[int, None]] = [None, None]
    predict_half_depths: List[int] = [0, 0]
    seed: int = 42
    train_to_val_ratio: float = 0.80
    patch_size: int = 64
    n_channels: int = 1
    n_classes: int = 7
    samples_per_class: Optional[Dict[int, int]] = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 4,
        6: 30,
    }

    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
        Return the image and label file paths for the specified keys.

        For LiverFibsem, we have [data_dir]/key/image.tif and
        [data_dir]/key/label.tif.
        """
        paths = {}
        for key in keys:
            img_path = Path(self.data_dir) / key / "image.tif"
            lbl_path = Path(self.data_dir) / key / "labs.tif"
            paths[key] = (img_path, lbl_path)
        return paths
