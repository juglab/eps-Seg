from eps_seg.config.base import BaseEPSConfig
from pydantic import Field, model_validator
from typing import List, Optional, Literal, Dict, Tuple, Union
from pathlib import Path
import yaml

class BaseEPSDatasetConfig(BaseEPSConfig):
    dim: int = Field(..., description="Dimensionality of the data (2D or 3D)")
    data_dir: str = Field(..., description="Path to the dataset directory")
    cache_dir: Optional[str] = Field(..., description="Path to cache directory where normalized data and split indices are stored")
    enable_cache: bool = Field(..., description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.")
    train_keys: List[str] = Field(..., description="List of dataset keys to load for training")
    test_keys: List[str] = Field(..., description="List of dataset keys to load for testing")
    test_center_slices: List[Union[int, None]] = Field(..., description="Center slice index for testing dataset. (one for each test_key). If None, use the whole volume")
    test_steppings: List[int] = Field(..., description="Stepping value for slicing the test dataset (each one applies in all dimensions, only for testing). (One for each test_key)")
    test_half_depths: List[int] = Field(..., description="Half depth of the test volume to consider around the center slice (applies in Z dimension for testing). (One for each test_key). Only has effect if test_center_slices is not None.")
    predict_center_slices: List[Union[int, None]] = Field(..., description="Center slice index for prediction dataset (one for each test_key). If None, use the whole volume")
    predict_half_depths: List[int] = Field(..., description="Half depth of the prediction volume to consider around the center slice (applies in Z dimension for prediction). (One for each test_key). Only has effect if predict_center_slices is not None.")
    seed: int = Field(..., description="Random seed for shuffling the dataset")
    train_to_val_ratio: float = Field(..., description="Ratio of training to validation data split. Used only when not using cached data. Delete your cache to re-split the data.")
    patch_size: int = Field(..., description="Size of the patches to extract from the images")
    n_channels: int = Field(..., description="Number of image channels in the dataset")
    n_classes: int = Field(..., description="Number of segmentation classes in the dataset")
    mode: Literal["supervised", "semisupervised"] = Field("supervised", description="Dataset mode: supervised or semisupervised")    
    samples_per_class_validation: Optional[Dict[int, int]] = Field(..., description="Number of samples per class for validation dataset.")
    samples_per_class_training: Optional[Dict[int, int]] = Field(..., description="Number of samples per class for training dataset.")
    n_neighbors: int = Field(7, description="Number of neighbors for neighbor sampling (only for semisupervised datasets).")

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load dataset configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config_dict["config_yaml_path"] = yaml_path
        dataset_type = config_dict.get("type")
        if dataset_type == "BetaSegDatasetConfig":
            return BetaSegDatasetConfig(**config_dict)
        elif dataset_type == "ZStackedSlice2DDatasetConfig":
            return ZStackedSlice2DDatasetConfig(**config_dict)
        elif dataset_type == "SliceBasedDatasetConfig":
            return SliceBasedDatasetConfig(**config_dict)
        elif dataset_type == "LiverFibsemDatasetConfig":
            return LiverFibsemDatasetConfig(**config_dict)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
            Returns the image and label file paths for the specified keys (that corresponds to train_keys and test_keys).
            Must be implemented in subclasses.

            Args:
                keys (List[str]): The dataset keys for which to get the paths.
            Returns:
                Dict[str, Tuple[Path, Path]]: A dictionary mapping dataset keys to their corresponding image and label file paths.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

class SliceBasedDatasetConfig(BaseEPSDatasetConfig):
    """
        Configuration for slice-based datasets, that looks for image and label files in the format:
        [data_dir]/[train|test]/[im|labs]/[key].tif
    """

    train_keys: List[str] = Field(default_factory=list, description="Auto-discovered train keys from [data_dir]/train/[im|labs]")
    test_keys: List[str] = Field(default_factory=list, description="Auto-discovered test keys from [data_dir]/test/[im|labs]")
    test_center_slices: List[Union[int, None]] = Field(default_factory=list, description="Auto-populated: full-slice testing for each discovered test key.")
    test_steppings: List[int] = Field(default_factory=list, description="Auto-populated: stepping=1 for each discovered test key.")
    test_half_depths: List[int] = Field(default_factory=list, description="Auto-populated: half-depth=0 for each discovered test key.")
    predict_center_slices: List[Union[int, None]] = Field(default_factory=list, description="Auto-populated: full-slice prediction for each discovered test key.")
    predict_half_depths: List[int] = Field(default_factory=list, description="Auto-populated: half-depth=0 for each discovered test key.")

    @model_validator(mode="after")
    def _populate_keys(self):
        self.train_keys = self._discover_split_keys("train")
        self.test_keys = self._discover_split_keys("test")
        # Slice-based dataset uses one file per slice; force full-slice test/predict for all discovered test keys.
        n_test = len(self.test_keys)
        self.test_center_slices = [None] * n_test
        self.test_steppings = [1] * n_test
        self.test_half_depths = [0] * n_test
        self.predict_center_slices = [None] * n_test
        self.predict_half_depths = [0] * n_test
        return self

    def _discover_split_keys(self, split: Literal["train", "test"]) -> List[str]:
        data_root = Path(self.data_dir)
        img_dir = data_root / split / "im"
        lbl_dir = data_root / split / "labs"

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image directory for split '{split}': {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing label directory for split '{split}': {lbl_dir}")

        img_keys = {p.stem for p in img_dir.glob("*.tif") if p.is_file()}
        lbl_keys = {p.stem for p in lbl_dir.glob("*.tif") if p.is_file()}

        missing_labels = sorted(img_keys - lbl_keys)
        missing_images = sorted(lbl_keys - img_keys)
        if missing_labels or missing_images:
            raise ValueError(
                f"Found mismatched image/label files in split '{split}'. "
                f"Missing labels for: {missing_labels}. Missing images for: {missing_images}."
            )

        common_keys = sorted(img_keys & lbl_keys)
        if not common_keys:
            raise ValueError(f"No paired .tif files found in split '{split}' under {img_dir} and {lbl_dir}.")
        return common_keys

    def _resolve_split_for_key(self, key: str) -> str:
        in_train = key in self.train_keys
        in_test = key in self.test_keys
        if in_train and in_test:
            raise ValueError(f"Key '{key}' exists in both train_keys and test_keys. Split is ambiguous.")
        if in_train:
            return "train"
        if in_test:
            return "test"

        candidate_splits = []
        for split in ("train", "test"):
            img_path = Path(self.data_dir) / split / "im" / f"{key}.tif"
            lbl_path = Path(self.data_dir) / split / "labs" / f"{key}.tif"
            if img_path.exists() and lbl_path.exists():
                candidate_splits.append(split)

        if len(candidate_splits) == 1:
            return candidate_splits[0]
        if len(candidate_splits) > 1:
            raise ValueError(f"Key '{key}' exists in both train and test folders. Split is ambiguous.")
        raise KeyError(f"Key '{key}' was not found in discovered train_keys/test_keys and no matching files exist.")

    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        paths: Dict[str, Tuple[Path, Path]] = {}
        for key in keys:
            split = self._resolve_split_for_key(key)
            img_path = Path(self.data_dir) / split / "im" / f"{key}.tif"
            lbl_path = Path(self.data_dir) / split / "labs" / f"{key}.tif"
            paths[key] = (img_path, lbl_path)
        return paths
        
class ZStackedSlice2DDatasetConfig(BaseEPSDatasetConfig):
    """
        Configuration for 2D slice-based datasets where the samples are patches that are previously stacked in the Z dimension. 
        This is useful so the user can balance the dataset or classes depending on intensity or other criteria, while still using the 2D training pipeline. 
        This has to be used with 2D models. 

        Dataset will have the following structure:
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
            For ZStackedSlice2DDataset, we have [data_dir]/[key]_source.tif and [data_dir]/[key]_gt.tif
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
    # Defaults for BetaSeg 2D dataset
    dim: int = 2
    data_dir: str = Field(..., description="Path to the dataset directory")
    cache_dir: Optional[str] = Field(None, description="Path to cache directory where normalized data and split indices are stored")
    enable_cache: bool = Field(True, description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.")
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
    samples_per_class_training: Optional[Dict[int, int]] = Field(..., description="Number of samples per class for training dataset.")
    samples_per_class_validation: Optional[Dict[int, int]] = Field(..., description="Number of samples per class for validation dataset.")
    
    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
            Return the image and label file paths for the specified keys.
            For BetaSeg, we have [data_dir]/[key]/[key]_source.tif and [data_dir]/[key]/[key]_gt.tif
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
    cache_dir: Optional[str] = Field(None, description="Path to cache directory where normalized data and split indices are stored")
    enable_cache: bool = Field(True, description="Whether to use/store cached dataset splits if available. Set to false to preserve disk space.")
    train_keys: List[str] = ["crop_01", "crop_02", "crop_03", "crop_04", "crop_05", "crop_06", "crop_07", "crop_08", "crop_09"]
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
    samples_per_class_training: Optional[Dict[int, int]] = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 30}
    samples_per_class_validation: Optional[Dict[int, int]] = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 30}


    def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
        """
            Return the image and label file paths for the specified keys.
            For LiverFibsem, we have [data_dir]/key/image.tif and [data_dir]/key/label.tif
        """
        paths = {}
        for key in keys:
            img_path = Path(self.data_dir) / key / "image.tif"
            lbl_path = Path(self.data_dir) / key / "labs.tif"
            paths[key] = (img_path, lbl_path)
        return paths
