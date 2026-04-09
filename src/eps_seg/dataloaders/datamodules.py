import lightning as L
from tqdm import tqdm
from eps_seg.config.datasets import BaseEPSDatasetConfig, BetaSegDatasetConfig, LiverFibsemDatasetConfig 
from pathlib import Path
import tifffile as tiff
import numpy as np
from typing import Dict, Tuple
from eps_seg.dataloaders.datasets import SemisupervisedDataset, PredictionDataset, PseudoLabelDataset
from eps_seg.config.train import TrainConfig
from torch.utils.data import DataLoader
from eps_seg.dataloaders.samplers import BalancedScheduledBatchSampler, ModeAwareBalancedAnchorBatchSampler, PseudoEpochDistributedParallelBatchSampler
from eps_seg.dataloaders.utils import flex_collate
from sklearn.model_selection import StratifiedKFold
import warnings

from typing import Literal, List, Optional

class EPSSegDataModule(L.LightningDataModule):
    """ 
        Base class for EPS-Seg DataModules.
        Supports semisupervised datasets with caching, normalization, and data splitting.

        To implement a new dataset, subclass this class and override:

            - prepare_data(): Cache data here if needed. Called on a single process, so do not assign state here.
            - _load_
    """

    def __init__(self, cfg: BaseEPSDatasetConfig, train_cfg: TrainConfig, scheduler_path: Optional[Path] = None, scheduler_stage_index: int = 0):
        super().__init__()
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.scheduler_path = Path(scheduler_path) if scheduler_path is not None else None
        self.scheduler_stage_index = scheduler_stage_index
        self.cache_dir = cfg.get_cache_folder()

        self.data = {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

         # Define cache paths
        self.cache_dirs = {
            "train_idx": self.cache_dir / "train_idx.npy",
            "val_idx": self.cache_dir / "val_idx.npy",
            "data_mean": self.cache_dir / "data_mean.npy",
            "data_std": self.cache_dir / "data_std.npy",    
        }
        for key in self.cfg.train_keys:
            self.cache_dirs[f"{key}_normalized"] = Path(self.cache_dir) / f"{key}_normalized.tif"
            self.cache_dirs[f"{key}_labels"] = Path(self.cache_dir) / f"{key}_labels.tif"

    def _check_cache_dir(self):
        """
            Ensure data_dir and cache_dir exist and has data in it.
        """
        for path in self.cache_dirs.values():
            if not path.exists():
                raise FileNotFoundError(f"Cache file {path} does not exist. Recaching required.")

    def prepare_data(self):
        """
            Prepare data for training, validation, testing, and prediction.
            In EPSSeg, this means caching dataset splits to a local folder that is accessible by all nodes.
            DO NOT assign any class state in this method.
        """
        if self.cfg.enable_cache:
            try:
                print(f"Checking cache directory at {self.cache_dir}...")
                self._check_cache_dir()
                print("Cache directory is valid. Skipping caching.")
            except Exception as e:
                print("Cache directory is invalid or incomplete. Loading original data and caching...")
                data_to_cache = self._load_original_dataset_split(split='trainval')
                self._cache_dataset_splits(data_to_cache)
                print("Caching complete.")
        else:
            print("Caching is disabled. Skipping caching step.")

    def get_data_statistics(self) -> Tuple[float, float]:
        """
            Returns the data mean and standard deviation.
            This function is called from the model during setup, to register the statistics as buffers.
            Mean and std are then used during inference for normalization.
        """
        return self.data["data_mean"], self.data["data_std"]
    
    def train_dataloader(self):
  
        train_sampler = BalancedScheduledBatchSampler(
            self.train_dataset,
            batch_size=self.train_cfg.batch_size,
            min_initial_label_fraction=self.train_cfg.min_initial_label_fraction,
            shuffle=True,
            seed=self.cfg.seed,
        )
        
        train_sampler = PseudoEpochDistributedParallelBatchSampler(
            self.train_dataset,
            sampler=train_sampler,
            shuffle=False, # The underlying sampler is already shuffled
            batches_per_pseudoepoch=self.train_cfg.batches_per_pseudoepoch,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            # collate_fn=flex_collate, Default collate for PseudoLabelDataset
        )

    def val_dataloader(self):
        
        val_sampler = ModeAwareBalancedAnchorBatchSampler(
                self.val_dataset,
                total_patches_per_batch=self.train_cfg.batch_size,
                shuffle=False,
                n_neighbors=self.cfg.n_neighbors,
            )
        val_sampler = PseudoEpochDistributedParallelBatchSampler(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
        )

        return DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            collate_fn=flex_collate,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.train_cfg.test_batch_size,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.train_cfg.test_batch_size,
            shuffle=False,
        )

    def _compute_statistics(
        self, images: Dict[str, np.ndarray], train_idx: Dict[str, np.ndarray]
    ):
        """
        Computes mean and standard deviation of the training data.

        Args:
            images (dict): Dictionary mapping keys to image arrays.
            train_idx (dict): Dictionary mapping keys to training indices.
        Returns:
            data_mean (float): Mean of the training data.
            data_std (float): Standard deviation of the training data.
        """
        all_elements = np.concatenate(
            [images[key][train_idx[key]].flatten() for key in self.cfg.train_keys]
        )
        data_mean = np.mean(all_elements)
        data_std = np.std(all_elements.astype(np.float32))
        return data_mean, data_std

    def _generate_fold_splits(self, labels: Dict[str, np.ndarray], shuffle=True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generates training and validation splits for cross-validation folds.
        This function should be used instead of _shuffle_and_split for generating fold splits.

        Args:
            labels (dict): Dictionary mapping keys (file names) to label arrays.
            shuffle (bool): Whether to shuffle the indices before splitting.
        Returns:
            train_idx (dict): Dictionary mapping keys (file names) to training indices.
            val_idx (dict): Dictionary mapping keys (file names) to validation indices.
        """
        keys = self.cfg.train_keys
        train_idx, val_idx = {}, {}
        
        # We use stratified K-fold splitting using strata defined by each stack (file) to ensure that each fold has samples from all stacks 
        # and that the distribution of slices across stacks is balanced in each fold.

        all_data = []
        keys_assignment = []
        for k, key in enumerate(keys):
            # We remove the slices where labels are all -1 (i.e., they are outside the cell) and we only consider the valid slices for splitting
            valid_indices = np.where(~np.all(labels[key] == -1, axis=(-2, -1)))[0]
            all_data.append(valid_indices)
            keys_assignment.append([k,] * len(valid_indices))
        all_data = np.concatenate(all_data)
        keys_assignment = np.concatenate(keys_assignment)

        skf = StratifiedKFold(n_splits=self.cfg.max_folds, shuffle=shuffle, random_state=self.cfg.seed)

        for fold, (train_idx_idx, val_idx_idx) in enumerate(skf.split(all_data, keys_assignment)):
            # We just keep the desired fold
            if fold == self.cfg.fold:
                for k, key in enumerate(keys):
                    # Keep only the indices that belong to the current fold for each stack
                    train_idx[key] = all_data[train_idx_idx][keys_assignment[train_idx_idx] == k]
                    val_idx[key] = all_data[val_idx_idx][keys_assignment[val_idx_idx] == k]
                break
        return train_idx, val_idx

        
    def _shuffle_and_split(
        self, labels: Dict[str, np.ndarray], shuffle=True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Splits dataset into training and validation sets (85% - 15%).
        And shuffles the indices if specified.

        Args:
            labels (dict): Dictionary mapping keys (file names) to label arrays.
            shuffle (bool): Whether to shuffle the indices before splitting.
        Returns:
            train_idx (dict): Dictionary mapping keys (file names) to training indices.
            val_idx (dict): Dictionary mapping keys (file names) to validation indices.

        """
        warnings.warn(
        "This function is deprecated and should not be used directly. It is kept for backward compatibility with original dataset splits. Please use the new _generate_fold_splits function).",
        DeprecationWarning,
        stacklevel=2
    )
        # Function adapted from original boilerplate.dataloader script to preserve reproducible splits
        keys = self.cfg.train_keys
        train_idx, val_idx = {}, {}
        # WARNING: Here RandomState is used to ensure backward compatibility with paper results (that were generated with np.random.seed)
        # DO NOT CHANGE TO np.random.default_rng unless you are ok with having different dataset splits than in the paper
        rng = np.random.RandomState(self.cfg.seed)
        for key in keys:
            # valid indices are the indices of z-slices where labels are not all -1 (i.e., they are not outside the cell)
            valid_indices = np.where(~np.all(labels[key] == -1, axis=(-2, -1)))[0]
            total_samples = valid_indices.shape[0]
            if shuffle:
                rng.shuffle(valid_indices)  # Shuffles in place

            # Compute split index
            split_idx = int(self.cfg.train_to_val_ratio * total_samples)

            # Split the indices
            train_idx[key] = valid_indices[:split_idx]
            val_idx[key] = valid_indices[split_idx:]
        return train_idx, val_idx
    
    def _normalize_data_inplace(
        self, images: Dict[str, np.ndarray], data_mean: float, data_std: float
    ) -> Dict[str, np.ndarray]:
        """
        Normalizes the images using the provided mean and standard deviation.
        This is done in-place.

        Args:
            images (dict): Dictionary mapping keys to image arrays.
            data_mean (float): Mean for normalization.
            data_std (float): Standard deviation for normalization.
        Returns:
            normalized_images (dict): Dictionary mapping keys to normalized image arrays.
        """
        for key in tqdm(self.cfg.train_keys, "Normalizing data"):
            images[key] = (images[key] - data_mean) / data_std
        return images
    
    def _load_original_img_lbls(self, keys: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
            Load the original images and labels from data_dir according to the folder structure defined in the dataset config.

            Args:
                keys (list): List of keys (file names without extensions) to load.
            Returns:
                images (dict): Dictionary mapping keys to image arrays.
                labels (dict): Dictionary mapping keys to label arrays.
        """
        img_lbl_paths = self.cfg.get_image_label_paths(keys)
        imgs = {}
        lbls = {}
        for key, (img_path, lbl_path) in img_lbl_paths.items():
            imgs[key] = tiff.imread(img_path).astype(np.float16)
            lbls[key] = tiff.imread(lbl_path).astype(np.float16)
        return imgs, lbls
    
    def setup(self, stage):
        """
            Setup the DataModule for different stages: 'fit', 'validate', 'test', 'predict'.
            This method should be overridden in subclasses to implement dataset loading.
        """
        super().setup(stage)
        print(f"Setting up data for stage: {stage}...")

        # Load images and labels from file or cache for the specified stage
        if stage in ["fit", "validate"]:
            data = self._load_cached_dataset_splits(split='trainval') if self.cfg.enable_cache else self._load_original_dataset_split(split='trainval')
        else:
            # stage in ['test', 'predict']
            data = self._load_original_dataset_split(split=stage)
        self.data.update(data)

        # Define Datasets based on the stage
        if stage in ["fit"]:
            print(f"Setting up training dataset with {self.cfg.samples_per_class_training} samples per class...")
            self.train_dataset = PseudoLabelDataset(
                images=self.data["trainval_images"],
                labels=self.data["trainval_labels"],
                indices_dict=self.data["train_idx"],
                patch_size=self.cfg.patch_size,
                label_size=1,
                n_classes=self.cfg.n_classes,
                ignore_lbl=-1,
                dim=self.cfg.dim,
                seed=self.cfg.seed,
                samples_per_class=self.cfg.samples_per_class_training,
                scheduler_path=self.scheduler_path,
                stage_index=self.scheduler_stage_index,
                confidence_threshold=self.train_cfg.pseudolabel_confidence_threshold,
            )
        if stage in ["fit", "validate"]:
            print(f"Setting up validation dataset with {self.cfg.samples_per_class_validation} samples per class...")
            self.val_dataset = SemisupervisedDataset(
                images=self.data["trainval_images"],
                labels=self.data["trainval_labels"],
                patch_size=self.cfg.patch_size,
                label_size=1,
                mode="supervised",
                n_classes=self.cfg.n_classes,
                ignore_lbl=-1,
                indices_dict=self.data["val_idx"],
                dim=self.cfg.dim,
                samples_per_class=self.cfg.samples_per_class_validation,
                n_neighbors=self.cfg.n_neighbors
            )
        if stage in ["test", "predict"]:
            # Define the mask for test dataset based on center slice, stepping, and half depth
            
            # Currently, the only thing that differs between test and predict datasets is the region of interest (mask)
            
            
            if stage == "test":
                # Define test region. Masking based on label is done in the dataset class
               
                # NOTICE: These images are NOT normalized! 
                # Normalization is done in the test/predict dataset class using the statistics stored in the LVAE model
                self.test_dataset = PredictionDataset(
                    images=self.data[f"test_images"],
                    labels=self.data[f"test_labels"],
                    keys=self.cfg.test_keys,
                    masks=self._build_slice_mask(stage=stage),
                    patch_size=self.cfg.patch_size,
                    dim=self.cfg.dim,
                    ignore_lbl=-1,
                )
            else:  # stage == "predict"
                # Define predict region. Masking based on label is done in the dataset class
               
                # NOTICE: These images are NOT normalized! 
                # Normalization is done in the test/predict dataset class using the statistics stored in the LVAE model
                # Also we use the same image for test and prediction to avoid loading twice
                self.predict_dataset = PredictionDataset(
                    images=self.data[f"test_images"],
                    labels=self.data[f"test_labels"],
                    keys=self.cfg.test_keys,
                    masks=self._build_slice_mask(stage=stage),
                    patch_size=self.cfg.patch_size,
                    dim=self.cfg.dim,
                    ignore_lbl=-1,
                )

    def _build_slice_mask(self, stage: str) -> Dict[str, Tuple[slice, slice, slice]]:
        """
            Build slice masks for test and predict datasets based on center slice, stepping, and half depth.
            (Stepping is applied in all three dimensions for consistency, and it's only supported for testing but not prediction.)

            Args:
                stage (str): One of 'test' or 'predict'.
            Returns:
                masks (dict): Dictionary mapping keys to slice masks.
        """
        masks = {}
        keys = self.cfg.test_keys # We use test_keys for both test and predict stages
        assert stage in ['test', 'predict'], "Stage must be either 'test' or 'predict'."

        for k, key in enumerate(keys):
            cs = self.cfg.test_center_slices[k] if stage == "test" else self.cfg.predict_center_slices[k] # center slice of the region to segment
            hd = self.cfg.test_half_depths[k] if stage == "test" else self.cfg.predict_half_depths[k] # half depth of the region to segment
            stp = self.cfg.test_steppings[k] if stage == "test" else 1 # No stepping for prediction
            if cs is None:
                # If center slice is None, use the whole volume
                masks[key] = (slice(None, None, stp),
                              slice(None, None, stp),
                              slice(None, None, stp))
            else:
                masks[key] = (slice(cs-hd, cs+hd+1, stp),
                            slice(None, None, stp),
                            slice(None, None, stp))
        return masks

    def _cache_dataset_splits(self, data_to_cache: Dict):
        assert self.cache_dir is not None, (
            "cache_dir must be specified to cache dataset splits."
        )
        assert Path(self.cache_dir).resolve() != Path(self.cfg.data_dir).resolve(), (
            "cache_dir and data_dir must not be the same."
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Save train_idx, val_idx, data_mean, data_std to cache_dir
        np.save(self.cache_dirs["train_idx"], data_to_cache["train_idx"])
        np.save(self.cache_dirs["val_idx"], data_to_cache["val_idx"])
        np.save(self.cache_dirs["data_mean"], data_to_cache["data_mean"])
        np.save(self.cache_dirs["data_std"], data_to_cache["data_std"])
        for key in self.cfg.train_keys:
            tiff.imwrite(
                self.cache_dirs[f"{key}_normalized"],
                data_to_cache["trainval_images"][key].astype(np.float16),
            )
            tiff.imwrite(
                self.cache_dirs[f"{key}_labels"],
                data_to_cache["trainval_labels"][key].astype(np.float16),
            )

    def _load_original_dataset_split(self, split: Literal['trainval', 'test', 'predict']):
        """
            Defines how to load the original dataset splits from data_dir.
            This function should be overridden in subclasses to implement dataset-specific loading logic when passing key / path pairs from config is not enough.

            Args:
                split (str): One of 'trainval', 'test', 'predict'.
            Returns:
                result (dict): Dictionary containing loaded images and labels for the specified split.
        """
        result = {}

        if split == 'trainval':
            result["trainval_images"], result["trainval_labels"] = self._load_original_img_lbls(self.cfg.train_keys)
            result["train_idx"], result["val_idx"] = self._generate_fold_splits(
                                                                             result["trainval_labels"], 
                                                                             shuffle=self.cfg.dim == 2,
                                                                             )
            result["data_mean"], result["data_std"] = self._compute_statistics(
                                                                             images=result["trainval_images"], 
                                                                             train_idx=result["train_idx"]
                                                                            )
            result["trainval_images"] = self._normalize_data_inplace(
                                                                     images=result["trainval_images"], 
                                                                     data_mean=result["data_mean"], 
                                                                     data_std=result["data_std"]
                                                                     )
        elif split in ['test', 'predict']:
            result[f"test_images"], result[f"test_labels"] = self._load_original_img_lbls(self.cfg.test_keys)
        return result

    def _load_cached_dataset_splits(self, split: Literal['trainval', 'test', 'predict']) -> Dict:
        """
            Load cached dataset splits from cache_dir.

            Args:
                split (str): One of 'trainval', 'test', 'predict'.
            Returns:
                data (dict): Dictionary containing loaded images, labels, train_idx, val_idx, data_mean, data_std.
        """
        # TODO: Maybe support also caching for test and predict splits in the future
        if split != 'trainval':
            raise NotImplementedError("Cached loading is currently only implemented for 'trainval' split. Please load original data for other splits.")
        if not Path(self.cache_dir).resolve().exists():
            raise FileNotFoundError(
                f"Cache directory {self.cache_dir} does not exist."
            )
        # Load cached splits, mean, std from cache_dir
        data = {}
        data["data_mean"] = np.load(self.cache_dirs["data_mean"])
        data["data_std"] = np.load(self.cache_dirs["data_std"])
        print(f"Loaded cached data statistics from {self.cache_dir}.")
        if split == 'trainval':
            data["train_idx"] = np.load(self.cache_dirs["train_idx"], allow_pickle=True).item()
            data["val_idx"] = np.load(self.cache_dirs["val_idx"], allow_pickle=True).item()
            data[f"{split}_images"] = {}
            data[f"{split}_labels"] = {}
            keys = self.cfg.train_keys if split == 'trainval' else self.cfg.test_keys
            for key in keys:
                data[f"{split}_images"][key] = tiff.imread(self.cache_dirs[f"{key}_normalized"]).astype(np.float16)
                data[f"{split}_labels"][key] = tiff.imread(self.cache_dirs[f"{key}_labels"]).astype(np.float16)
            print(f"Loaded cached train/validation dataset splits from {self.cache_dir} (fold: {self.cfg.fold}).")
        return data
