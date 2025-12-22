import lightning as L
from tqdm import tqdm
from eps_seg.config.datasets import BaseEPSDatasetConfig, BetaSegDatasetConfig, LiverFibsemDatasetConfig
from pathlib import Path
import tifffile as tiff
import numpy as np
from typing import Dict, Tuple
from eps_seg.dataloaders.datasets import SemisupervisedDataset
from eps_seg.config.train import TrainConfig
from torch.utils.data import DataLoader
from eps_seg.dataloaders.samplers import ModeAwareBalancedAnchorBatchSampler, PseudoEpochDistributedParallelBatchSampler
from eps_seg.dataloaders.utils import flex_collate
import yaml
from typing import Literal, List, Optional

class LiverFibsemTrainDataModule(L.LightningDataModule):
    def __init__(self, cfg: LiverFibsemDatasetConfig, train_cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.train_cfg = train_cfg

    def setup(self, stage):
        super().setup(stage)
        try:
            print(f"Loading cached dataset splits from {self.cfg.cache_dir}...")
            self._load_cached_dataset_splits()
        except Exception as e:
            print("Could not load cached dataset splits. Loading original data.")
            self.images, self.labels = self._load_original_img_lbls()

            # Split the data, only shuffle if 2D data
            self.train_idx, self.val_idx = self._shuffle_and_split(
                self.labels, shuffle=self.cfg.dim == 2
            )
            self.data_mean, self.data_std = self._compute_statistics(
                self.images, self.train_idx
            )
            self.images = self._normalize_data_inplace(
                self.images, self.data_mean, self.data_std
            )
            print("Caching dataset splits...")
            self._cache_dataset_splits()

        # Define Datasets for each split
        self.train_dataset = SemisupervisedDataset(
            images=self.images,
            labels=self.labels,
            patch_size=self.cfg.patch_size,
            label_size=1,
            mode=self.cfg.mode,
            n_classes=self.cfg.n_classes,
            ignore_lbl=-1,
            indices_dict=self.train_idx,
            radius=self.train_cfg.initial_radius,
            dim=self.cfg.dim,
        )

        self.val_dataset = SemisupervisedDataset(
            images=self.images,
            labels=self.labels,
            patch_size=self.cfg.patch_size,
            label_size=1,
            mode="supervised",
            n_classes=self.cfg.n_classes,
            ignore_lbl=-1,
            indices_dict=self.val_idx,
            dim=self.cfg.dim,
        )

    def get_data_statistics(self) -> Tuple[float, float]:
        """
        Returns the data mean and standard deviation.
        This function is called from the model during setup, to register the statistics as buffers.
        Mean and std are then used during inference for normalization.
        """
        return self.data_mean, self.data_std

    def _cache_dataset_splits(self):
        assert self.cfg.cache_dir is not None, (
            "cache_dir must be specified to cache dataset splits."
        )
        assert Path(self.cfg.cache_dir).resolve() != Path(self.cfg.data_dir).resolve(), (
            "cache_dir and data_dir must not be the same."
        )
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        # Save train_idx, val_idx, data_mean, data_std to cache_dir
        np.save(Path(self.cfg.cache_dir) / "train_idx.npy", self.train_idx)
        np.save(Path(self.cfg.cache_dir) / "val_idx.npy", self.val_idx)
        np.save(Path(self.cfg.cache_dir) / "data_mean.npy", self.data_mean)
        np.save(Path(self.cfg.cache_dir) / "data_std.npy", self.data_std)
        for key in self.cfg.train_keys:
            tiff.imwrite(
                Path(self.cfg.cache_dir) / f"{key}_normalized.tif",
                self.images[key].astype(np.float16),
            )
            tiff.imwrite(
                Path(self.cfg.cache_dir) / f"{key}_labels.tif",
                self.labels[key].astype(np.float16),
            )

    def _load_cached_dataset_splits(self):
        if not Path(self.cfg.cache_dir).resolve().exists():
            raise FileNotFoundError(
                f"Cache directory {self.cfg.cache_dir} does not exist."
            )
        # Load cached splits, mean, std from cache_dir
        self.train_idx = np.load(
            Path(self.cfg.cache_dir) / "train_idx.npy", allow_pickle=True
        ).item()
        self.val_idx = np.load(
            Path(self.cfg.cache_dir) / "val_idx.npy", allow_pickle=True
        ).item()
        self.data_mean = np.load(Path(self.cfg.cache_dir) / "data_mean.npy")
        self.data_std = np.load(Path(self.cfg.cache_dir) / "data_std.npy")
        self.images = {}
        self.labels = {}
        for key in self.cfg.train_keys:
            self.images[key] = tiff.imread(
                Path(self.cfg.cache_dir) / f"{key}_normalized.tif"
            ).astype(np.float16)
            self.labels[key] = tiff.imread(
                Path(self.cfg.cache_dir) / f"{key}_labels.tif"
            ).astype(np.float16)
        print(f"Loaded cached dataset splits from {self.cfg.cache_dir}.")

    def _load_original_img_lbls(self):
        keys = self.cfg.train_keys
        data_dir = self.cfg.data_dir
        img_paths = [Path(data_dir) / key / "image.tif" for key in keys]
        lbl_paths = [Path(data_dir) / key / "labs.tif" for key in keys]
        imgs = {
            key: tiff.imread(path).astype(np.float16)
            for key, path in zip(keys, img_paths)
        }
        lbls = {
            key: tiff.imread(path).astype(np.float16)
            for key, path in zip(keys, lbl_paths)
        }
        return imgs, lbls

    def _shuffle_and_split(
        self, labels: Dict[str, np.ndarray], shuffle=True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Splits dataset into training and validation sets (80% - 20%).
        And shuffles the indices if specified.

        Args:
            labels (dict): Dictionary mapping keys to label arrays.
            shuffle (bool): Whether to shuffle the indices before splitting.
        Returns:
            train_idx (dict): Dictionary mapping keys to training indices.
            val_idx (dict): Dictionary mapping keys to validation indices.

        """
        # Function adapted from original boilerplate.dataloader script to preserve reproducible splits
        keys = self.cfg.train_keys
        train_idx, val_idx = {}, {}
        # WARNING: Here RandomState is used to ensure backward compatibility with paper results (that were generated with np.random.seed)
        # DO NOT CHANGE TO np.random.default_rng unless you are ok with having different dataset splits than in the paper
        rng = np.random.RandomState(self.cfg.seed)
        for key in keys:
            # Create a mask for valid indices where labels are not all -1
            # -1 indicates outside of the cell
            #
            valid_indices = np.where(~np.all(labels[key] == -1, axis=(1, 2)))[0]
            total_samples = valid_indices.shape[0]
            if shuffle:
                rng.shuffle(valid_indices)  # Shuffles in place

            # Compute split index
            split_idx = int(0.80 * total_samples)

            # Split the indices
            train_idx[key] = valid_indices[:split_idx]
            val_idx[key] = valid_indices[split_idx:]
        return train_idx, val_idx

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=ModeAwareBalancedAnchorBatchSampler(
                self.train_dataset,
                total_patches_per_batch=self.train_cfg.batch_size,
                shuffle=True,
            ),
            collate_fn=flex_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=ModeAwareBalancedAnchorBatchSampler(
                self.val_dataset,
                total_patches_per_batch=self.train_cfg.batch_size,
                shuffle=False,
            ),
            collate_fn=flex_collate,
        )

    def set_mode(self, mode: str):
        """Switch between supervised and semisupervised modes."""
        print(f"Switching datamodule mode to {mode}...")
        self.train_dataset.set_mode(mode)

    def increase_radius(self):
        """Increase the radius used for semisupervised sampling."""
        print("Increasing semisupervised sampling radius...")
        self.train_dataset.increase_radius()
