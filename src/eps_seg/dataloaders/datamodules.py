from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import lightning as L
from torch.utils.data import DataLoader

from eps_seg.config.datasets import BaseEPSDatasetConfig
from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.caching import DatasetCache
from eps_seg.dataloaders.datasets import PredictionDataset, PseudoLabelDataset, SemisupervisedDataset
from eps_seg.dataloaders.samplers import (
    BalancedScheduledBatchSampler,
    ModeAwareBalancedAnchorBatchSampler,
    PseudoEpochDistributedParallelBatchSampler,
)
from eps_seg.dataloaders.utils import flex_collate


class EPSSegDataModule(L.LightningDataModule):
    """
    Base class for EPS-Seg DataModules.

    This Datamodule is compatible with Cache-v2.

    It handles data preparation, loading, and batching for training, validation, testing, and prediction stages.


    """

    def __init__(
        self,
        cfg: BaseEPSDatasetConfig,
        train_cfg: TrainConfig,
        scheduler_path: Optional[Path] = None,
        scheduler_stage_index: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.scheduler_path = Path(scheduler_path) if scheduler_path is not None else None
        self.scheduler_stage_index = scheduler_stage_index
        self.cache_dir = cfg.get_cache_folder()
        self.cache = DatasetCache(cfg, train_cfg=train_cfg)

        self.data: Dict[str, object] = {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        """
        Prepare data for training, validation, testing, and prediction.
        """
        if not self.cfg.enable_cache:
            print("[DataModule] Training cache is mandatory. Ignoring enable_cache=false and using cache-v2.")
        try:
            print(f"[DataModule] Checking cache at {self.cache_dir}...")
            self.cache.check_cache_dir()
            print("[DataModule] Cache is valid. Reusing cached fold data.")
        except Exception:
            print("[DataModule] Cache is missing or incomplete. Building fold cache...")
            self.cache.build_cache()
            print("[DataModule] Cache build complete.")

    def get_data_statistics(self) -> Tuple[float, float]:
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
            shuffle=False,
            batches_per_pseudoepoch=self.train_cfg.batches_per_pseudoepoch,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
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

    def setup(self, stage):
        super().setup(stage)
        print(f"[DataModule] Setting up stage='{stage}' for fold {self.cfg.fold}/{self.cfg.max_folds - 1}...")

        if stage in ["fit", "validate"]:
            data = self._load_cached_dataset_splits(split="trainval")
        else:
            data = self._load_original_dataset_split(split=stage)
        self.data.update(data)

        if stage in ["fit", "validate"]:
            print(
                f"[DataModule] Loaded fold payload | train_anchors={len(self.data.get('train_anchor_records', []))} "
                f"| val_anchors={len(self.data.get('val_anchor_records', []))} "
                f"| mean={self.data['data_mean']:.4f} std={self.data['data_std']:.4f}"
            )

        if stage in ["fit"]:
            self.train_dataset = PseudoLabelDataset(
                images=self.data["trainval_images"],
                labels=self.data["trainval_labels"],
                patch_size=self.cfg.patch_size,
                label_size=1,
                n_classes=self.cfg.n_classes,
                ignore_lbl=-1,
                dim=self.cfg.dim,
                seed=self.cfg.seed,
                samples_per_class=self.cfg.samples_per_class,
                anchor_records=self.data["train_anchor_records"],
                train_substacks=self.data["train_substacks"],
                scheduler_path=self.scheduler_path,
                stage_index=self.scheduler_stage_index,
                train_cfg=self.train_cfg,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = SemisupervisedDataset(
                images=self.data["trainval_images"],
                labels=self.data["trainval_labels"],
                patch_size=self.cfg.patch_size,
                label_size=1,
                mode="supervised",
                n_classes=self.cfg.n_classes,
                ignore_lbl=-1,
                dim=self.cfg.dim,
                seed=self.cfg.seed,
                samples_per_class=self.cfg.samples_per_class,
                n_neighbors=self.cfg.n_neighbors,
                anchor_records=self.data["val_anchor_records"],
            )
        if stage in ["test", "predict"]:
            if stage == "test":
                self.test_dataset = PredictionDataset(
                    images=self.data["test_images"],
                    labels=self.data["test_labels"],
                    keys=self.cfg.test_keys,
                    masks=self._build_slice_mask(stage=stage),
                    patch_size=self.cfg.patch_size,
                    dim=self.cfg.dim,
                    ignore_lbl=-1,
                )
            else:
                self.predict_dataset = PredictionDataset(
                    images=self.data["test_images"],
                    labels=self.data["test_labels"],
                    keys=self.cfg.test_keys,
                    masks=self._build_slice_mask(stage=stage),
                    patch_size=self.cfg.patch_size,
                    dim=self.cfg.dim,
                    ignore_lbl=-1,
                )

    def _build_slice_mask(self, stage: str) -> Dict[str, Tuple[slice, slice, slice]]:
        masks = {}
        keys = self.cfg.test_keys
        assert stage in ["test", "predict"], "Stage must be either 'test' or 'predict'."

        for k, key in enumerate(keys):
            cs = (
                self.cfg.test_center_slices[k]
                if stage == "test"
                else self.cfg.predict_center_slices[k]
            )
            hd = (
                self.cfg.test_half_depths[k]
                if stage == "test"
                else self.cfg.predict_half_depths[k]
            )
            stp = self.cfg.test_steppings[k] if stage == "test" else 1
            if cs is None:
                masks[key] = (
                    slice(None, None, stp),
                    slice(None, None, stp),
                    slice(None, None, stp),
                )
            else:
                masks[key] = (
                    slice(cs - hd, cs + hd + 1, stp),
                    slice(None, None, stp),
                    slice(None, None, stp),
                )
        return masks

    def _load_original_dataset_split(self, split: Literal["trainval", "test", "predict"]):
        result = {}
        if split == "trainval":
            result.update(self.cache.load_uncached_trainval_fold(self.cfg.fold))
        elif split in ["test", "predict"]:
            result["test_images"], result["test_labels"] = self.cache.load_source_and_labels(
                self.cfg.test_keys
            )
        return result

    def _load_cached_dataset_splits(
        self, split: Literal["trainval", "test", "predict"]
    ) -> Dict[str, object]:
        if split != "trainval":
            raise NotImplementedError(
                "Cached loading is currently only implemented for 'trainval' split. "
                "Please load original data for other splits."
            )
        return self.cache.load_cached_trainval_fold(self.cfg.fold)
