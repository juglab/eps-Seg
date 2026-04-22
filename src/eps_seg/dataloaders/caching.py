import csv
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import tifffile as tiff
import yaml
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from eps_seg.config.datasets import BaseEPSDatasetConfig


class DatasetCache:
    """
    Helper class responsible for cache validation, cache building, and fold
    payload loading for EPS-Seg datasets.

    The datamodule uses this class to keep cache-related file handling and fold
    materialization separate from Lightning orchestration.
    """

    def __init__(self, cfg: BaseEPSDatasetConfig):
        self.cfg = cfg
        self.cache_dir = cfg.get_cache_folder()
        self.cache_paths = {
            "manifest": self.cache_dir / "manifest.yaml",
            "substacks": self.cache_dir / "substacks.csv",
            "sampled_coords": self.cache_dir / "sampled_coords.csv",
            "fold_assignments": self.cache_dir / "fold_assignments.csv",
            "fold_stats": self.cache_dir / "fold_stats.csv",
        }

    def fold_dir(self, fold: int) -> Path:
        return self.cache_dir / f"fold_{fold}"

    def fold_normalized_path(self, fold: int, key: str) -> Path:
        return self.fold_dir(fold) / "normalized" / f"{key}.tif"

    def fold_labels_path(self, fold: int, key: str) -> Path:
        return self.fold_dir(fold) / "labels" / f"{key}.tif"

    def check_cache_dir(self):
        """
        Ensure the shared cache directory exists and contains all fold artifacts
        required to load the selected dataset configuration.
        """
        for path in self.cache_paths.values():
            if not path.exists():
                raise FileNotFoundError(f"Cache file {path} does not exist. Recaching required.")

        manifest = self.read_manifest()
        expected_manifest = self.build_manifest()
        manifest_keys = [
            "name",
            "seed",
            "substacking",
            "max_folds",
            "train_keys",
            "test_keys",
            "patch_size",
            "dim",
            "ignore_lbl",
            "train_to_val_ratio",
            "samples_per_class",
        ]
        for key in manifest_keys:
            if manifest.get(key) != expected_manifest.get(key):
                raise ValueError(
                    f"Cache manifest mismatch for '{key}': "
                    f"{manifest.get(key)} != {expected_manifest.get(key)}."
                )

        for fold in range(self.cfg.max_folds):
            for key in self.cfg.train_keys:
                if not self.fold_normalized_path(fold, key).exists():
                    raise FileNotFoundError(
                        f"Missing normalized cache file {self.fold_normalized_path(fold, key)}."
                    )
                if not self.fold_labels_path(fold, key).exists():
                    raise FileNotFoundError(
                        f"Missing label cache file {self.fold_labels_path(fold, key)}."
                    )

    def build_cache(self):
        """Build and persist the full shared cache for the current dataset configuration."""
        payload = self.build_trainval_cache_payload()
        self.write_cached_dataset_splits(payload)

    def build_manifest(self) -> dict:
        return {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "substacking": self.cfg.substacking,
            "max_folds": self.cfg.max_folds,
            "train_keys": list(self.cfg.train_keys),
            "test_keys": list(self.cfg.test_keys),
            "patch_size": self.cfg.patch_size,
            "dim": self.cfg.dim,
            "ignore_lbl": -1,
            "train_to_val_ratio": self.cfg.train_to_val_ratio,
            "samples_per_class": dict(self.cfg.samples_per_class or {}),
            "load_train_coords_from": self.cfg.load_train_coords_from,
            "load_val_coords_from": self.cfg.load_val_coords_from,
        }

    def read_manifest(self) -> dict:
        with open(self.cache_paths["manifest"], "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def write_manifest(self, manifest: dict):
        with open(self.cache_paths["manifest"], "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=True)

    def write_csv(self, path: Path, rows: List[dict], fieldnames: List[str]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def read_csv(self, path: Path) -> List[dict]:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def patch_offset(self) -> int:
        return self.cfg.patch_size // 2 - 1

    def is_valid_anchor_coord(
        self, labels: Dict[str, np.ndarray], name: str, z: int, y: int, x: int
    ) -> bool:
        """
        Check whether a coordinate can be used as the center of a training patch.
        """
        Z, H, W = labels[name].shape
        offset = self.patch_offset()
        valid = offset <= y < H - offset - 1 and offset <= x < W - offset - 1
        if self.cfg.dim == 3:
            valid = valid and (offset <= z < Z - offset - 1)
        return valid and labels[name][z, y, x] != -1

    def load_source_and_labels(
        self, keys: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load the original images and labels from data_dir according to the
        folder structure defined in the dataset config.
        """
        img_lbl_paths = self.cfg.get_image_label_paths(keys)
        imgs = {}
        lbls = {}
        for key, (img_path, lbl_path) in tqdm(
            img_lbl_paths.items(),
            desc="Loading source/GT stacks",
            leave=False,
        ):
            imgs[key] = tiff.imread(img_path).astype(np.float16)
            lbls[key] = tiff.imread(lbl_path).astype(np.float16)
        return imgs, lbls

    def compute_fold_statistics(
        self, images: Dict[str, np.ndarray], train_substacks: List[dict]
    ) -> Tuple[float, float]:
        """
        Compute mean and standard deviation over the image voxels assigned to the
        training substacks of a fold.
        """
        selected_chunks = []
        for substack in train_substacks:
            key = substack["stack_name"]
            z_start = substack["z_start"]
            z_stop = substack["z_stop"]
            selected_chunks.append(images[key][z_start:z_stop].reshape(-1))

        if not selected_chunks:
            raise ValueError("Cannot compute fold statistics without training substacks.")

        all_elements = np.concatenate(selected_chunks).astype(np.float32)
        data_mean = float(np.mean(all_elements))
        data_std = float(np.std(all_elements))
        if data_std == 0.0:
            data_std = 1.0
        return data_mean, data_std

    def normalize_images(
        self, images: Dict[str, np.ndarray], data_mean: float, data_std: float
    ) -> Dict[str, np.ndarray]:
        """
        Normalize the images using the provided mean and standard deviation and
        return a new dictionary of normalized arrays.
        """
        normalized_images = {}
        for key in tqdm(self.cfg.train_keys, "Normalizing data"):
            normalized_images[key] = ((images[key] - data_mean) / data_std).astype(np.float16)
        return normalized_images

    def build_valid_substacks(self, labels: Dict[str, np.ndarray]) -> List[dict]:
        """
        Build contiguous valid z-substacks from each training volume.

        A valid slice is any z-index whose label plane is not entirely -1.
        Remainder chunks smaller than substacking are discarded.
        """
        substacks: List[dict] = []
        next_substack_id = 0

        for key in tqdm(self.cfg.train_keys, desc="Building valid substacks", leave=False):
            valid_mask = ~np.all(labels[key] == -1, axis=(-2, -1))
            valid_indices = np.flatnonzero(valid_mask)
            if valid_indices.size == 0:
                continue

            run_start = valid_indices[0]
            prev = valid_indices[0]
            runs: List[Tuple[int, int]] = []
            for current in valid_indices[1:]:
                if current != prev + 1:
                    runs.append((run_start, prev + 1))
                    run_start = current
                prev = current
            runs.append((run_start, prev + 1))

            for z_start, z_stop in runs:
                cursor = z_start
                while cursor + self.cfg.substacking <= z_stop:
                    block_start = cursor
                    block_stop = cursor + self.cfg.substacking
                    if np.all(labels[key][block_start:block_stop] == -1):
                        cursor = block_stop
                        continue
                    substacks.append(
                        {
                            "substack_id": next_substack_id,
                            "stack_name": key,
                            "z_start": block_start,
                            "z_stop": block_stop,
                            "depth": block_stop - block_start,
                        }
                    )
                    next_substack_id += 1
                    cursor = block_stop

        if not substacks:
            raise ValueError("No valid substacks were found for the requested dataset.")
        return substacks

    def sample_coords_for_class(
        self, stack: np.ndarray, class_idx: int, rng: random.Random
    ) -> List[Tuple[int, int]]:
        """
        Return up to N (y, x) coordinates for a class from one 2D label slice.
        """
        n_needed = (self.cfg.samples_per_class or {}).get(class_idx, 1)
        label_coords = np.argwhere(stack == class_idx)
        if len(label_coords) == 0 or n_needed <= 0:
            return []

        n_take = min(len(label_coords), n_needed)
        sampled_idx = rng.sample(range(len(label_coords)), n_take)
        sampled = label_coords[sampled_idx]
        return [(int(y), int(x)) for (y, x) in sampled]

    def sample_canonical_anchor_records(
        self, labels: Dict[str, np.ndarray], substacks: List[dict]
    ) -> List[dict]:
        """
        Sample canonical GT anchors once within each substack, before any fold
        assignment takes place.
        """
        rng = random.Random(self.cfg.seed)
        records: List[dict] = []
        next_coord_id = 0

        for substack in tqdm(substacks, desc="Sampling canonical anchors", leave=False):
            key = substack["stack_name"]
            z_start = substack["z_start"]
            z_stop = substack["z_stop"]
            for z in range(z_start, z_stop):
                stack = labels[key][z]
                for class_idx in range(self.cfg.n_classes):
                    sampled_coords = self.sample_coords_for_class(stack, class_idx, rng)
                    for y, x in sampled_coords:
                        if not self.is_valid_anchor_coord(labels, key, z, y, x):
                            continue
                        records.append(
                            {
                                "coord_id": next_coord_id,
                                "stack_name": key,
                                "z": int(z),
                                "y": int(y),
                                "x": int(x),
                                "gt_label": int(labels[key][z, y, x]),
                                "substack_id": int(substack["substack_id"]),
                            }
                        )
                        next_coord_id += 1
        return records

    def read_external_coords_csv(
        self,
        csv_path: Path,
        labels: Dict[str, np.ndarray],
    ) -> List[dict]:
        """
        Read and validate an external coordinate CSV with columns name,z,y,x.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"External coordinate CSV {csv_path} does not exist.")

        rows: List[dict] = []
        seen_coords = set()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_columns = {"name", "z", "y", "x"}
            if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
                raise ValueError(
                    f"External coordinate CSV {csv_path} must have exactly columns "
                    f"{sorted(expected_columns)}. Found {reader.fieldnames}."
                )

            for raw_row in tqdm(
                reader,
                desc=f"Loading external coords ({csv_path.name})",
                leave=False,
            ):
                stack_name = raw_row["name"]
                if stack_name not in self.cfg.train_keys:
                    raise ValueError(
                        f"Coordinate CSV {csv_path} references unknown stack '{stack_name}'."
                    )

                z = int(raw_row["z"])
                y = int(raw_row["y"])
                x = int(raw_row["x"])
                Z, H, W = labels[stack_name].shape
                if not (0 <= z < Z and 0 <= y < H and 0 <= x < W):
                    raise ValueError(
                        f"Out-of-bounds coordinate in {csv_path}: {(stack_name, z, y, x)} "
                        f"for volume shape {(Z, H, W)}."
                    )
                if not self.is_valid_anchor_coord(labels, stack_name, z, y, x):
                    raise ValueError(
                        f"Invalid anchor coordinate in {csv_path}: {(stack_name, z, y, x)}. "
                        "Coordinates must be inside the valid patchable region and not ignore-labeled."
                    )

                coord_key = (stack_name, z, y, x)
                if coord_key in seen_coords:
                    raise ValueError(f"Duplicate coordinate found in external CSV {csv_path}: {coord_key}")
                seen_coords.add(coord_key)

                rows.append(
                    {
                        "stack_name": stack_name,
                        "z": z,
                        "y": y,
                        "x": x,
                        "gt_label": int(labels[stack_name][z, y, x]),
                    }
                )
        return rows

    def load_external_anchor_records(
        self,
        labels: Dict[str, np.ndarray],
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Load externally sampled train/validation coordinates from CSV files.

        External import is only supported in 1-fold mode with substacking=1.
        Each referenced z-slice becomes one substack descriptor, and each such
        substack must belong entirely to either train or validation.
        """
        assert self.cfg.load_train_coords_from is not None
        assert self.cfg.load_val_coords_from is not None

        train_rows = self.read_external_coords_csv(Path(self.cfg.load_train_coords_from), labels)
        val_rows = self.read_external_coords_csv(Path(self.cfg.load_val_coords_from), labels)

        train_coords = {
            (row["stack_name"], row["z"], row["y"], row["x"])
            for row in train_rows
        }
        val_coords = {
            (row["stack_name"], row["z"], row["y"], row["x"])
            for row in val_rows
        }
        overlap = train_coords & val_coords
        if overlap:
            raise ValueError(
                "External train/val CSVs contain overlapping coordinates. "
                f"Example overlap: {sorted(overlap)[:5]}"
            )

        substack_descriptors: List[dict] = []
        substack_id_by_slice: Dict[Tuple[str, int], int] = {}
        next_substack_id = 0
        split_by_substack: Dict[int, str] = {}

        def register_substack(stack_name: str, z: int, split: str) -> int:
            nonlocal next_substack_id
            key = (stack_name, z)
            if key not in substack_id_by_slice:
                substack_id_by_slice[key] = next_substack_id
                substack_descriptors.append(
                    {
                        "substack_id": next_substack_id,
                        "stack_name": stack_name,
                        "z_start": z,
                        "z_stop": z + 1,
                        "depth": 1,
                    }
                )
                next_substack_id += 1
            substack_id = substack_id_by_slice[key]
            existing_split = split_by_substack.get(substack_id)
            if existing_split is not None and existing_split != split:
                raise ValueError(
                    "A slice referenced by external coordinates appears in both "
                    f"train and val CSVs: {(stack_name, z)}"
                )
            split_by_substack[substack_id] = split
            return substack_id

        sampled_coords: List[dict] = []
        next_coord_id = 0
        for split, rows in (("train", train_rows), ("val", val_rows)):
            for row in rows:
                substack_id = register_substack(row["stack_name"], row["z"], split)
                sampled_coords.append(
                    {
                        "coord_id": next_coord_id,
                        "stack_name": row["stack_name"],
                        "z": row["z"],
                        "y": row["y"],
                        "x": row["x"],
                        "gt_label": row["gt_label"],
                        "substack_id": substack_id,
                    }
                )
                next_coord_id += 1

        if not sampled_coords:
            raise ValueError("External coordinate import produced no canonical anchors.")

        fold_assignments = [
            {
                "fold": 0,
                "substack_id": descriptor["substack_id"],
                "assigned_split": split_by_substack[descriptor["substack_id"]],
            }
            for descriptor in substack_descriptors
        ]

        return substack_descriptors, sampled_coords, fold_assignments

    def build_fold_assignments(self, substacks: List[dict]) -> List[dict]:
        """
        Build train/validation assignments at the substack level.
        """
        rng = random.Random(self.cfg.seed)
        shuffled_substacks = list(substacks)
        rng.shuffle(shuffled_substacks)

        assignments: List[dict] = []
        if self.cfg.max_folds == 1:
            grouped: Dict[str, List[dict]] = {}
            for substack in shuffled_substacks:
                grouped.setdefault(substack["stack_name"], []).append(substack)

            for group_substacks in grouped.values():
                split_idx = int(self.cfg.train_to_val_ratio * len(group_substacks))
                for substack in group_substacks[:split_idx]:
                    assignments.append(
                        {
                            "fold": 0,
                            "substack_id": int(substack["substack_id"]),
                            "assigned_split": "train",
                        }
                    )
                for substack in group_substacks[split_idx:]:
                    assignments.append(
                        {
                            "fold": 0,
                            "substack_id": int(substack["substack_id"]),
                            "assigned_split": "val",
                        }
                    )
            return assignments

        strata = [substack["stack_name"] for substack in shuffled_substacks]
        stratum_counts = {label: strata.count(label) for label in set(strata)}
        if min(stratum_counts.values()) < self.cfg.max_folds:
            raise ValueError(
                "Not enough substacks per source stack to run StratifiedKFold "
                f"with n_splits={self.cfg.max_folds}: {stratum_counts}"
            )

        skf = StratifiedKFold(n_splits=self.cfg.max_folds, shuffle=False)
        shuffled_ids = np.array([substack["substack_id"] for substack in shuffled_substacks])
        dummy_x = np.arange(len(shuffled_substacks))

        for fold, (train_idx, val_idx) in enumerate(skf.split(dummy_x, strata)):
            for substack_id in shuffled_ids[train_idx]:
                assignments.append(
                    {
                        "fold": int(fold),
                        "substack_id": int(substack_id),
                        "assigned_split": "train",
                    }
                )
            for substack_id in shuffled_ids[val_idx]:
                assignments.append(
                    {
                        "fold": int(fold),
                        "substack_id": int(substack_id),
                        "assigned_split": "val",
                    }
                )

        return assignments

    def compute_fold_stats_rows(
        self,
        images: Dict[str, np.ndarray],
        substacks: List[dict],
        sampled_coords: List[dict],
        fold_assignments: List[dict],
    ) -> Tuple[List[dict], Dict[int, Dict[str, np.ndarray]]]:
        """
        Compute per-fold statistics and normalized data payloads.
        """
        substacks_by_id = {substack["substack_id"]: substack for substack in substacks}
        anchors_by_substack: Dict[int, List[dict]] = {}
        for record in sampled_coords:
            anchors_by_substack.setdefault(record["substack_id"], []).append(record)

        assignments_by_fold: Dict[int, Dict[int, str]] = {}
        for row in fold_assignments:
            assignments_by_fold.setdefault(int(row["fold"]), {})[int(row["substack_id"])] = row[
                "assigned_split"
            ]

        fold_stats_rows: List[dict] = []
        normalized_by_fold: Dict[int, Dict[str, np.ndarray]] = {}

        for fold in tqdm(range(self.cfg.max_folds), desc="Computing fold statistics", leave=False):
            fold_map = assignments_by_fold[fold]
            train_substacks = [
                substacks_by_id[substack_id]
                for substack_id, split in fold_map.items()
                if split == "train"
            ]
            val_substacks = [
                substacks_by_id[substack_id]
                for substack_id, split in fold_map.items()
                if split == "val"
            ]

            data_mean, data_std = self.compute_fold_statistics(images, train_substacks)
            normalized_by_fold[fold] = self.normalize_images(images, data_mean, data_std)

            row = {
                "fold": fold,
                "data_mean": data_mean,
                "data_std": data_std,
                "n_train_substacks": len(train_substacks),
                "n_val_substacks": len(val_substacks),
            }

            train_anchors: List[dict] = []
            val_anchors: List[dict] = []
            for substack_id, split in fold_map.items():
                if split == "train":
                    train_anchors.extend(anchors_by_substack.get(substack_id, []))
                else:
                    val_anchors.extend(anchors_by_substack.get(substack_id, []))

            row["n_train_coords"] = len(train_anchors)
            row["n_val_coords"] = len(val_anchors)

            for class_idx in range(self.cfg.n_classes):
                row[f"train_class_{class_idx}_count"] = sum(
                    1 for record in train_anchors if int(record["gt_label"]) == class_idx
                )
                row[f"val_class_{class_idx}_count"] = sum(
                    1 for record in val_anchors if int(record["gt_label"]) == class_idx
                )

            for key in self.cfg.train_keys:
                row[f"train_stack_{key}_count"] = sum(
                    1 for record in train_anchors if record["stack_name"] == key
                )
                row[f"val_stack_{key}_count"] = sum(
                    1 for record in val_anchors if record["stack_name"] == key
                )

            fold_stats_rows.append(row)

        return fold_stats_rows, normalized_by_fold

    def build_trainval_cache_payload(self) -> Dict[str, object]:
        """
        Build the canonical-anchor training/validation payload from the original dataset.
        """
        images, labels = self.load_source_and_labels(self.cfg.train_keys)
        if self.cfg.load_train_coords_from and self.cfg.load_val_coords_from:
            print("[DataModule] Importing canonical anchors from external train/val CSVs...")
            substacks, sampled_coords, fold_assignments = self.load_external_anchor_records(
                labels
            )
        else:
            print("[DataModule] Sampling canonical anchors from raw labels...")
            substacks = self.build_valid_substacks(labels)
            sampled_coords = self.sample_canonical_anchor_records(labels, substacks)
            fold_assignments = self.build_fold_assignments(substacks)
        fold_stats, normalized_by_fold = self.compute_fold_stats_rows(
            images=images,
            substacks=substacks,
            sampled_coords=sampled_coords,
            fold_assignments=fold_assignments,
        )

        print(
            f"[DataModule] Prepared cache payload | substacks={len(substacks)} "
            f"| canonical_coords={len(sampled_coords)} | folds={self.cfg.max_folds}"
        )

        return {
            "manifest": self.build_manifest(),
            "substacks": substacks,
            "sampled_coords": sampled_coords,
            "fold_assignments": fold_assignments,
            "fold_stats": fold_stats,
            "normalized_by_fold": normalized_by_fold,
            "labels": labels,
        }

    def write_cached_dataset_splits(self, data_to_cache: Dict[str, object]):
        """
        Persist the shared canonical-anchor cache to disk.
        """
        assert self.cache_dir is not None, "cache_dir must be specified to cache dataset splits."
        assert Path(self.cache_dir).resolve() != Path(self.cfg.data_dir).resolve(), (
            "cache_dir and data_dir must not be the same."
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.write_manifest(data_to_cache["manifest"])
        self.write_csv(
            self.cache_paths["substacks"],
            data_to_cache["substacks"],
            ["substack_id", "stack_name", "z_start", "z_stop", "depth"],
        )
        self.write_csv(
            self.cache_paths["sampled_coords"],
            data_to_cache["sampled_coords"],
            ["coord_id", "stack_name", "z", "y", "x", "gt_label", "substack_id"],
        )
        self.write_csv(
            self.cache_paths["fold_assignments"],
            data_to_cache["fold_assignments"],
            ["fold", "substack_id", "assigned_split"],
        )

        fold_stat_fieldnames = sorted(
            {key for row in data_to_cache["fold_stats"] for key in row.keys()},
            key=lambda item: (item != "fold", item),
        )
        self.write_csv(self.cache_paths["fold_stats"], data_to_cache["fold_stats"], fold_stat_fieldnames)

        for fold in tqdm(range(self.cfg.max_folds), desc="Writing fold cache", leave=False):
            normalized_dir = self.fold_dir(fold) / "normalized"
            labels_dir = self.fold_dir(fold) / "labels"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for key in tqdm(self.cfg.train_keys, desc=f"Fold {fold} TIFFs", leave=False):
                tiff.imwrite(
                    self.fold_normalized_path(fold, key),
                    data_to_cache["normalized_by_fold"][fold][key].astype(np.float16),
                )
                tiff.imwrite(
                    self.fold_labels_path(fold, key),
                    data_to_cache["labels"][key].astype(np.float16),
                )

    def deserialize_substacks(self, rows: List[dict]) -> List[dict]:
        return [
            {
                "substack_id": int(row["substack_id"]),
                "stack_name": row["stack_name"],
                "z_start": int(row["z_start"]),
                "z_stop": int(row["z_stop"]),
                "depth": int(row["depth"]),
            }
            for row in rows
        ]

    def deserialize_sampled_coords(self, rows: List[dict]) -> List[dict]:
        return [
            {
                "coord_id": int(row["coord_id"]),
                "stack_name": row["stack_name"],
                "z": int(row["z"]),
                "y": int(row["y"]),
                "x": int(row["x"]),
                "gt_label": int(row["gt_label"]),
                "substack_id": int(row["substack_id"]),
            }
            for row in rows
        ]

    def deserialize_fold_assignments(self, rows: List[dict]) -> List[dict]:
        return [
            {
                "fold": int(row["fold"]),
                "substack_id": int(row["substack_id"]),
                "assigned_split": row["assigned_split"],
            }
            for row in rows
        ]

    def deserialize_fold_stats(self, rows: List[dict]) -> List[dict]:
        parsed_rows = []
        for row in rows:
            parsed = {}
            for key, value in row.items():
                if key == "fold":
                    parsed[key] = int(value)
                elif key in {"data_mean", "data_std"}:
                    parsed[key] = float(value)
                elif value == "":
                    parsed[key] = value
                else:
                    parsed[key] = int(value)
            parsed_rows.append(parsed)
        return parsed_rows

    def select_fold_payload(
        self,
        images: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        substacks: List[dict],
        sampled_coords: List[dict],
        fold_assignments: List[dict],
        fold_stats: List[dict],
        fold: int,
    ) -> Dict[str, object]:
        """
        Select the data structures required to instantiate the train/validation
        datasets for a single fold.
        """
        substacks_by_id = {substack["substack_id"]: substack for substack in substacks}
        selected_assignments = {
            row["substack_id"]: row["assigned_split"]
            for row in fold_assignments
            if row["fold"] == fold
        }

        train_records = []
        val_records = []
        train_substacks = []
        val_substacks = []
        for substack in substacks:
            split = selected_assignments.get(substack["substack_id"])
            if split == "train":
                train_substacks.append(substack)
            elif split == "val":
                val_substacks.append(substack)

        for record in sampled_coords:
            split = selected_assignments.get(record["substack_id"])
            if split is None:
                continue
            descriptor = substacks_by_id[record["substack_id"]]
            enriched_record = {
                **record,
                "z_start": descriptor["z_start"],
                "z_stop": descriptor["z_stop"],
                "depth": descriptor["depth"],
            }
            if split == "train":
                train_records.append(enriched_record)
            else:
                val_records.append(enriched_record)

        fold_stat_row = next(row for row in fold_stats if row["fold"] == fold)
        return {
            "trainval_images": images,
            "trainval_labels": labels,
            "data_mean": fold_stat_row["data_mean"],
            "data_std": fold_stat_row["data_std"],
            "train_anchor_records": train_records,
            "val_anchor_records": val_records,
            "train_substacks": train_substacks,
            "val_substacks": val_substacks,
        }

    def load_uncached_trainval_fold(self, fold: int) -> Dict[str, object]:
        """
        Build and return a train/validation payload directly from source data.
        """
        payload = self.build_trainval_cache_payload()
        fold_images = payload["normalized_by_fold"][fold]
        return self.select_fold_payload(
            images=fold_images,
            labels=payload["labels"],
            substacks=payload["substacks"],
            sampled_coords=payload["sampled_coords"],
            fold_assignments=payload["fold_assignments"],
            fold_stats=payload["fold_stats"],
            fold=fold,
        )

    def load_cached_trainval_fold(self, fold: int) -> Dict[str, object]:
        """
        Load cached train/validation images, labels, canonical anchors, and
        fold statistics for the selected fold.
        """
        if not self.cache_dir.resolve().exists():
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist.")

        substacks = self.deserialize_substacks(self.read_csv(self.cache_paths["substacks"]))
        sampled_coords = self.deserialize_sampled_coords(
            self.read_csv(self.cache_paths["sampled_coords"])
        )
        fold_assignments = self.deserialize_fold_assignments(
            self.read_csv(self.cache_paths["fold_assignments"])
        )
        fold_stats = self.deserialize_fold_stats(self.read_csv(self.cache_paths["fold_stats"]))

        images = {}
        labels = {}
        for key in self.cfg.train_keys:
            images[key] = tiff.imread(self.fold_normalized_path(fold, key)).astype(np.float16)
            labels[key] = tiff.imread(self.fold_labels_path(fold, key)).astype(np.float16)

        print(f"[DataModule] Loaded cached train/validation fold data from {self.cache_dir}.")
        return self.select_fold_payload(
            images=images,
            labels=labels,
            substacks=substacks,
            sampled_coords=sampled_coords,
            fold_assignments=fold_assignments,
            fold_stats=fold_stats,
            fold=fold,
        )
