import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=1,
        n_classes=4,
        ignore_lbl=-1,
        indices_dict=None,
        dim=2,
        seed=42,
        samples_per_class: Optional[Dict[int, int]] = None,
        scheduler_path=None,
        stage_index: int = 0,
        confidence_threshold: float = 0.75,
        anchor_records: Optional[List[Dict[str, int]]] = None,
        train_substacks: Optional[List[Dict[str, int]]] = None,
    ):
        """
        Dataset storing the full staged scheduler used by the refactored
        training loop.

        Stage-0 rows can be initialized either from canonical cached anchor
        records (cache v2 path) or from legacy per-slice sampling
        ``indices_dict`` (backward compatibility path).
        """
        self.images = images
        self.labels = labels
        self.indices_dict = indices_dict or {}
        self.anchor_records = anchor_records or []
        self.train_substacks = train_substacks or []
        self.patch_size = patch_size
        self.label_size = label_size
        self.offset = self.patch_size // 2 - self.label_size
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        self.unique_labels = np.array(range(n_classes))
        self.dim = dim
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.samples_per_class = samples_per_class or {}
        self.default_samples_per_class = 1
        self.scheduler_path = Path(scheduler_path) if scheduler_path is not None else None
        self.stage_index = int(stage_index)
        self.confidence_threshold = float(confidence_threshold)
        self.sampling_version = 0

        self.stack_names = list(self.images.keys())
        self.name_to_id = {name: i for i, name in enumerate(self.stack_names)}
        self.id_to_name = {i: name for i, name in enumerate(self.stack_names)}
        self.label_source_names = {0: "gt_initial", 1: "pseudo"}
        self.slice_sampling_table = self._build_slice_sampling_table()

        self.schedule = {
            "name_id": np.empty((0,), dtype=np.int32),
            "coords": np.empty((0, 3), dtype=np.int32),
            "current_label": np.empty((0,), dtype=np.int32),
            "gt_label": np.empty((0,), dtype=np.int32),
            "confidence": np.empty((0,), dtype=np.float32),
            "label_source": np.empty((0,), dtype=np.int32),
            "stage_index": np.empty((0,), dtype=np.int32),
            "is_enabled": np.empty((0,), dtype=np.bool_),
            "stage_disabled": np.empty((0,), dtype=np.int32),
            "consecutive_keep_failures": np.empty((0,), dtype=np.int32),
            "last_predicted_label": np.empty((0,), dtype=np.int32),
            "last_confidence": np.empty((0,), dtype=np.float32),
        }

        if self.scheduler_path is not None and self.scheduler_path.exists():
            print(f"Loading scheduler from {self.scheduler_path}...")
            self.load_scheduler_npz(self.scheduler_path)
        else:
            print("Initializing stage-0 labelled scheduler...")
            if self.anchor_records:
                self._load_initial_supervised_records()
            else:
                self._sample_initial_supervised_samples()
            self._print_schedule_report()
            self._bump_sampling_version()

    def _build_slice_sampling_table(self) -> List[Tuple[str, int]]:
        if self.train_substacks:
            table = []
            for substack in self.train_substacks:
                name = substack["stack_name"]
                for z in range(int(substack["z_start"]), int(substack["z_stop"])):
                    table.append((name, int(z)))
            return table
        return [
            (name, int(z))
            for name, z_indices in self.indices_dict.items()
            for z in z_indices
        ]

    def _bump_sampling_version(self):
        self.sampling_version += 1

    def _add_to_schedule(
        self,
        name_id,
        coords,
        current_label,
        gt_label,
        confidence,
        label_source,
        stage_index,
        is_enabled=True,
        stage_disabled=-1,
        consecutive_keep_failures=0,
        last_predicted_label=None,
        last_confidence=None,
    ):
        if last_predicted_label is None:
            last_predicted_label = current_label
        if last_confidence is None:
            last_confidence = confidence
        self.schedule["name_id"] = np.append(self.schedule["name_id"], np.array([name_id], dtype=np.int32))
        self.schedule["coords"] = np.append(self.schedule["coords"], [coords], axis=0)
        self.schedule["current_label"] = np.append(self.schedule["current_label"], np.array([current_label], dtype=np.int32))
        self.schedule["gt_label"] = np.append(self.schedule["gt_label"], np.array([gt_label], dtype=np.int32))
        self.schedule["confidence"] = np.append(self.schedule["confidence"], np.array([confidence], dtype=np.float32))
        self.schedule["label_source"] = np.append(self.schedule["label_source"], np.array([label_source], dtype=np.int32))
        self.schedule["stage_index"] = np.append(self.schedule["stage_index"], np.array([stage_index], dtype=np.int32))
        self.schedule["is_enabled"] = np.append(self.schedule["is_enabled"], np.array([is_enabled], dtype=np.bool_))
        self.schedule["stage_disabled"] = np.append(self.schedule["stage_disabled"], np.array([stage_disabled], dtype=np.int32))
        self.schedule["consecutive_keep_failures"] = np.append(
            self.schedule["consecutive_keep_failures"],
            np.array([consecutive_keep_failures], dtype=np.int32),
        )
        self.schedule["last_predicted_label"] = np.append(
            self.schedule["last_predicted_label"],
            np.array([last_predicted_label], dtype=np.int32),
        )
        self.schedule["last_confidence"] = np.append(
            self.schedule["last_confidence"],
            np.array([last_confidence], dtype=np.float32),
        )

    def _load_initial_supervised_records(self):
        coords_in_use = set()
        sorted_records = sorted(
            self.anchor_records,
            key=lambda record: (
                int(record.get("coord_id", 10**9)),
                str(record["stack_name"]),
                int(record["z"]),
                int(record["y"]),
                int(record["x"]),
            ),
        )
        for record in sorted_records:
            name = record["stack_name"]
            z = int(record["z"])
            y = int(record["y"])
            x = int(record["x"])
            coord_key = (name, z, y, x)
            if coord_key in coords_in_use:
                continue

            Z, H, W = self.images[name].shape
            if not self._is_valid_coord(name, z, y, x, Z, H, W):
                continue

            coords_in_use.add(coord_key)
            gt_label = int(record.get("gt_label", self.labels[name][z, y, x]))
            self._add_to_schedule(
                name_id=self.name_to_id[name],
                coords=(z, y, x),
                current_label=gt_label,
                gt_label=gt_label,
                confidence=1.0,
                label_source=0,
                stage_index=0,
                is_enabled=True,
                stage_disabled=-1,
                consecutive_keep_failures=0,
                last_predicted_label=gt_label,
                last_confidence=1.0,
            )

    def _sample_initial_supervised_samples(self):
        for name, z_indices in self.indices_dict.items():
            img_stack = self.images[name]
            lbl_stack = self.labels[name]
            name_id = self.name_to_id[name]
            Z, H, W = img_stack.shape
            coords_in_use = set(map(tuple, self.schedule["coords"]))
            for cz in tqdm(z_indices, desc=f"Sampling stage-0 labels for {name}"):
                for lbl in self.unique_labels:
                    n_samples = self.samples_per_class.get(int(lbl), self.default_samples_per_class)
                    for cy, cx in self._sample_coordinates_from_slice(lbl_stack[cz], int(lbl), n_samples):
                        if not self._is_valid_coord(name, int(cz), int(cy), int(cx), Z, H, W):
                            continue
                        if (int(cz), int(cy), int(cx)) in coords_in_use:
                            continue
                        coords_in_use.add((int(cz), int(cy), int(cx)))
                        gt_label = int(lbl_stack[cz, cy, cx])
                        self._add_to_schedule(
                            name_id=name_id,
                            coords=(int(cz), int(cy), int(cx)),
                            current_label=gt_label,
                            gt_label=gt_label,
                            confidence=1.0,
                            label_source=0,
                            stage_index=0,
                            is_enabled=True,
                            stage_disabled=-1,
                            consecutive_keep_failures=0,
                            last_predicted_label=gt_label,
                            last_confidence=1.0,
                        )

    def _sample_coordinates_from_slice(self, z_slice: np.ndarray, class_label: int, num_samples: int):
        label_coords = np.argwhere(z_slice == class_label)
        if len(label_coords) < num_samples:
            return []
        idx = self.rng.sample(range(len(label_coords)), num_samples)
        sampled = label_coords[idx]
        return [(int(y), int(x)) for y, x in sampled]

    def _is_valid_coord(self, name, z, y, x, Z, H, W):
        valid = self.offset <= y < H - self.offset - 1 and self.offset <= x < W - self.offset - 1
        if self.dim == 3:
            valid = valid and (self.offset <= z < Z - self.offset - 1)
        in_cell = self.labels[name][z, y, x] != self.ignore_lbl
        return valid and in_cell

    def __len__(self):
        return len(self.get_active_schedule_indices())

    def get_initial_label_mask(self) -> np.ndarray:
        return self.schedule["label_source"] == 0

    def get_active_schedule_indices(self) -> np.ndarray:
        if "is_enabled" not in self.schedule:
            return np.arange(len(self.schedule["name_id"]), dtype=np.int32)
        return np.where(self.schedule["is_enabled"])[0].astype(np.int32)

    def get_active_pseudolabel_indices(self) -> np.ndarray:
        pseudo_mask = self.schedule["label_source"] == 1
        return np.where(self.schedule["is_enabled"] & pseudo_mask)[0].astype(np.int32)

    def count_active_pseudolabels(self) -> int:
        return int(len(self.get_active_pseudolabel_indices()))

    def _scheduled_coordinate_set(self, active_only: bool = True) -> set[tuple[str, int, int, int]]:
        schedule_indices = self.get_active_schedule_indices() if active_only else np.arange(len(self.schedule["name_id"]))
        return {
            (self.id_to_name[int(name_id)], int(z), int(y), int(x))
            for name_id, (z, y, x) in zip(
                self.schedule["name_id"][schedule_indices],
                self.schedule["coords"][schedule_indices],
            )
        }

    def _sample_random_candidate(self, forbidden_coords: set[tuple[str, int, int, int]], max_tries: int = 1024):
        if len(self.slice_sampling_table) == 0:
            return None
        for _ in range(max_tries):
            name, z = self.rng.choice(self.slice_sampling_table)
            valid_positions = np.argwhere(self.labels[name][z] != self.ignore_lbl)
            if len(valid_positions) == 0:
                continue
            y, x = valid_positions[self.rng.randrange(len(valid_positions))]
            y = int(y)
            x = int(x)
            Z, H, W = self.images[name].shape
            if not self._is_valid_coord(name, int(z), y, x, Z, H, W):
                continue
            coord = (name, int(z), y, x)
            if coord in forbidden_coords:
                continue
            return coord
        return None

    def _sample_candidate_batch(self, forbidden_coords: set[tuple[str, int, int, int]], batch_size: int):
        coords_batch = []
        local_forbidden = set(forbidden_coords)
        for _ in range(batch_size):
            candidate = self._sample_random_candidate(local_forbidden)
            if candidate is None:
                break
            coords_batch.append(candidate)
            local_forbidden.add(candidate)
        return coords_batch

    def _weighted_class_targets(self, total_items: int) -> Dict[int, int]:
        class_weights = {
            int(label): float(self.samples_per_class.get(int(label), self.default_samples_per_class))
            for label in self.unique_labels
        }
        total_weight = float(sum(max(weight, 0.0) for weight in class_weights.values()))
        if total_items <= 0:
            return {label: 0 for label in class_weights}
        if total_weight <= 0.0:
            base = total_items // len(class_weights)
            rem = total_items % len(class_weights)
            targets = {label: base for label in class_weights}
            for label in list(class_weights.keys())[:rem]:
                targets[label] += 1
            return targets

        raw_targets = {label: total_items * max(weight, 0.0) / total_weight for label, weight in class_weights.items()}
        targets = {label: int(np.floor(raw_targets[label])) for label in class_weights}
        remainder = total_items - sum(targets.values())
        if remainder > 0:
            ranked_labels = sorted(
                class_weights.keys(),
                key=lambda label: (raw_targets[label] - targets[label], -label),
                reverse=True,
            )
            for label in ranked_labels[:remainder]:
                targets[label] += 1
        return targets

    def build_candidate_batch(self, coords_batch: List[Tuple[str, int, int, int]]) -> dict:
        patches = []
        gt_labels = []
        names = []
        coords = []
        for name, z, y, x in coords_batch:
            patches.append(self.patch_at(self.images[name], z, y, x))
            gt_labels.append(int(self.labels[name][z, y, x]))
            names.append(name)
            coords.append((z, y, x))
        return {
            "patch": torch.stack(patches, dim=0),
            "gt": torch.tensor(gt_labels).long(),
            "coords": torch.tensor(coords).long(),
            "name": names,
        }

    def build_scheduler_batch(self, schedule_indices: List[int]) -> dict:
        patches = []
        gt_labels = []
        names = []
        coords = []
        current_labels = []
        for schedule_idx in schedule_indices:
            name_id = int(self.schedule["name_id"][schedule_idx])
            name = self.id_to_name[name_id]
            z, y, x = map(int, self.schedule["coords"][schedule_idx])
            patches.append(self.patch_at(self.images[name], z, y, x))
            gt_labels.append(int(self.schedule["gt_label"][schedule_idx]))
            names.append(name)
            coords.append((z, y, x))
            current_labels.append(int(self.schedule["current_label"][schedule_idx]))
        return {
            "patch": torch.stack(patches, dim=0),
            "gt": torch.tensor(gt_labels).long(),
            "coords": torch.tensor(coords).long(),
            "name": names,
            "label": torch.tensor(current_labels).long(),
            "schedule_idx": torch.tensor(schedule_indices).long(),
        }

    def add_pseudolabels_for_stage(self, stage_index: int, target_active_pseudolabels: int, evaluator, evaluation_batch_size: int) -> int:
        current_active_pseudolabels = self.count_active_pseudolabels()
        if target_active_pseudolabels <= current_active_pseudolabels:
            return 0

        forbidden_coords = self._scheduled_coordinate_set(active_only=True)
        target_new_pseudolabels = int(target_active_pseudolabels - current_active_pseudolabels)
        class_targets = self._weighted_class_targets(target_new_pseudolabels)
        accepted_per_class = {label: 0 for label in class_targets}
        accepted = 0
        no_progress_rounds = 0
        max_no_progress_rounds = 32
        while self.count_active_pseudolabels() < target_active_pseudolabels and no_progress_rounds < max_no_progress_rounds:
            coords_batch = self._sample_candidate_batch(forbidden_coords, evaluation_batch_size)
            if len(coords_batch) == 0:
                break
            batch = self.build_candidate_batch(coords_batch)
            predicted_labels, confidences = evaluator(batch)
            accepted_this_round = 0
            for (name, z, y, x), pred_label, confidence, gt_label in zip(
                coords_batch,
                predicted_labels,
                confidences,
                batch["gt"].tolist(),
            ):
                pred_label = int(pred_label)
                if float(confidence) < self.confidence_threshold:
                    continue
                self._add_to_schedule(
                    name_id=self.name_to_id[name],
                    coords=(z, y, x),
                    current_label=pred_label,
                    gt_label=int(gt_label),
                    confidence=float(confidence),
                    label_source=1,
                    stage_index=int(stage_index),
                    is_enabled=True,
                    stage_disabled=-1,
                    consecutive_keep_failures=0,
                    last_predicted_label=pred_label,
                    last_confidence=float(confidence),
                )
                forbidden_coords.add((name, z, y, x))
                accepted_per_class[pred_label] = accepted_per_class.get(pred_label, 0) + 1
                accepted += 1
                accepted_this_round += 1
                if self.count_active_pseudolabels() >= target_active_pseudolabels:
                    break
            no_progress_rounds = 0 if accepted_this_round > 0 else no_progress_rounds + 1

        self.stage_index = max(self.stage_index, int(stage_index))
        self._bump_sampling_version()
        self._print_schedule_report()
        return accepted

    def reevaluate_pseudolabels_for_stage(
        self,
        next_stage_idx: int,
        evaluator,
        evaluation_batch_size: int,
        keep_threshold: float,
        pruning_patience: int,
        enable_pruning: bool,
    ) -> int:
        pseudo_indices = self.get_active_pseudolabel_indices().tolist()
        if len(pseudo_indices) == 0:
            return 0

        disabled_count = 0
        for batch_start in range(0, len(pseudo_indices), evaluation_batch_size):
            batch_indices = pseudo_indices[batch_start : batch_start + evaluation_batch_size]
            batch = self.build_scheduler_batch(batch_indices)
            predicted_labels, confidences = evaluator(batch)

            for schedule_idx, pred_label, confidence in zip(batch_indices, predicted_labels, confidences):
                pred_label = int(pred_label)
                confidence = float(confidence)
                self.schedule["last_predicted_label"][schedule_idx] = pred_label
                self.schedule["last_confidence"][schedule_idx] = confidence

                keep_row = pred_label == int(self.schedule["current_label"][schedule_idx]) and confidence >= keep_threshold
                if keep_row:
                    self.schedule["consecutive_keep_failures"][schedule_idx] = 0
                    continue

                self.schedule["consecutive_keep_failures"][schedule_idx] += 1
                if enable_pruning and int(self.schedule["consecutive_keep_failures"][schedule_idx]) >= int(pruning_patience):
                    self.schedule["is_enabled"][schedule_idx] = False
                    self.schedule["stage_disabled"][schedule_idx] = int(next_stage_idx)
                    disabled_count += 1

        if len(pseudo_indices) > 0:
            self._bump_sampling_version()
        return disabled_count

    def save_scheduler_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            name_id=self.schedule["name_id"],
            coords=self.schedule["coords"],
            current_label=self.schedule["current_label"],
            gt_label=self.schedule["gt_label"],
            confidence=self.schedule["confidence"],
            label_source=self.schedule["label_source"],
            stage_index=self.schedule["stage_index"],
            is_enabled=self.schedule["is_enabled"],
            stage_disabled=self.schedule["stage_disabled"],
            consecutive_keep_failures=self.schedule["consecutive_keep_failures"],
            last_predicted_label=self.schedule["last_predicted_label"],
            last_confidence=self.schedule["last_confidence"],
            metadata=np.array([self.seed, self.stage_index, len(self.schedule["name_id"])], dtype=np.int32),
        )

    def load_scheduler_npz(self, path):
        npz = np.load(path)
        for key in self.schedule:
            if key in npz:
                self.schedule[key] = np.array(npz[key])
            else:
                self.schedule[key] = self._default_scheduler_field(key, len(np.array(npz["name_id"])))
        if "metadata" in npz:
            metadata = np.array(npz["metadata"]).astype(np.int32)
            if metadata.size >= 2:
                self.stage_index = int(metadata[1])
        self._bump_sampling_version()

    def _default_scheduler_field(self, key: str, n_rows: int) -> np.ndarray:
        if key == "is_enabled":
            return np.ones((n_rows,), dtype=np.bool_)
        if key == "stage_disabled":
            return np.full((n_rows,), -1, dtype=np.int32)
        if key == "consecutive_keep_failures":
            return np.zeros((n_rows,), dtype=np.int32)
        if key == "last_predicted_label":
            return self.schedule["current_label"].copy()
        if key == "last_confidence":
            return self.schedule["confidence"].copy()
        raise KeyError(f"Unsupported missing scheduler field: {key}")

    def patch_at(self, img_stack, z, y, x):
        if self.dim == 2:
            p = img_stack[z, y - self.offset : y + self.offset + 2, x - self.offset : x + self.offset + 2]
            return torch.from_numpy(p).unsqueeze(0)
        p = img_stack[
            z - self.offset : z + self.offset + 2,
            y - self.offset : y + self.offset + 2,
            x - self.offset : x + self.offset + 2,
        ]
        return torch.from_numpy(p).unsqueeze(0)

    def _print_schedule_report(self):
        if len(self.schedule["name_id"]) == 0:
            print("Empty scheduler.")
            return
        sch = {k: v for k, v in self.schedule.items()}
        sch["z"] = self.schedule["coords"][:, 0]
        sch["y"] = self.schedule["coords"][:, 1]
        sch["x"] = self.schedule["coords"][:, 2]
        del sch["coords"]
        spd = pd.DataFrame(sch)
        spd["name_id"] = spd["name_id"].apply(lambda x: self.id_to_name[x])
        spd["label_source"] = spd["label_source"].apply(lambda x: self.label_source_names.get(int(x), f"source_{x}"))
        spd["is_enabled"] = spd["is_enabled"].astype(bool)
        print(
            spd.groupby(["stage_index", "label_source", "is_enabled", "name_id"])["current_label"]
            .value_counts()
            .unstack(fill_value=0)
        )

    def __getitem__(self, idx):
        schedule_idx = int(self.get_active_schedule_indices()[idx])
        coords = self.schedule["coords"][schedule_idx]
        current_label = self.schedule["current_label"][schedule_idx]
        gt_label = self.schedule["gt_label"][schedule_idx]
        name_id = self.schedule["name_id"][schedule_idx]
        name = self.id_to_name[int(name_id)]
        patch = self.patch_at(self.images[name], coords[0], coords[1], coords[2])
        segmentation = self.patch_at(self.labels[name], coords[0], coords[1], coords[2])
        return {
            "name": name,
            "patch": patch,
            "label": torch.tensor(current_label).long(),
            "gt": torch.tensor(gt_label).long(),
            "coords": torch.tensor(coords).long(),
            "schedule_idx": torch.tensor(int(schedule_idx)).long(),
            "is_initial_label": torch.tensor(bool(self.schedule["label_source"][schedule_idx] == 0), dtype=torch.bool),
            "stage_index": torch.tensor(int(self.schedule["stage_index"][schedule_idx])).long(),
            "segmentation": segmentation,
            "confidence": torch.tensor(float(self.schedule["confidence"][schedule_idx])).float(),
        }


class SemisupervisedDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=1,
        mode="semisupervised",
        n_classes=4,
        ignore_lbl=-1,
        indices_dict=None,
        radius=5,
        dim=2,
        seed=42,
        n_neighbors=7,
        samples_per_class: Dict[int, int] | None = None,
        anchor_records: Optional[List[Dict[str, int]]] = None,
    ):
        """
        
        Args:
            images: dict of np.ndarrays
                Key: image name,
                Value: Image volume of shape (C,Z,H,W) or (Z,H,W)
            labels: dict of np.ndarrays
                Key: image name,
                Value: Label volume of shape (Z,H,W) or (1,Z,H,W)
            patch_size: int (Default: 64)
                Spatial size of the extracted patches
            label_size: int (Default: 1)
                Spatial size of the label patch (must be <= patch_size//2)
            mode: str (Default: "semisupervised")
                Whether to return only anchors ("supervised") or anchors + neighbors ("semisupervised")
            n_classes: int (Default: 4)
                Number of semantic classes in the labels (including background)
            ignore_lbl: int (Default: -1)
                Label value to ignore when sampling anchors and neighbors
            indices_dict: dict (Default: None)
                Key: image name,
                Value: list of z indices to sample from for that image
            radius: int (Default: 5)
                Maximum distance in pixels to sample neighbors from the anchor
            dim: int (Default: 2)
                Whether to sample 2D or 3D patches (must be 2 or 3)
            seed: int (Default: 42)
                Random seed for reproducible neighbor sampling         n_neighbors: int (Default: 7)
                Number of neighbors to sample per anchor in semisupervised mode
            samples_per_class: dict (Default: None)
                Optional per-class override for number of anchors to sample from each class.
                Key: class label (int), Value: number of anchors to sample (int).
                 If not provided, defaults to 1 anchor per class.
        
        """
        self.patch_size = patch_size
        self.label_size = label_size
        self.offset = self.patch_size // 2 - self.label_size
        self.images = images
        self.labels = labels
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        self.unique_labels = np.array(range(n_classes))
        self.unique_vals = self.unique_labels
        self.mode = mode
        self.indices_dict = indices_dict or {}
        self.anchor_records = anchor_records or []
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.samples_per_class = samples_per_class or {}
        self.default_samples_per_class = 1
        self.dim = dim
        self.groups = self._prepare_metadata()
        self._refresh_group_index_cache()
        self._report_dataset_summary()

    def set_mode(self, mode: str):
        if mode not in ("supervised", "semisupervised"):
            raise ValueError("stage must be 'supervised' or 'semisupervised'")
        self.mode = mode
        self.groups = self._prepare_metadata()
        self._refresh_group_index_cache()

    def set_radius(self, radius: int):
        if self.radius != radius:
            self.radius = radius
            self.groups = self._modify_metadata()
            self._refresh_group_index_cache()

    def increase_radius(self):
        self.radius += 1
        self.groups = self._modify_metadata()
        self._refresh_group_index_cache()

    def _is_valid_coord(
        self,
        name,
        z,
        y,
        x,
        Z,
        H,
        W,
        z_start: Optional[int] = None,
        z_stop: Optional[int] = None,
    ):
        valid = self.offset <= y < H - self.offset - 1 and self.offset <= x < W - self.offset - 1
        if self.dim == 3:
            valid = valid and (self.offset <= z < Z - self.offset - 1)
        if z_start is not None and z_stop is not None:
            valid = valid and (z_start <= z < z_stop)
        in_cell = self.labels[name][z, y, x] != self.ignore_lbl
        return valid and in_cell

    def __len__(self):
        return len(self.groups)

    def patch_at(self, img_stack, z, y, x):
        if self.dim == 2:
            p = img_stack[z, y - self.offset : y + self.offset + 2, x - self.offset : x + self.offset + 2]
            return torch.from_numpy(p).unsqueeze(0)
        p = img_stack[
            z - self.offset : z + self.offset + 2,
            y - self.offset : y + self.offset + 2,
            x - self.offset : x + self.offset + 2,
        ]
        return torch.from_numpy(p).unsqueeze(0)

    def __getitem__(self, idx):
        g = self.groups[idx]
        name = g["name"]
        img_vol = self.images[name]
        lbl_vol = self.labels[name]

        if self.mode == "supervised":
            cz, cy, cx = map(int, g["coords"][0])
            patch = self.patch_at(img_vol, cz, cy, cx).unsqueeze(0)
            label = torch.tensor([int(g["labels"][0])], dtype=torch.long)
            segment = self.patch_at(lbl_vol, cz, cy, cx).unsqueeze(0)
            return patch, label, segment, torch.tensor(g["coords"][0])

        coords = torch.tensor([tuple(map(int, xyz)) for xyz in g["coords"]])
        patches = torch.stack([self.patch_at(img_vol, cz, cy, cx) for (cz, cy, cx) in coords])
        labels = torch.tensor([g["labels"][0]] + [-1] * self.n_neighbors, dtype=torch.long)
        segments = torch.stack([self.patch_at(lbl_vol, cz, cy, cx) for (cz, cy, cx) in coords])
        return patches, labels, segments, coords

    def _prepare_metadata(self) -> List[dict]:
        if self.anchor_records:
            return self._prepare_metadata_from_anchors()
        return self._prepare_metadata_from_indices()

    def _prepare_metadata_from_indices(self) -> List[dict]:
        groups: List[dict] = []
        for name, z_list in tqdm(self.indices_dict.items(), desc="Preparing supervised anchors from slice indices", leave=False):
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape
            used_coords = set()
            for cz in z_list:
                stack = lbl[cz]
                for c in range(self.n_classes):
                    for cy, cx in self._sample_coords_for_class(stack, c):
                        if not self._is_valid_coord(name, cz, cy, cx, Z, H, W):
                            continue
                        if (cz, cy, cx) in used_coords:
                            continue
                        used_coords.add((int(cz), cy, cx))
                        neighbors = self._sample_neighbors(
                            name=name,
                            cz=cz,
                            cy=cy,
                            cx=cx,
                            Z=Z,
                            H=H,
                            W=W,
                            used_coords=used_coords,
                            lbl=lbl,
                            k=self.n_neighbors,
                            max_tries=100,
                        )
                        if self.mode == "supervised" or len(neighbors) == self.n_neighbors:
                            groups.append(self._make_group_record(name=name, cz=cz, cy=cy, cx=cx, c=c, neighbors=neighbors))
        self._report_class_counts(groups)
        return groups

    def _prepare_metadata_from_anchors(self) -> List[dict]:
        groups: List[dict] = []
        for record in tqdm(self.anchor_records, desc="Preparing cached anchor groups", leave=False):
            name = record["stack_name"]
            cz = int(record["z"])
            cy = int(record["y"])
            cx = int(record["x"])
            c = int(record["gt_label"])
            z_start = int(record["z_start"])
            z_stop = int(record["z_stop"])

            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape
            if not self._is_valid_coord(name, cz, cy, cx, Z, H, W, z_start=z_start, z_stop=z_stop):
                continue

            used_coords = {(cz, cy, cx)}
            neighbors: List[Dict[str, int]] = []
            if self.mode == "semisupervised":
                neighbors = self._sample_neighbors(
                    name=name,
                    cz=cz,
                    cy=cy,
                    cx=cx,
                    Z=Z,
                    H=H,
                    W=W,
                    used_coords=used_coords,
                    lbl=lbl,
                    z_start=z_start,
                    z_stop=z_stop,
                    k=self.n_neighbors,
                    max_tries=100,
                )
                if len(neighbors) != self.n_neighbors:
                    continue

            groups.append(
                self._make_group_record(
                    name=name,
                    cz=cz,
                    cy=cy,
                    cx=cx,
                    c=c,
                    neighbors=neighbors,
                    substack_id=int(record["substack_id"]),
                    z_start=z_start,
                    z_stop=z_stop,
                    coord_id=int(record["coord_id"]),
                )
            )
        self._report_class_counts(groups)
        return groups

    def _modify_metadata(self) -> List[dict]:
        if self.anchor_records:
            return self._modify_cached_anchor_metadata()

        for g in self.groups:
            name = g["name"]
            z = int(g["z"])
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape
            used_coords = set()
            _, cy, cx = g["coords"][0]
            used_coords.add((int(z), cy, cx))
            neighbors = self._sample_neighbors(
                name=name,
                cz=z,
                cy=cy,
                cx=cx,
                Z=Z,
                H=H,
                W=W,
                used_coords=used_coords,
                lbl=lbl,
                k=self.n_neighbors,
                max_tries=100,
            )
            if len(neighbors) == self.n_neighbors:
                modified_group = self._make_group_record(name=name, cz=z, cy=cy, cx=cx, c=g["labels"][0], neighbors=neighbors)
                g["coords"] = modified_group["coords"]
                g["labels"] = modified_group["labels"]

        self._report_class_counts(self.groups)
        return self.groups

    def _modify_cached_anchor_metadata(self) -> List[dict]:
        for g in self.groups:
            name = g["name"]
            cz, cy, cx = map(int, g["coords"][0])
            z_start = g.get("z_start")
            z_stop = g.get("z_stop")
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape

            if self.mode == "supervised":
                g["coords"] = [(cz, cy, cx)]
                g["labels"] = [int(g["labels"][0])]
                continue

            current_neighbor_coords = [tuple(map(int, coord)) for coord in g["coords"][1:]]
            current_neighbor_labels = [int(label) for label in g["labels"][1:]]
            used_coords = {(cz, cy, cx), *current_neighbor_coords}
            neighbors = self._sample_neighbors(
                name=name,
                cz=cz,
                cy=cy,
                cx=cx,
                Z=Z,
                H=H,
                W=W,
                used_coords=used_coords,
                lbl=lbl,
                z_start=z_start,
                z_stop=z_stop,
                k=self.n_neighbors,
                max_tries=100,
            )

            if len(neighbors) == self.n_neighbors:
                modified_group = self._make_group_record(
                    name=name,
                    cz=cz,
                    cy=cy,
                    cx=cx,
                    c=int(g["labels"][0]),
                    neighbors=neighbors,
                    substack_id=g.get("substack_id"),
                    z_start=z_start,
                    z_stop=z_stop,
                    coord_id=g.get("coord_id"),
                )
                g["coords"] = modified_group["coords"]
                g["labels"] = modified_group["labels"]
                continue
            if len(neighbors) > 0:
                new_neighbors = [
                    (int(neighbor["z"]), int(neighbor["y"]), int(neighbor["x"]), int(neighbor["label"]))
                    for neighbor in neighbors
                ]
                n_replace = min(len(new_neighbors), len(current_neighbor_coords))
                kept_coords = current_neighbor_coords[n_replace:]
                kept_labels = current_neighbor_labels[n_replace:]
                updated_coords = kept_coords + [(z, y, x) for z, y, x, _ in new_neighbors[:n_replace]]
                updated_labels = kept_labels + [label for _, _, _, label in new_neighbors[:n_replace]]
                if len(updated_coords) == self.n_neighbors:
                    g["coords"] = [(cz, cy, cx)] + updated_coords
                    g["labels"] = [int(g["labels"][0])] + updated_labels

        self._report_class_counts(self.groups)
        return self.groups

    def _sample_coords_for_class(self, stack: np.ndarray, c: int) -> Iterable[Tuple[int, int]]:
        n_needed = self.samples_per_class.get(c, self.default_samples_per_class)
        label_coords = np.argwhere(stack == c)
        if len(label_coords) == 0 or n_needed <= 0:
            return []
        n_take = min(len(label_coords), n_needed)
        idx = self.rng.sample(range(len(label_coords)), n_take)
        sampled = label_coords[idx]
        return [(int(y), int(x)) for (y, x) in sampled]

    def _sample_neighbors(
        self,
        name: str,
        cz: int,
        cy: int,
        cx: int,
        Z: int,
        H: int,
        W: int,
        used_coords: set,
        lbl: np.ndarray,
        z_start: Optional[int] = None,
        z_stop: Optional[int] = None,
        k: int = 3,
        max_tries: int = 100,
    ) -> List[Dict[str, int]]:
        neighbors: List[Dict[str, int]] = []
        tries = 0
        while len(neighbors) < k and tries < max_tries:
            dz = self.rng.randint(-self.radius, self.radius) if self.dim == 3 else 0
            dy = self.rng.randint(-self.radius, self.radius)
            dx = self.rng.randint(-self.radius, self.radius)
            if dx * dx + dy * dy + dz * dz > self.radius * self.radius:
                tries += 1
                continue
            if dx == 0 and dy == 0 and dz == 0:
                tries += 1
                continue
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            coord = (nz, ny, nx)
            if coord in used_coords:
                tries += 1
                continue
            if self._is_valid_coord(name, nz, ny, nx, Z, H, W, z_start=z_start, z_stop=z_stop):
                used_coords.add(coord)
                neighbors.append(
                    {
                        "z": int(nz),
                        "y": int(ny),
                        "x": int(nx),
                        "label": int(lbl[nz, ny, nx].item()),
                    }
                )
            tries += 1
        return neighbors

    def _make_group_record(
        self,
        name: str,
        cz: int,
        cy: int,
        cx: int,
        c: int,
        neighbors: List[Dict[str, int]],
        substack_id: Optional[int] = None,
        z_start: Optional[int] = None,
        z_stop: Optional[int] = None,
        coord_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "z": int(cz),
            "coords": [(int(cz), int(cy), int(cx))] + [(n["z"], n["y"], n["x"]) for n in neighbors],
            "labels": [int(c)] + [n["label"] for n in neighbors],
            "substack_id": substack_id,
            "z_start": z_start,
            "z_stop": z_stop,
            "coord_id": coord_id,
        }

    def _report_class_counts(self, groups: List[dict]) -> None:
        centers = [g["labels"][0] for g in groups]
        neighbors = [lab for g in groups for lab in g["labels"][1:]]
        print(self._format_class_balance_table(centers, neighbors))

    def _refresh_group_index_cache(self) -> None:
        self.n_label_per_class = {c: len([g for g in self.groups if g["labels"][0] == c]) for c in range(self.n_classes)}
        self.anchor_indices_by_label = {c: [i for i, g in enumerate(self.groups) if g["labels"][0] == c] for c in range(self.n_classes)}

    def _report_dataset_summary(self) -> None:
        anchor_mode = "cached anchors" if self.anchor_records else "sampled indices"
        print(
            f"Dataset ready | mode={self.mode} | dim={self.dim}D | groups={len(self.groups)} "
            f"| neighbors={self.n_neighbors} | source={anchor_mode}"
        )

    def _format_class_balance_table(self, centers: List[int], neighbors: List[int]) -> str:
        anchor_counts = Counter(centers)
        neighbor_counts = Counter(neighbors)
        all_labels = sorted(set(anchor_counts) | set(neighbor_counts) | set(range(self.n_classes)))
        anchor_total = sum(anchor_counts.values())
        neighbor_total = sum(neighbor_counts.values())
        lines = ["Class balance:", f"{'label':>7} {'anchors':>10} {'anchor_%':>10} {'neighbors':>10} {'neighbor_%':>10}"]
        for label in all_labels:
            anchor_count = anchor_counts.get(label, 0)
            neighbor_count = neighbor_counts.get(label, 0)
            anchor_pct = 100.0 * anchor_count / anchor_total if anchor_total else 0.0
            neighbor_pct = 100.0 * neighbor_count / neighbor_total if neighbor_total else 0.0
            lines.append(
                f"{label:>7} {anchor_count:>10} {anchor_pct:>9.2f}% {neighbor_count:>10} {neighbor_pct:>9.2f}%"
            )
        lines.append(
            f"{'total':>7} {anchor_total:>10} {100.0 if anchor_total else 0.0:>9.2f}% "
            f"{neighbor_total:>10} {100.0 if neighbor_total else 0.0:>9.2f}%"
        )
        return "\n".join(lines)


class PredictionDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        keys: List[str] = None,
        masks: Dict[str, Union[slice, None]] = None,
        patch_size=64,
        dim=2,
        ignore_lbl=-1,
    ):
        self.dim = dim
        self.ignore_lbl = ignore_lbl
        self.images = images
        self.labels = labels
        self.masks = masks

        self.ps = int(patch_size)
        assert self.ps % 2 == 0, "Patch size must be even; center is (ps/2-1, ps/2-1)."
        self.half = self.ps // 2

        self.images = self._fix_images_shape(self.images)
        self.labels = self._fix_images_shape(self.labels)
        self.masks = self._fix_mask_shape(self.masks)
        self.centers = []

        for k in (self.images.keys() if keys is None else keys):
            centers = self._get_valid_centers(
                mask=self.masks.get(k, None) if self.masks is not None else None,
                label=self.labels[k],
            )
            for c in centers:
                self.centers.append((k, c[0], c[1], c[2]))

    def _get_valid_centers(self, mask, label):
        final_mask = label != self.ignore_lbl
        C, Z, H, W = label.shape
        final_mask[:, :, :, : self.half] = False
        final_mask[:, :, :, W - self.half :] = False
        final_mask[:, :, : self.half, :] = False
        final_mask[:, :, H - self.half :, :] = False
        if self.dim == 3 and Z > self.half:
            final_mask[:, : self.half, :, :] = False
            final_mask[:, Z - self.half :, :, :] = False
        if mask is not None:
            mask_array = np.zeros_like(final_mask, dtype=bool)
            mask_array[mask] = True
            final_mask = final_mask & mask_array
        _, zs, ys, xs = np.where(final_mask)
        centers = np.stack([zs, ys, xs], axis=1).astype(np.int32)
        return centers

    def _fix_images_shape(self, images: dict) -> dict:
        for key, image in images.items():
            if image.ndim == 3:
                images[key] = image[None, ...]
        return images

    def _fix_mask_shape(self, masks: dict) -> dict:
        if masks is not None:
            for key, mask in masks.items():
                if mask is not None and len(mask) == 3:
                    masks[key] = (slice(None),) + mask
        return masks

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        key, z, y, x = self.centers[idx]
        if self.dim == 3:
            z0, z1 = z - self.half, z + self.half
        else:
            z0, z1 = z, z + 1
        y0, y1 = y - self.half, y + self.half
        x0, x1 = x - self.half, x + self.half
        patch = self.images[key][:, z0:z1, y0:y1, x0:x1]
        patch = torch.from_numpy(patch).float()
        segment = self.labels[key][:, z0:z1, y0:y1, x0:x1]
        segment = torch.from_numpy(segment).long()
        center_label = torch.tensor(self.labels[key][:, z, y, x]).long()
        if self.dim == 2:
            patch = patch.squeeze(-3)
            segment = segment.squeeze(-3)
        coords = torch.stack([torch.tensor(z), torch.tensor(y), torch.tensor(x)])
        return patch, center_label, segment, coords, key
