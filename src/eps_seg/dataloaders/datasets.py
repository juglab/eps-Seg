import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eps_seg.config.train import TrainConfig
from eps_seg.data.factory import (
    build_candidate_sampling_strategy,
    build_schedule_admission_policy,
    build_pseudolabel_maintenance_policy,
)
from eps_seg.data.initial_label_sampling import (
    InitialLabelSamplingContext,
    populate_schedule_from_coordinate_records,
)
from eps_seg.data.sampling import SamplingDomain
from eps_seg.data.schedule import DataSchedule
from eps_seg.data.schedule_engine import ScheduleEngine


class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        patch_size: int = 64,
        label_size: int = 1,
        n_classes: int = 4,
        ignore_lbl: int = -1,
        dim: int = 2,
        seed: int = 42,
        samples_per_class: Optional[Dict[int, int]] = None,
        scheduler_path: Optional[Union[str, Path]] = None,
        stage_index: int = 0,
        coordinate_records: Optional[List[Dict[str, int]]] = None,
        train_substacks: Optional[List[Dict[str, int]]] = None,
        train_cfg: Optional[TrainConfig] = None,
    ) -> None:
        """
        Dataset storing the full staged scheduler used by the refactored
        training loop.

        Stage-0 rows are initialized from canonical cached coordinate records when
        no scheduler file is provided. Initial GT sampling happens during cache
        creation, not during dataset construction.

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
                Spatial label patch size (must be <= patch_size//2)
            n_classes: int (Default: 4)
                Number of semantic classes in the labels (including background)
            ignore_lbl: int (Default: -1)
                Label value to ignore when sampling candidates
            dim: int (Default: 2)
                Whether to sample 2D or 3D patches (must be 2 or 3)
            seed: int (Default: 42)
                Random seed for reproducible candidate sampling and schedule mutations
            samples_per_class: dict (Default: None)
                Optional per-class override for number of candidates to sample from each class during stage-0 initialization.
                Key: class label (int), Value: number of candidates to sample (int).
                If not provided, defaults to 1 candidate per class.
            scheduler_path: str or Path (Default: None)
                Optional path to a ``.npz`` scheduler snapshot. If provided and the file exists, the scheduler will be loaded from disk instead of being initialized from the coordinate records.
            stage_index: int (Default: 0)
                Initial stage index for the dataset scheduler. This does not affect the loaded scheduler state if a scheduler snapshot is provided, 
                but it does affect the random seed used for runtime candidate sampling and schedule mutations, which is computed as ``seed + stage_index``.
            coordinate_records: list of dicts (Default: None)
                Optional list of coordinate records for initializing the dataset.
                Each record should be a dict with keys: "stack_name", "z", "y", "x", "gt_label", "substack_id", "z_start", "z_stop", and "coord_id".
                If not provided, stage-0 initialization will be skipped (resulting in an empty scheduler) unless a scheduler snapshot is loaded from disk.
            train_substacks: list of dicts (Default: None)
                Optional list of train substacks for initializing the dataset.
                Each record should be a dict with keys: "stack_name", "z_start", and "z_stop".
                If not provided, stage-0 initialization will be skipped (resulting in an empty scheduler) unless a scheduler snapshot is loaded from disk.
            train_cfg: TrainConfig (Default: None)
                Optional training configuration for the dataset.
        """
        self.images = images
        self.labels = labels
        self.coordinate_records = coordinate_records or []
        self.train_substacks = train_substacks or []
        self.patch_size = patch_size
        self.label_size = label_size
        self.offset = self.patch_size // 2 - self.label_size
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        self.unique_labels = np.array(range(n_classes))
        self.dim = dim
        self.seed = seed
        self._runtime_stage_index = int(stage_index)
        self.rng = random.Random(self._runtime_rng_seed())
        self.samples_per_class = samples_per_class or {}
        self.default_samples_per_class = 1
        self.scheduler_path = Path(scheduler_path) if scheduler_path is not None else None
        self.stage_index = int(stage_index)
        self.sampling_version = 0
        self.train_cfg = train_cfg or TrainConfig()

        self.stack_names = list(self.images.keys())
        self.name_to_id = {name: i for i, name in enumerate(self.stack_names)}
        self.id_to_name = {i: name for i, name in enumerate(self.stack_names)}
        self.label_source_names = {0: "gt_initial", 1: "pseudo", 2: "gt_acquired"}
        self._schedule = DataSchedule.empty(seed=self.seed, stage_index=self.stage_index)
        self.initial_label_sampling_context = InitialLabelSamplingContext(
            images=self.images,
            labels=self.labels,
            coordinate_records=self.coordinate_records,
            train_substacks=self.train_substacks,
            samples_per_class=self.samples_per_class,
            unique_labels=self.unique_labels,
            default_samples_per_class=self.default_samples_per_class,
            ignore_lbl=self.ignore_lbl,
            patch_size=self.patch_size,
            label_size=self.label_size,
            dim=self.dim,
            max_patch_size=None,
            name_to_id=self.name_to_id,
        )
        self.sampling_domain = SamplingDomain(
            images=self.images,
            labels=self.labels,
            train_substacks=self.train_substacks,
            ignore_lbl=self.ignore_lbl,
            patch_size=self.patch_size,
            label_size=self.label_size,
            dim=self.dim,
        )
        self._build_components()

        if self.scheduler_path is not None and self.scheduler_path.exists():
            print(f"Loading scheduler from {self.scheduler_path}...")
            self._load_scheduler_npz(self.scheduler_path)
        else:
            print("Initializing stage-0 labelled scheduler...")
            if self.coordinate_records:
                populate_schedule_from_coordinate_records(
                    schedule=self.schedule,
                    context=self.initial_label_sampling_context,
                )
            self._print_schedule_report()
            self.schedule.bump_version()
            self.sampling_version = self.schedule.sampling_version

    @property
    def schedule(self) -> DataSchedule:
        """
        Return the scheduler state object backing this dataset.

        Args:
            None

        Returns:
            DataSchedule: Mutable scheduler state used by the staged pipeline.
        """

        return self._schedule

    def _build_components(self) -> None:
        """
        Build the sampling strategy, policies, and schedule engine used by the dataset.

        Args:
            None

        Returns:
            None
        """
        # Runtime candidate sampling should depend on the target stage being
        # constructed, not on the stage metadata loaded from a previous
        # scheduler snapshot.
        self.rng = random.Random(self._runtime_rng_seed())
        self.candidate_sampling_strategy = build_candidate_sampling_strategy(
            cfg=self.train_cfg.candidate_sampling,
            domain=self.sampling_domain,
            rng=self.rng,
            samples_per_class=self.train_cfg.extension_samples_per_class or self.samples_per_class,
        )
        self.schedule_admission_policy = build_schedule_admission_policy(self.train_cfg.schedule_admission)
        self.pseudolabel_maintenance_policy = build_pseudolabel_maintenance_policy(
            cfg=self.train_cfg.pseudolabel_maintenance,
            admission_policy=self.schedule_admission_policy,
        )

        self.schedule_engine = ScheduleEngine(
            dataset=self,
            schedule=self.schedule,
            sampler=self.candidate_sampling_strategy,
            admission_policy=self.schedule_admission_policy,
            maintenance_policy=self.pseudolabel_maintenance_policy,
        )
        self.slice_sampling_table = getattr(self.candidate_sampling_strategy, "slice_sampling_table", [])

    def _bump_sampling_version(self) -> None:
        """
        Increase the dataset scheduler version after a schedule update.
        """
        self.sampling_version += 1
        self.schedule.sampling_version = self.sampling_version

    def _runtime_rng_seed(self) -> int:
        return int(self.seed) + int(self._runtime_stage_index)

    def _is_valid_coord(self, name: str, z: int, y: int, x: int, Z: int, H: int, W: int) -> bool:
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
        return self.schedule.get_active_indices()

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

    def _load_scheduler_npz(self, path: Union[str, Path]) -> None:
        """Load a scheduler from disk without changing its semantics.

        Args:
            path: Input path for the ``.npz`` scheduler file.

        Returns:
            None
        """

        self._schedule = DataSchedule.load_npz(Path(path))
        self.stage_index = int(self.schedule.stage_index)
        self._build_components()
        self.schedule.bump_version()
        self.sampling_version = self.schedule.sampling_version

    def patch_at(self, img_stack: np.ndarray, z: int, y: int, x: int) -> torch.Tensor:
        if self.dim == 2:
            p = img_stack[z, y - self.offset : y + self.offset + 2, x - self.offset : x + self.offset + 2]
            return torch.from_numpy(p).unsqueeze(0)
        p = img_stack[
            z - self.offset : z + self.offset + 2,
            y - self.offset : y + self.offset + 2,
            x - self.offset : x + self.offset + 2,
        ]
        return torch.from_numpy(p).unsqueeze(0)

    def _print_schedule_report(self) -> None:
        """
        Print a compact schedule summary grouped by stage, source, and stack.

        Args:
            None

        Returns:
            None
        """

        self.schedule.print_report(id_to_name=self.id_to_name, label_source_names=self.label_source_names)

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
        coordinate_records: Optional[List[Dict[str, int]]] = None,
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
                Whether to return only centers ("supervised") or centers + neighbors ("semisupervised")
            n_classes: int (Default: 4)
                Number of semantic classes in the labels (including background)
            ignore_lbl: int (Default: -1)
                Label value to ignore when sampling centers and neighbors
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
                Optional per-class override for number of centers to sample from each class.
                Key: class label (int), Value: number of centers to sample (int).
                 If not provided, defaults to 1 center per class.
        
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
        self.coordinate_records = coordinate_records or []
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
        if self.coordinate_records:
            return self._prepare_metadata_from_coordinate_records()
        return self._prepare_metadata_from_indices()

    def _prepare_metadata_from_indices(self) -> List[dict]:
        groups: List[dict] = []
        for name, z_list in tqdm(
            self.indices_dict.items(),
            desc="Preparing supervised centers from slice indices",
            leave=False,
        ):
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

    def _prepare_metadata_from_coordinate_records(self) -> List[dict]:
        groups: List[dict] = []
        for record in tqdm(self.coordinate_records, desc="Preparing cached coordinate groups", leave=False):
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
        if self.coordinate_records:
            return self._modify_cached_coordinate_metadata()

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

    def _modify_cached_coordinate_metadata(self) -> List[dict]:
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
        coordinate_mode = "cached coordinates" if self.coordinate_records else "sampled indices"
        print(
            f"Dataset ready | mode={self.mode} | dim={self.dim}D | groups={len(self.groups)} "
            f"| neighbors={self.n_neighbors} | source={coordinate_mode}"
        )

    def _format_class_balance_table(self, centers: List[int], neighbors: List[int]) -> str:
        center_counts = Counter(centers)
        neighbor_counts = Counter(neighbors)
        all_labels = sorted(set(center_counts) | set(neighbor_counts) | set(range(self.n_classes)))
        center_total = sum(center_counts.values())
        neighbor_total = sum(neighbor_counts.values())
        lines = ["Class balance:", f"{'label':>7} {'centers':>10} {'center_%':>10} {'neighbors':>10} {'neighbor_%':>10}"]
        for label in all_labels:
            center_count = center_counts.get(label, 0)
            neighbor_count = neighbor_counts.get(label, 0)
            center_pct = 100.0 * center_count / center_total if center_total else 0.0
            neighbor_pct = 100.0 * neighbor_count / neighbor_total if neighbor_total else 0.0
            lines.append(
                f"{label:>7} {center_count:>10} {center_pct:>9.2f}% {neighbor_count:>10} {neighbor_pct:>9.2f}%"
            )
        lines.append(
            f"{'total':>7} {center_total:>10} {100.0 if center_total else 0.0:>9.2f}% "
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
