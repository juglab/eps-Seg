from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Protocol

import numpy as np

from eps_seg.data.schedule import DataSchedule


@dataclass(frozen=True)
class InitialLabelSamplingContext:
    """Immutable context used to build the stage-0 labelled scheduler.

    Args:
        images: Mapping from stack name to image volume.
        labels: Mapping from stack name to label volume.
        anchor_records: Optional canonical labelled voxels loaded from cache or CSV.
        train_substacks: Train-region substack descriptors.
        samples_per_class: Requested number of stage-0 labels per class and slice or substack, depending on the strategy.
        unique_labels: Semantic labels available in the dataset.
        default_samples_per_class: Fallback count for labels not explicitly configured.
        ignore_lbl: Label value to ignore during sampling.
        patch_size: Spatial patch size used by the dataset.
        label_size: Spatial label patch size used by the dataset.
        dim: Patch dimensionality.
        name_to_id: Mapping from stack names to integer identifiers.

    Returns:
        InitialLabelSamplingContext: Immutable description of the stage-0 sampling domain.
    """

    images: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    anchor_records: List[Dict[str, int]]
    train_substacks: List[Dict[str, int]]
    samples_per_class: Dict[int, int]
    unique_labels: np.ndarray
    default_samples_per_class: int
    ignore_lbl: int
    patch_size: int
    label_size: int
    dim: int
    name_to_id: Mapping[str, int]

    @property
    def offset(self) -> int:
        """Return the spatial patch offset used to validate voxel centers.

        Args:
            None

        Returns:
            int: Patch offset around the candidate center.
        """

        return self.patch_size // 2 - self.label_size


class InitialLabelSamplingStrategy(Protocol):
    """Interface for constructing the initial labelled scheduler at stage K0.

    Args:
        None

    Returns:
        InitialLabelSamplingStrategy: Object able to populate the stage-0 schedule.
    """

    def populate_schedule(
        self,
        schedule: DataSchedule,
        context: InitialLabelSamplingContext,
        rng: random.Random,
    ) -> None:
        """Populate the stage-0 schedule with GT-labelled rows.

        Args:
            schedule: Schedule to mutate.
            context: Immutable stage-0 sampling context.
            rng: Random generator used by sampling-based strategies.

        Returns:
            None
        """


def _is_valid_coord(context: InitialLabelSamplingContext, name: str, z: int, y: int, x: int) -> bool:
    """Check whether a voxel center is valid for stage-0 schedule construction.

    Args:
        context: Immutable stage-0 sampling context.
        name: Stack name.
        z: Z coordinate.
        y: Y coordinate.
        x: X coordinate.

    Returns:
        bool: Whether the coordinate is valid for the current patch geometry.
    """

    z_size, height, width = context.images[name].shape
    offset = context.offset
    valid = offset <= y < height - offset - 1 and offset <= x < width - offset - 1
    if context.dim == 3:
        valid = valid and (offset <= z < z_size - offset - 1)
    return bool(valid and context.labels[name][z, y, x] != context.ignore_lbl)


def _selected_slices_from_context(context: InitialLabelSamplingContext) -> dict[str, list[int]]:
    """Return the selected z-slices for slice-based initial-label sampling.

    Args:
        context: Immutable stage-0 sampling context.

    Returns:
        dict[str, list[int]]: Mapping from stack name to selected z-indices.
    """

    selected: dict[str, list[int]] = {}
    for substack in context.train_substacks:
        name = substack["stack_name"]
        selected.setdefault(name, []).extend(range(int(substack["z_start"]), int(substack["z_stop"])))
    return selected


class FromCSVInitialLabelSamplingStrategy:
    """Initialize stage-0 labels from canonical labelled-voxel records.

    Args:
        None

    Returns:
        FromCSVInitialLabelSamplingStrategy: Strategy loading the initial schedule from canonical records.
    """

    def populate_schedule(
        self,
        schedule: DataSchedule,
        context: InitialLabelSamplingContext,
        rng: random.Random,
    ) -> None:
        """Populate the schedule from canonical anchor records.

        Args:
            schedule: Schedule to mutate.
            context: Immutable stage-0 sampling context.
            rng: Unused random generator kept for interface consistency.

        Returns:
            None
        """

        del rng
        coords_in_use: set[tuple[str, int, int, int]] = set()
        sorted_records = sorted(
            context.anchor_records,
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
            if coord_key in coords_in_use or not _is_valid_coord(context=context, name=name, z=z, y=y, x=x):
                continue

            coords_in_use.add(coord_key)
            gt_label = int(record.get("gt_label", context.labels[name][z, y, x]))
            schedule.add_record(
                name_id=context.name_to_id[name],
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


class ClassBalancedSliceInitialLabelSamplingStrategy:
    """Initialize stage-0 labels by class-balanced sampling inside selected slices.

    Args:
        None

    Returns:
        ClassBalancedSliceInitialLabelSamplingStrategy: Strategy reproducing the current legacy per-slice stage-0 sampling behavior.
    """

    def populate_schedule(
        self,
        schedule: DataSchedule,
        context: InitialLabelSamplingContext,
        rng: random.Random,
    ) -> None:
        """Populate the schedule with class-balanced GT voxels sampled from each selected slice.

        Args:
            schedule: Schedule to mutate.
            context: Immutable stage-0 sampling context.
            rng: Random generator used for class-balanced slice sampling.

        Returns:
            None
        """

        for name, z_indices in _selected_slices_from_context(context).items():
            lbl_stack = context.labels[name]
            name_id = context.name_to_id[name]
            coords_in_use = set(map(tuple, schedule["coords"]))
            for cz in z_indices:
                for lbl in context.unique_labels:
                    n_samples = context.samples_per_class.get(int(lbl), context.default_samples_per_class)
                    for cy, cx in self._sample_coordinates_from_slice(lbl_stack[int(cz)], int(lbl), n_samples, rng):
                        if not _is_valid_coord(context=context, name=name, z=int(cz), y=int(cy), x=int(cx)):
                            continue
                        if (int(cz), int(cy), int(cx)) in coords_in_use:
                            continue
                        coords_in_use.add((int(cz), int(cy), int(cx)))
                        gt_label = int(lbl_stack[int(cz), int(cy), int(cx)])
                        schedule.add_record(
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

    def _sample_coordinates_from_slice(
        self,
        z_slice: np.ndarray,
        class_label: int,
        num_samples: int,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        """Sample class-balanced in-slice coordinates for the initial labelled pool.

        Args:
            z_slice: Label slice from which to sample.
            class_label: Semantic class to sample.
            num_samples: Requested number of coordinates.
            rng: Random generator used for sampling.

        Returns:
            list[tuple[int, int]]: Sampled ``(y, x)`` coordinates.
        """

        label_coords = np.argwhere(z_slice == class_label)
        if len(label_coords) < num_samples:
            return []
        idx = rng.sample(range(len(label_coords)), num_samples)
        sampled = label_coords[idx]
        return [(int(y), int(x)) for y, x in sampled]


class ClassBalancedSubstackInitialLabelSamplingStrategy:
    """Initialize stage-0 labels by class-balanced sampling over whole substacks.

    Args:
        None

    Returns:
        ClassBalancedSubstackInitialLabelSamplingStrategy: Strategy sampling GT voxels per class over the full substack volume.
    """

    def populate_schedule(
        self,
        schedule: DataSchedule,
        context: InitialLabelSamplingContext,
        rng: random.Random,
    ) -> None:
        """Populate the schedule with class-balanced GT voxels sampled over each full substack.

        Args:
            schedule: Schedule to mutate.
            context: Immutable stage-0 sampling context.
            rng: Random generator used for substack-level class-balanced sampling.

        Returns:
            None
        """

        if not context.train_substacks:
            raise ValueError("class_balanced_substack requires train_substacks to be available.")

        for substack in context.train_substacks:
            name = substack["stack_name"]
            name_id = context.name_to_id[name]
            z_start = int(substack["z_start"])
            z_stop = int(substack["z_stop"])
            coords_in_use = set(map(tuple, schedule["coords"][schedule["name_id"] == name_id]))
            label_volume = context.labels[name][z_start:z_stop]
            for lbl in context.unique_labels:
                n_samples = context.samples_per_class.get(int(lbl), context.default_samples_per_class)
                for z, y, x in self._sample_coordinates_from_substack(
                    label_volume=label_volume,
                    class_label=int(lbl),
                    num_samples=n_samples,
                    z_start=z_start,
                    rng=rng,
                ):
                    if not _is_valid_coord(context=context, name=name, z=z, y=y, x=x):
                        continue
                    if (z, y, x) in coords_in_use:
                        continue
                    coords_in_use.add((z, y, x))
                    gt_label = int(context.labels[name][z, y, x])
                    schedule.add_record(
                        name_id=name_id,
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

    def _sample_coordinates_from_substack(
        self,
        label_volume: np.ndarray,
        class_label: int,
        num_samples: int,
        z_start: int,
        rng: random.Random,
    ) -> list[tuple[int, int, int]]:
        """Sample class-balanced coordinates over the full substack volume.

        Args:
            label_volume: Label subvolume for one train substack.
            class_label: Semantic class to sample.
            num_samples: Requested number of coordinates for that class.
            z_start: Global z-offset of the substack.
            rng: Random generator used for sampling.

        Returns:
            list[tuple[int, int, int]]: Sampled global ``(z, y, x)`` coordinates.
        """

        label_coords = np.argwhere(label_volume == class_label)
        if len(label_coords) == 0 or num_samples <= 0:
            return []
        n_take = min(len(label_coords), num_samples)
        sampled_idx = rng.sample(range(len(label_coords)), n_take)
        sampled = label_coords[sampled_idx]
        return [(int(z_start + z), int(y), int(x)) for z, y, x in sampled]
