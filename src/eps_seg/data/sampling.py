from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np


Coordinate = tuple[str, int, int, int]


class CandidateSamplingStrategy(Protocol):
    """
    Interface for sampling candidate voxel coordinates from the train region.

    Returns:
        CandidateSamplingStrategy: Object able to propose candidate voxel coordinates.
    """

    def sample_candidate(self, forbidden_coords: set[Coordinate]) -> Coordinate | None:
        """
        Sample one candidate coordinate from the allowed train region.

        Args:
            forbidden_coords: Coordinates that must not be returned.

        Returns:
            Coordinate | None: Sampled coordinate, or ``None`` if none can be found.
        """

    def sample_batch(self, forbidden_coords: set[Coordinate], batch_size: int) -> list[Coordinate]:
        """
        Sample a batch of unique candidate coordinates from the allowed train region.

        Args:
            forbidden_coords: Coordinates that must not be returned.
            batch_size: Maximum number of candidates to sample.

        Returns:
            list[Coordinate]: Sampled candidate coordinates.
        """


@dataclass(frozen=True)
class SamplingDomain:
    """
    Immutable sampling domain for staged pseudo-label candidate selection.

    Args:
        images: Mapping from stack name to image volume.
        labels: Mapping from stack name to label volume.
        train_substacks: Substack descriptors defining the train region.
        ignore_lbl: Label value that must not be sampled.
        patch_size: Spatial patch size used by the dataset.
        label_size: Spatial label patch size used by the dataset.
        dim: Patch dimensionality.

    Returns:
        SamplingDomain: Immutable train-region sampling description.
    """

    images: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    train_substacks: List[Dict[str, int]]
    ignore_lbl: int
    patch_size: int
    label_size: int
    dim: int

    @property
    def offset(self) -> int:
        """
        Return the spatial patch offset used to validate candidate centers.

        Args:
            None

        Returns:
            int: Patch offset around the candidate center.
        """

        return self.patch_size // 2 - self.label_size


class UniformTrainRegionSamplingStrategy:
    """
    Uniform candidate sampler over the allowed train-region voxels.

    Args:
        domain: Immutable sampling domain.
        rng: Random generator used for candidate sampling.

    Returns:
        UniformTrainRegionSamplingStrategy: Sampler reproducing the current uniform train-region behavior.
    """

    def __init__(self, domain: SamplingDomain, rng: random.Random) -> None:
        self.domain = domain
        self.rng = rng
        self.slice_sampling_table = self._build_slice_sampling_table()

    def _build_slice_sampling_table(self) -> list[tuple[str, int]]:
        """
        Build the list of eligible ``(stack_name, z)`` sampling locations.

        Args:
            None

        Returns:
            list[tuple[str, int]]: Slice-level sampling domain.
        """

        table: list[tuple[str, int]] = []
        for substack in self.domain.train_substacks:
            name = substack["stack_name"]
            for z in range(int(substack["z_start"]), int(substack["z_stop"])):
                table.append((name, int(z)))
        return table

    def _is_valid_coord(self, name: str, z: int, y: int, x: int) -> bool:
        """
        Check whether a voxel center can be used for patch extraction and labeling.

        Args:
            name: Stack name.
            z: Z coordinate.
            y: Y coordinate.
            x: X coordinate.

        Returns:
            bool: Whether the coordinate is valid for the current patch geometry.
        """

        image = self.domain.images[name]
        z_size, height, width = image.shape
        offset = self.domain.offset
        valid = offset <= y < height - offset - 1 and offset <= x < width - offset - 1
        if self.domain.dim == 3:
            valid = valid and (offset <= z < z_size - offset - 1)
        return bool(valid and self.domain.labels[name][z, y, x] != self.domain.ignore_lbl)

    def sample_candidate(self, forbidden_coords: set[Coordinate]) -> Coordinate | None:
        """
        Sample one candidate coordinate uniformly from the allowed train region.

        Args:
            forbidden_coords: Coordinates that must not be returned.

        Returns:
            Coordinate | None: Sampled coordinate, or ``None`` if no valid candidate is found.
        """

        if len(self.slice_sampling_table) == 0:
            return None

        for _ in range(1024):
            name, z = self.rng.choice(self.slice_sampling_table)
            valid_positions = np.argwhere(self.domain.labels[name][z] != self.domain.ignore_lbl)
            if len(valid_positions) == 0:
                continue
            y, x = valid_positions[self.rng.randrange(len(valid_positions))]
            coord = (name, int(z), int(y), int(x))
            if coord in forbidden_coords:
                continue
            if not self._is_valid_coord(name=name, z=int(z), y=int(y), x=int(x)):
                continue
            return coord
        return None

    def sample_batch(self, forbidden_coords: set[Coordinate], batch_size: int) -> list[Coordinate]:
        """
        Sample a batch of unique candidate coordinates uniformly from the allowed train region.

        Args:
            forbidden_coords: Coordinates that must not be returned.
            batch_size: Maximum number of coordinates to sample.

        Returns:
            list[Coordinate]: Sampled candidate coordinates.
        """

        coords_batch: list[Coordinate] = []
        local_forbidden = set(forbidden_coords)
        for _ in range(batch_size):
            candidate = self.sample_candidate(local_forbidden)
            if candidate is None:
                break
            coords_batch.append(candidate)
            local_forbidden.add(candidate)
        return coords_batch


class ClassWeightedTrainRegionSamplingStrategy(UniformTrainRegionSamplingStrategy):
    """
    Placeholder class-weighted candidate sampler.

    Args:
        domain: Immutable sampling domain.
        rng: Random generator used for candidate sampling.
        samples_per_class: Optional class weighting specification.

    Returns:
        ClassWeightedTrainRegionSamplingStrategy: Sampler keeping the strategy interface stable for future weighting logic.
    """

    def __init__(
        self,
        domain: SamplingDomain,
        rng: random.Random,
        samples_per_class: Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__(domain=domain, rng=rng)
        self.samples_per_class = samples_per_class or {}
