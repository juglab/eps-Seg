from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, ItemsView, KeysView, Mapping, Optional, ValuesView

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluatedCandidate:
    """
    Container describing one evaluated voxel candidate.

    Args:
        stack_name: Name of the volume from which the voxel was sampled.
        z: Z coordinate of the voxel center.
        y: Y coordinate of the voxel center.
        x: X coordinate of the voxel center.
        predicted_label: Label predicted by the current model.
        confidence: Confidence assigned to the predicted label.
        gt_label: Ground-truth label at the candidate location.

    Returns:
        EvaluatedCandidate: Immutable candidate description used by admission policies.
    """

    stack_name: str
    z: int
    y: int
    x: int
    predicted_label: int
    confidence: float
    gt_label: int


class DataSchedule(Mapping[str, np.ndarray]):
    """
    Mutable scheduler state used by the staged pseudo-label pipeline.

    Args:
        fields: Dictionary containing the scheduler arrays.
        stage_index: Highest stage represented in the scheduler.
        seed: Sampling seed associated with the scheduler.
        sampling_version: Monotonic version counter used by samplers to detect schedule updates.

    Returns:
        DataSchedule: Scheduler wrapper exposing dict-like field access plus helper operations.
    """

    FIELD_DTYPES: Dict[str, np.dtype] = {
        "name_id": np.int32,
        "coords": np.int32,
        "current_label": np.int32,
        "gt_label": np.int32,
        "confidence": np.float32,
        "label_source": np.int32,
        "stage_index": np.int32,
        "is_enabled": np.bool_,
        "stage_disabled": np.int32,
        "consecutive_keep_failures": np.int32,
        "last_predicted_label": np.int32,
        "last_confidence": np.float32,
    }

    def __init__(
        self,
        fields: Dict[str, np.ndarray],
        stage_index: int = 0,
        seed: int = 42,
        sampling_version: int = 0,
    ) -> None:
        self._fields = fields
        self.stage_index = int(stage_index)
        self.seed = int(seed)
        self.sampling_version = int(sampling_version)

    @classmethod
    def empty(cls, seed: int = 42, stage_index: int = 0) -> "DataSchedule":
        """
        Build an empty scheduler with the default fields.

        Args:
            seed: Sampling seed associated with the scheduler.
            stage_index: Highest stage represented in the scheduler.

        Returns:
            DataSchedule: Empty scheduler with correctly typed arrays.
        """

        fields = {
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
        return cls(fields=fields, stage_index=stage_index, seed=seed, sampling_version=0)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, np.ndarray],
        seed: int = 42,
        stage_index: int = 0,
        sampling_version: int = 0,
    ) -> "DataSchedule":
        """
        Build a scheduler from an existing mapping of arrays.

        Args:
            mapping: Mapping containing the scheduler arrays.
            seed: Sampling seed associated with the scheduler.
            stage_index: Highest stage represented in the scheduler.
            sampling_version: Monotonic version counter used by samplers to detect schedule updates.

        Returns:
            DataSchedule: Scheduler object wrapping the provided arrays.
        """

        n_rows = len(np.asarray(mapping["name_id"]))
        fields: Dict[str, np.ndarray] = {}
        for key, dtype in cls.FIELD_DTYPES.items():
            if key in mapping:
                fields[key] = np.asarray(mapping[key], dtype=dtype)
            else:
                fields[key] = cls._default_field(key=key, n_rows=n_rows, mapping=mapping)

        if "metadata" in mapping:
            metadata = np.asarray(mapping["metadata"]).astype(np.int32)
            if metadata.size >= 1:
                seed = int(metadata[0])
            if metadata.size >= 2:
                stage_index = int(metadata[1])
        elif n_rows > 0:
            stage_index = int(np.max(fields["stage_index"]))

        return cls(fields=fields, stage_index=stage_index, seed=seed, sampling_version=sampling_version)

    @classmethod
    def load_npz(cls, path: Path) -> "DataSchedule":
        """
        Load a scheduler from an ``.npz`` file.

        Args:
            path: Path to the scheduler file.

        Returns:
            DataSchedule: Scheduler loaded from disk.
        """

        npz = np.load(path)
        return cls.from_mapping(mapping={key: np.array(npz[key]) for key in npz.files})

    @staticmethod
    def _default_field(key: str, n_rows: int, mapping: Mapping[str, np.ndarray]) -> np.ndarray:
        """
        Build a default array for a missing scheduler field.

        Args:
            key: Name of the missing field.
            n_rows: Number of rows already present in the schedule.
            mapping: Existing scheduler mapping.

        Returns:
            np.ndarray: Default array matching the missing field semantics.
        """

        if key == "is_enabled":
            return np.ones((n_rows,), dtype=np.bool_)
        if key == "stage_disabled":
            return np.full((n_rows,), -1, dtype=np.int32)
        if key == "consecutive_keep_failures":
            return np.zeros((n_rows,), dtype=np.int32)
        if key == "last_predicted_label":
            return np.asarray(mapping["current_label"], dtype=np.int32).copy()
        if key == "last_confidence":
            return np.asarray(mapping["confidence"], dtype=np.float32).copy()
        raise KeyError(f"Unsupported missing scheduler field: {key}")

    def __getitem__(self, key: str) -> np.ndarray:
        return self._fields[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self._fields[key] = np.asarray(value, dtype=self.FIELD_DTYPES[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, key: object) -> bool:
        return key in self._fields

    def keys(self) -> KeysView[str]:
        return self._fields.keys()

    def items(self) -> ItemsView[str, np.ndarray]:
        return self._fields.items()

    def values(self) -> ValuesView[np.ndarray]:
        return self._fields.values()

    def get(self, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self._fields.get(key, default)

    def bump_version(self) -> None:
        """
        Increase the scheduler version counter.

        Args:
            None

        Returns:
            None
        """

        self.sampling_version += 1

    def add_record(
        self,
        name_id: int,
        coords: tuple[int, int, int],
        current_label: int,
        gt_label: int,
        confidence: float,
        label_source: int,
        stage_index: int,
        is_enabled: bool = True,
        stage_disabled: int = -1,
        consecutive_keep_failures: int = 0,
        last_predicted_label: Optional[int] = None,
        last_confidence: Optional[float] = None,
    ) -> int:
        """
        Append one row to the scheduler.

        Args:
            name_id: Integer stack identifier.
            coords: Voxel coordinates as ``(z, y, x)``.
            current_label: Currently assigned label.
            gt_label: Ground-truth label at the same location.
            confidence: Confidence attached to the current label.
            label_source: Source code for the label.
            stage_index: Stage that introduced the row.
            is_enabled: Whether the row is active.
            stage_disabled: Stage at which the row was disabled, or ``-1`` if still active.
            consecutive_keep_failures: Number of consecutive reevaluation failures.
            last_predicted_label: Most recent reevaluated label. Defaults to ``current_label``.
            last_confidence: Most recent reevaluated confidence. Defaults to ``confidence``.

        Returns:
            int: Index of the appended row.
        """

        if last_predicted_label is None:
            last_predicted_label = int(current_label)
        if last_confidence is None:
            last_confidence = float(confidence)

        row_idx = int(len(self._fields["name_id"]))
        self._fields["name_id"] = np.append(self._fields["name_id"], np.array([name_id], dtype=np.int32))
        self._fields["coords"] = np.append(self._fields["coords"], [coords], axis=0)
        self._fields["current_label"] = np.append(self._fields["current_label"], np.array([current_label], dtype=np.int32))
        self._fields["gt_label"] = np.append(self._fields["gt_label"], np.array([gt_label], dtype=np.int32))
        self._fields["confidence"] = np.append(self._fields["confidence"], np.array([confidence], dtype=np.float32))
        self._fields["label_source"] = np.append(self._fields["label_source"], np.array([label_source], dtype=np.int32))
        self._fields["stage_index"] = np.append(self._fields["stage_index"], np.array([stage_index], dtype=np.int32))
        self._fields["is_enabled"] = np.append(self._fields["is_enabled"], np.array([is_enabled], dtype=np.bool_))
        self._fields["stage_disabled"] = np.append(self._fields["stage_disabled"], np.array([stage_disabled], dtype=np.int32))
        self._fields["consecutive_keep_failures"] = np.append(
            self._fields["consecutive_keep_failures"],
            np.array([consecutive_keep_failures], dtype=np.int32),
        )
        self._fields["last_predicted_label"] = np.append(
            self._fields["last_predicted_label"],
            np.array([last_predicted_label], dtype=np.int32),
        )
        self._fields["last_confidence"] = np.append(
            self._fields["last_confidence"],
            np.array([last_confidence], dtype=np.float32),
        )
        self.stage_index = max(self.stage_index, int(stage_index))
        return row_idx

    def disable_record(self, schedule_idx: int, stage_disabled: int) -> None:
        """
        Disable an existing scheduler row.

        Args:
            schedule_idx: Index of the row to disable.
            stage_disabled: Stage at which the row is being disabled.

        Returns:
            None
        """

        self._fields["is_enabled"][schedule_idx] = False
        self._fields["stage_disabled"][schedule_idx] = int(stage_disabled)

    def get_active_indices(self) -> np.ndarray:
        """
        Return the indices of active scheduler rows.

        Args:
            None

        Returns:
            np.ndarray: Active row indices.
        """

        if "is_enabled" not in self._fields:
            return np.arange(len(self._fields["name_id"]), dtype=np.int32)
        return np.where(self._fields["is_enabled"])[0].astype(np.int32)

    def get_active_pseudolabel_indices(self) -> np.ndarray:
        """
        Return the indices of active pseudo-labeled rows.

        Args:
            None

        Returns:
            np.ndarray: Active pseudo-label row indices.
        """

        pseudo_mask = self._fields["label_source"] == 1
        return np.where(self._fields["is_enabled"] & pseudo_mask)[0].astype(np.int32)

    def count_active_pseudolabels(self) -> int:
        """
        Count active pseudo-label rows.

        Args:
            None

        Returns:
            int: Number of active pseudo-label rows.
        """

        return int(len(self.get_active_pseudolabel_indices()))

    def scheduled_coordinate_set(
        self,
        id_to_name: Mapping[int, str],
        active_only: bool = True,
    ) -> set[tuple[str, int, int, int]]:
        """
        Return scheduled coordinates as a set of ``(stack_name, z, y, x)`` tuples.

        Args:
            id_to_name: Mapping from integer stack identifiers to names.
            active_only: Whether to include only active rows.

        Returns:
            set[tuple[str, int, int, int]]: Coordinate set used to avoid duplicates.
        """

        schedule_indices = self.get_active_indices() if active_only else np.arange(len(self._fields["name_id"]), dtype=np.int32)
        return {
            (id_to_name[int(name_id)], int(z), int(y), int(x))
            for name_id, (z, y, x) in zip(self._fields["name_id"][schedule_indices], self._fields["coords"][schedule_indices])
        }

    def save_npz(self, path: Path) -> None:
        """
        Save the scheduler to an ``.npz`` file.

        Args:
            path: Output path.

        Returns:
            None
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            name_id=self._fields["name_id"],
            coords=self._fields["coords"],
            current_label=self._fields["current_label"],
            gt_label=self._fields["gt_label"],
            confidence=self._fields["confidence"],
            label_source=self._fields["label_source"],
            stage_index=self._fields["stage_index"],
            is_enabled=self._fields["is_enabled"],
            stage_disabled=self._fields["stage_disabled"],
            consecutive_keep_failures=self._fields["consecutive_keep_failures"],
            last_predicted_label=self._fields["last_predicted_label"],
            last_confidence=self._fields["last_confidence"],
            metadata=np.array([self.seed, self.stage_index, len(self._fields["name_id"])], dtype=np.int32),
        )

    def print_report(self, id_to_name: Mapping[int, str], label_source_names: Mapping[int, str]) -> None:
        """
        Print a compact schedule summary grouped by stage, source, and stack.

        Args:
            id_to_name: Mapping from integer stack identifiers to names.
            label_source_names: Mapping from label source codes to display names.

        Returns:
            None
        """

        if len(self._fields["name_id"]) == 0:
            print("Empty scheduler.")
            return

        table = {key: value for key, value in self._fields.items()}
        table["z"] = self._fields["coords"][:, 0]
        table["y"] = self._fields["coords"][:, 1]
        table["x"] = self._fields["coords"][:, 2]
        del table["coords"]
        frame = pd.DataFrame(table)
        frame["name_id"] = frame["name_id"].apply(lambda x: id_to_name[int(x)])
        frame["label_source"] = frame["label_source"].apply(lambda x: label_source_names.get(int(x), f"source_{x}"))
        frame["is_enabled"] = frame["is_enabled"].astype(bool)
        print(
            frame.groupby(["stage_index", "label_source", "is_enabled", "name_id"])["current_label"]
            .value_counts()
            .unstack(fill_value=0)
        )
