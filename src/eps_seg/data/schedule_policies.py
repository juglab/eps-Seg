from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from eps_seg.data.schedule import DataSchedule, EvaluatedCandidate


class PseudolabelAdmissionPolicy(Protocol):
    """
    Interface for pseudo-label admission policies.

    Args:
        None

    Returns:
        PseudolabelAdmissionPolicy: Object deciding which evaluated candidates enter the schedule.
    """

    def admit(
        self,
        stage_index: int,
        evaluated_candidates: list[EvaluatedCandidate],
        schedule: DataSchedule,
        name_to_id: dict[str, int],
        target_active_pseudolabels: int,
    ) -> int:
        """
        Admit evaluated candidates into the scheduler.

        Args:
            stage_index: Stage receiving the newly accepted pseudo-labels.
            evaluated_candidates: Evaluated candidate voxels with predictions and confidences.
            schedule: Schedule to mutate.
            name_to_id: Mapping from stack names to integer identifiers.
            target_active_pseudolabels: Desired number of active pseudo-labels after admission.

        Returns:
            int: Number of admitted pseudo-labels.
        """

    def admission_rule(self, confidence: float) -> bool:
        """
        Check whether a confidence value satisfies the admission rule.

        Args:
            confidence: Confidence attached to the candidate or reevaluated row.

        Returns:
            bool: Whether the confidence satisfies the admission rule.
        """


class PseudolabelMaintenancePolicy(Protocol):
    """
    Interface for pseudo-label reevaluation and pruning policies.

    Args:
        None

    Returns:
        PseudolabelMaintenancePolicy: Object deciding which active pseudo-labels remain enabled.
    """

    def reevaluate(
        self,
        next_stage_idx: int,
        batch_indices: list[int],
        predicted_labels: np.ndarray,
        confidences: np.ndarray,
        schedule: DataSchedule,
    ) -> int:
        """
        Apply maintenance logic to one reevaluated batch of scheduler rows.

        Args:
            next_stage_idx: Stage that is about to start.
            batch_indices: Scheduler indices reevaluated in the current batch.
            predicted_labels: Predicted labels for the same indices.
            confidences: Predicted confidences for the same indices.
            schedule: Schedule to mutate.

        Returns:
            int: Number of rows disabled by the policy.
        """


@dataclass
class ConfidenceThresholdAdmissionPolicy(PseudolabelAdmissionPolicy):
    """
    Confidence-window pseudo-label admission policy.

    Args:
        confidence_min: Minimum confidence required for admission.
        confidence_max: Maximum confidence allowed for admission.

    Returns:
        ConfidenceThresholdAdmissionPolicy: Admission policy accepting candidates inside a confidence window.
    """

    confidence_min: float
    confidence_max: float

    def admit(
        self,
        stage_index: int,
        evaluated_candidates: list[EvaluatedCandidate],
        schedule: DataSchedule,
        name_to_id: dict[str, int],
        target_active_pseudolabels: int,
    ) -> int:
        """
        Admit all evaluated candidates whose confidence lies inside the configured confidence window.

        Args:
            stage_index: Stage receiving the newly accepted pseudo-labels.
            evaluated_candidates: Evaluated candidate voxels with predictions and confidences.
            schedule: Schedule to mutate.
            name_to_id: Mapping from stack names to integer identifiers.
            target_active_pseudolabels: Desired number of active pseudo-labels after admission.

        Returns:
            int: Number of admitted pseudo-labels.
        """

        accepted = 0
        for candidate in evaluated_candidates:
            if schedule.count_active_pseudolabels() >= target_active_pseudolabels:
                break
            if not self.admission_rule(float(candidate.confidence)):
                continue
            schedule.add_record(
                name_id=name_to_id[candidate.stack_name],
                coords=(candidate.z, candidate.y, candidate.x),
                current_label=int(candidate.predicted_label),
                gt_label=int(candidate.gt_label),
                confidence=float(candidate.confidence),
                label_source=1,
                stage_index=int(stage_index),
                is_enabled=True,
                stage_disabled=-1,
                consecutive_keep_failures=0,
                last_predicted_label=int(candidate.predicted_label),
                last_confidence=float(candidate.confidence),
            )
            accepted += 1
        return accepted

    def admission_rule(self, confidence: float) -> bool:
        """
        Check whether a confidence value falls inside the configured admission window.

        Args:
            confidence: Confidence attached to the candidate or reevaluated row.

        Returns:
            bool: Whether the confidence falls inside the configured admission window.
        """

        return float(self.confidence_min) <= float(confidence) <= float(self.confidence_max)


@dataclass
class PruningPolicy(PseudolabelMaintenancePolicy):
    """
    Admission-rule-based pseudo-label reevaluation and pruning policy.

    Args:
        admission_policy: Admission policy used to decide whether a reevaluated row is still valid.
        pruning_patience: Number of failed reevaluations tolerated before disabling a row.
        enable_pruning: Whether failed rows should actually be disabled.

    Returns:
        PruningPolicy: Maintenance policy that disables rows after repeated admission-rule failures.
    """

    admission_policy: PseudolabelAdmissionPolicy
    pruning_patience: int
    enable_pruning: bool

    def reevaluate(
        self,
        next_stage_idx: int,
        batch_indices: list[int],
        predicted_labels: np.ndarray,
        confidences: np.ndarray,
        schedule: DataSchedule,
    ) -> int:
        """
        Reevaluate one batch of active pseudo-label rows and optionally disable rows that no longer satisfy admission.

        Args:
            next_stage_idx: Stage that is about to start.
            batch_indices: Scheduler indices reevaluated in the current batch.
            predicted_labels: Predicted labels for the same indices.
            confidences: Predicted confidences for the same indices.
            schedule: Schedule to mutate.

        Returns:
            int: Number of rows disabled by the policy.
        """

        disabled_count = 0
        for schedule_idx, pred_label, confidence in zip(batch_indices, predicted_labels, confidences):
            pred_label = int(pred_label)
            confidence = float(confidence)
            schedule["last_predicted_label"][schedule_idx] = pred_label
            schedule["last_confidence"][schedule_idx] = confidence

            keep_row = (
                pred_label == int(schedule["current_label"][schedule_idx])
                and self.admission_policy.admission_rule(confidence)
            )
            if keep_row:
                schedule["consecutive_keep_failures"][schedule_idx] = 0
                continue

            schedule["consecutive_keep_failures"][schedule_idx] += 1
            failures = int(schedule["consecutive_keep_failures"][schedule_idx])
            if self.enable_pruning and failures >= int(self.pruning_patience):
                schedule.disable_record(schedule_idx=schedule_idx, stage_disabled=int(next_stage_idx))
                disabled_count += 1
        return disabled_count
