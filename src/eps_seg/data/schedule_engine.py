from __future__ import annotations

from typing import Callable, Literal, Protocol

import numpy as np

from eps_seg.data.sampling import CandidateSamplingStrategy
from eps_seg.data.schedule import CandidateBatchEvaluation, DataSchedule, EvaluatedCandidate
from eps_seg.data.schedule_policies import PseudolabelMaintenancePolicy, ScheduleAdmissionPolicy


class SchedulerBackedDataset(Protocol):
    """Interface required by the schedule engine from a scheduler-backed dataset.

    Args:
        None

    Returns:
        SchedulerBackedDataset: Dataset interface used by the schedule engine.
    """

    id_to_name: dict[int, str]
    name_to_id: dict[str, int]

    def build_candidate_batch(self, coords_batch: list[tuple[str, int, int, int]]) -> dict:
        """Build an evaluator batch for newly sampled candidate coordinates.

        Args:
            coords_batch: Candidate coordinates to evaluate.

        Returns:
            dict: Batch dictionary consumed by the model evaluator.
        """

    def build_scheduler_batch(self, schedule_indices: list[int]) -> dict:
        """Build an evaluator batch for existing scheduled coordinates.

        Args:
            schedule_indices: Scheduler rows to reevaluate.

        Returns:
            dict: Batch dictionary consumed by the model evaluator.
        """


class ScheduleEngine:
    """
    Coordinator for pseudo-label reevaluation and staged scheduler extension.

    Args:
        dataset: Scheduler-backed dataset used to build evaluator batches.
        schedule: Mutable scheduler state.
        sampler: Candidate sampling strategy.
        admission_policy: Pseudo-label admission policy.
        maintenance_policy: Pseudo-label reevaluation/pruning policy.

    Returns:
        ScheduleEngine: Coordinator for staged scheduler mutations.
    """

    def __init__(
        self,
        dataset: SchedulerBackedDataset,
        schedule: DataSchedule,
        sampler: CandidateSamplingStrategy,
        admission_policy: ScheduleAdmissionPolicy,
        maintenance_policy: PseudolabelMaintenancePolicy,
    ) -> None:
        self.dataset = dataset
        self.schedule = schedule
        self.sampler = sampler
        self.admission_policy = admission_policy
        self.maintenance_policy = maintenance_policy

    def extend_schedule(
        self,
        stage_index: int,
        target_active_rows: int,
        evaluation_batch_size: int,
        evaluator: Callable[[dict], tuple[np.ndarray, np.ndarray] | CandidateBatchEvaluation],
        target_label_source: int = 1,
        candidate_evaluation_budget: int | None = None,
        candidate_order: Literal["fifo", "ascending", "descending"] = "fifo",
    ) -> int:
        """
        Extend the scheduler with new admitted rows.

        Args:
            stage_index: Stage receiving the new admitted rows.
            target_active_rows: Desired number of active rows after extension.
            evaluation_batch_size: Number of proposed candidates evaluated per round.
            evaluator: Callable returning ``(predicted_labels, confidences)`` or
                ``CandidateBatchEvaluation`` for a candidate batch.
            target_label_source: Label source to extend. 1 for pseudo-labels, 2 for active learning labels.
            candidate_evaluation_budget: Optional maximum number of newly sampled candidates to evaluate.
                When set, admission is run once on the evaluated candidate pool.
            candidate_order: Ordering applied to evaluated candidates before admission.

        Returns:
            int: Number of admitted rows.
        """

        if target_active_rows <= self.schedule.count_active_label_source_rows(target_label_source):
            return 0
        if candidate_evaluation_budget is not None:
            forbidden_coords = self.schedule.scheduled_coordinate_set(
                id_to_name=self.dataset.id_to_name,
                active_only=True,
            )
            coords_batch = self.sampler.sample_batch(
                forbidden_coords=forbidden_coords,
                batch_size=int(candidate_evaluation_budget),
            )
            evaluated_candidates: list[EvaluatedCandidate] = []
            for batch_start in range(0, len(coords_batch), int(evaluation_batch_size)):
                eval_coords = coords_batch[batch_start : batch_start + int(evaluation_batch_size)]
                evaluated_candidates.extend(
                    self._evaluate_candidate_batch(
                        coords_batch=eval_coords,
                        evaluator=evaluator,
                    )
                )
            evaluated_candidates = self._order_candidates(
                evaluated_candidates,
                candidate_order=candidate_order,
            )

            accepted = self.admission_policy.admit(
                stage_index=stage_index,
                evaluated_candidates=evaluated_candidates,
                schedule=self.schedule,
                name_to_id=self.dataset.name_to_id,
                target_active_rows=target_active_rows,
            )
            self.schedule.stage_index = max(self.schedule.stage_index, int(stage_index))
            return accepted

        forbidden_coords = self.schedule.scheduled_coordinate_set(id_to_name=self.dataset.id_to_name, active_only=True)
        accepted = 0
        no_progress_rounds = 0
        max_no_progress_rounds = 32

        while (
            self.schedule.count_active_label_source_rows(target_label_source) < target_active_rows
            and no_progress_rounds < max_no_progress_rounds
        ):
            coords_batch = self.sampler.sample_batch(forbidden_coords=forbidden_coords, batch_size=evaluation_batch_size)
            if len(coords_batch) == 0:
                break

            evaluated_candidates = self._evaluate_candidate_batch(
                coords_batch=coords_batch,
                evaluator=evaluator,
            )
            evaluated_candidates = self._order_candidates(
                evaluated_candidates,
                candidate_order=candidate_order,
            )

            accepted_this_round = self.admission_policy.admit(
                stage_index=stage_index,
                evaluated_candidates=evaluated_candidates,
                schedule=self.schedule,
                name_to_id=self.dataset.name_to_id,
                target_active_rows=target_active_rows,
            )
            accepted += accepted_this_round

            if accepted_this_round > 0:
                forbidden_coords = self.schedule.scheduled_coordinate_set(
                    id_to_name=self.dataset.id_to_name,
                    active_only=True,
                )
                no_progress_rounds = 0
            else:
                no_progress_rounds += 1
        self.schedule.stage_index = max(self.schedule.stage_index, int(stage_index))
        return accepted

    @staticmethod
    def _order_candidates(
        evaluated_candidates: list[EvaluatedCandidate],
        candidate_order: Literal["fifo", "ascending", "descending"] = "fifo",
    ) -> list[EvaluatedCandidate]:
        """
        Order evaluated candidates before admission.

        Args:
            evaluated_candidates: Candidate records to order.
            candidate_order: ``fifo`` preserves sampling order, ``ascending`` sorts low score first,
                and ``descending`` sorts high score first.

        Returns:
            list[EvaluatedCandidate]: Ordered candidate records.
        """

        if candidate_order == "fifo":
            return evaluated_candidates
        if candidate_order == "ascending":
            return sorted(evaluated_candidates, key=lambda candidate: candidate.confidence)
        if candidate_order == "descending":
            return sorted(evaluated_candidates, key=lambda candidate: candidate.confidence, reverse=True)
        raise ValueError(f"Unknown candidate order: {candidate_order}")

    def _evaluate_candidate_batch(
        self,
        coords_batch: list[tuple[str, int, int, int]],
        evaluator: Callable[[dict], tuple[np.ndarray, np.ndarray] | CandidateBatchEvaluation],
    ) -> list[EvaluatedCandidate]:
        """
        Build, evaluate, and convert one candidate-coordinate batch.

        Args:
            coords_batch: Candidate coordinates to evaluate.
            evaluator: Callable returning ``(predicted_labels, confidences)`` or
                ``CandidateBatchEvaluation`` for the batch.

        Returns:
            list[EvaluatedCandidate]: Evaluated candidate records.
        """

        batch = self.dataset.build_candidate_batch(coords_batch)
        evaluation = self._coerce_candidate_batch_evaluation(evaluator(batch))
        predicted_labels = evaluation.predicted_labels
        confidences = evaluation.scores
        metrics = {
            key: np.asarray(values)
            for key, values in evaluation.metrics.items()
        }
        return [
            EvaluatedCandidate(
                stack_name=name,
                z=int(z),
                y=int(y),
                x=int(x),
                predicted_label=int(pred_label),
                confidence=float(confidence),
                gt_label=int(gt_label),
                metrics={
                    metric_name: float(metric_values[idx])
                    for metric_name, metric_values in metrics.items()
                    if len(metric_values) > idx and np.asarray(metric_values[idx]).ndim == 0
                },
                class_probabilities=(
                    np.asarray(evaluation.class_probabilities[idx]).copy()
                    if evaluation.class_probabilities is not None
                    else None
                ),
                head_logits=(
                    np.asarray(evaluation.head_logits[idx]).copy()
                    if evaluation.head_logits is not None
                    else None
                ),
                head_predicted_labels=(
                    np.asarray(evaluation.head_predicted_labels[idx]).copy()
                    if evaluation.head_predicted_labels is not None
                    else None
                ),
                latent_summary={
                    key: np.asarray(values[idx]).copy()
                    for key, values in evaluation.latent_summary.items()
                    if len(values) > idx
                },
            )
            for idx, ((name, z, y, x), pred_label, confidence, gt_label) in enumerate(zip(
                coords_batch,
                predicted_labels,
                confidences,
                batch["gt"].tolist(),
            ))
        ]

    @staticmethod
    def _coerce_candidate_batch_evaluation(
        evaluation: tuple[np.ndarray, np.ndarray] | CandidateBatchEvaluation,
    ) -> CandidateBatchEvaluation:
        if isinstance(evaluation, CandidateBatchEvaluation):
            return evaluation
        predicted_labels, confidences = evaluation
        confidences = np.asarray(confidences, dtype=np.float32)
        return CandidateBatchEvaluation(
            predicted_labels=np.asarray(predicted_labels, dtype=np.int32),
            scores=confidences,
            metrics={"score": confidences},
        )

    def reevaluate_schedule(
        self,
        next_stage_idx: int,
        evaluation_batch_size: int,
        evaluator: Callable[[dict], tuple[np.ndarray, np.ndarray] | CandidateBatchEvaluation],
        target_label_source: int = 1,
    ) -> int:
        """
        Reevaluate active pseudo-labels and apply the configured maintenance policy.

        Args:
            next_stage_idx: Stage that is about to start.
            evaluation_batch_size: Number of active pseudo-label rows reevaluated per batch.
            evaluator: Callable returning ``(predicted_labels, confidences)`` or
                ``CandidateBatchEvaluation`` for a scheduler batch.
            target_label_source: Label source to reevaluate. 1 for pseudo-labels, 2 for active learning labels.

        Returns:
            int: Number of disabled pseudo-label rows.
        """

        pseudo_indices = self.schedule.get_active_label_source_indices(target_label_source).tolist()
        if len(pseudo_indices) == 0:
            return 0

        disabled_count = 0
        for batch_start in range(0, len(pseudo_indices), evaluation_batch_size):
            batch_indices = pseudo_indices[batch_start : batch_start + evaluation_batch_size]
            batch = self.dataset.build_scheduler_batch(batch_indices)
            evaluation = self._coerce_candidate_batch_evaluation(evaluator(batch))
            predicted_labels = evaluation.predicted_labels
            confidences = evaluation.scores
            disabled_count += self.maintenance_policy.reevaluate(
                next_stage_idx=next_stage_idx,
                batch_indices=batch_indices,
                predicted_labels=predicted_labels,
                confidences=confidences,
                schedule=self.schedule,
            )

        return disabled_count
