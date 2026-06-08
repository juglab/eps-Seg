from __future__ import annotations

import random
from typing import Optional

from eps_seg.config.train import (
    CandidateSamplingConfig,
    InitialLabelSamplingConfig,
    ScheduleAdmissionConfig,
    PseudolabelMaintenanceConfig,
)
from eps_seg.data.initial_label_sampling import (
    ClassBalancedSliceInitialLabelSamplingStrategy,
    ClassBalancedSubstackInitialLabelSamplingStrategy,
    InitialLabelSamplingStrategy,
)
from eps_seg.data.sampling import (
    CandidateSamplingStrategy,
    ClassBalancedSubstackSamplingStrategy,
    SamplingDomain,
    UniformCoordinateSamplingStrategy,
)
from eps_seg.data.schedule_policies import (
    ConfidenceThresholdWithPlAdmissionPolicy,
    ConfidencePercentileWindowWithGtAdmissionPolicy,
    ConfidenceWindowWithGtAdmissionPolicy,
    AdmitAllWithGtAdmissionPolicy,
    NoOpMaintenancePolicy,
    ReconstructionErrorWithGtAdmissionPolicy,
    ScheduleAdmissionPolicy,
    PseudolabelMaintenancePolicy,
    PruningPolicy,
)


def build_candidate_sampling_strategy(
    cfg: CandidateSamplingConfig,
    domain: SamplingDomain,
    rng: random.Random,
    samples_per_class: Optional[dict[int, int]] = None,
) -> CandidateSamplingStrategy:
    """Build the configured candidate sampling strategy.

    Args:
        cfg: Candidate sampling configuration.
        domain: Immutable train-region sampling domain.
        rng: Random generator used by the strategy.
        samples_per_class: Optional class weighting specification.

    Returns:
        CandidateSamplingStrategy: Instantiated candidate sampling strategy.
    """

    if not domain.train_substacks:
        raise ValueError(
            "Candidate sampling requires train_substacks. Build or load the cache-v2 fold payload before staged training."
        )

    if cfg.name == "uniform_coordinate_sampling":
        return UniformCoordinateSamplingStrategy(domain=domain, rng=rng)
    if cfg.name == "class_balanced_substack":
        return ClassBalancedSubstackSamplingStrategy(
            domain=domain,
            rng=rng,
            samples_per_class=samples_per_class or {},
        )
    raise ValueError(f"Unknown candidate sampling strategy: {cfg.name}")


def build_initial_label_sampling_strategy(
    cfg: InitialLabelSamplingConfig | None,
    has_train_substacks: bool,
) -> InitialLabelSamplingStrategy:
    """Build the configured initial-label sampling strategy.

    Args:
        cfg: Optional initial-label sampling configuration.
        has_train_substacks: Whether cache-v2 train-substack descriptors are available.

    Returns:
        InitialLabelSamplingStrategy: Instantiated stage-0 initialization strategy.
    """

    strategy_name = cfg.name if cfg is not None else "class_balanced_substack"
    if strategy_name == "from_csv":
        raise ValueError(
            "Initial-label sampling strategy 'from_csv' has been removed. "
            "Use DatasetConfig.load_train_coords_from/load_val_coords_from for external coordinate import."
        )
    if not has_train_substacks:
        raise ValueError(
            f"Initial-label sampling strategy '{strategy_name}' requires train_substacks from cache-v2."
        )
    if strategy_name == "class_balanced_slice":
        return ClassBalancedSliceInitialLabelSamplingStrategy()
    if strategy_name == "class_balanced_substack":
        return ClassBalancedSubstackInitialLabelSamplingStrategy()
    raise ValueError(f"Unknown initial label sampling strategy: {strategy_name}")


def build_schedule_admission_policy(cfg: ScheduleAdmissionConfig) -> ScheduleAdmissionPolicy:
    """Build the configured schedule admission policy.

    Args:
        cfg: Schedule admission configuration.

    Returns:
        ScheduleAdmissionPolicy: Instantiated schedule admission policy.
    """

    if cfg.name == "confidence_threshold_with_pseudolabels":
        return ConfidenceThresholdWithPlAdmissionPolicy(confidence_min=cfg.confidence_min, confidence_max=cfg.confidence_max)
    if cfg.name == "confidence_window_with_gt":
        return ConfidenceWindowWithGtAdmissionPolicy(confidence_min=cfg.confidence_min, confidence_max=cfg.confidence_max)
    if cfg.name == "confidence_percentile_window_with_gt":
        return ConfidencePercentileWindowWithGtAdmissionPolicy(
            confidence_percentile_min=cfg.confidence_percentile_min,
            confidence_percentile_max=cfg.confidence_percentile_max,
        )
    if cfg.name == "admit_all_with_gt":
        return AdmitAllWithGtAdmissionPolicy()
    if cfg.name == "reconstruction_error_with_gt":
        return ReconstructionErrorWithGtAdmissionPolicy(score_metric=cfg.score_metric)
    raise ValueError(f"Unknown schedule admission policy: {cfg.name}")


def build_pseudolabel_maintenance_policy(
    cfg: PseudolabelMaintenanceConfig,
    admission_policy: ScheduleAdmissionPolicy,
) -> PseudolabelMaintenancePolicy:
    """Build the configured pseudo-label maintenance policy.

    Args:
        cfg: Pseudo-label maintenance configuration.
        admission_policy: Admission policy reused to decide whether reevaluated rows are still valid.

    Returns:
        PseudolabelMaintenancePolicy: Instantiated pseudo-label maintenance policy.
    """

    if cfg.name == "pruning":
        return PruningPolicy(
            admission_policy=admission_policy,
            pruning_patience=cfg.pruning_patience,
            enable_pruning=cfg.enable_pruning,
        )
    if cfg.name == "noop":
        return NoOpMaintenancePolicy()
    raise ValueError(f"Unknown pseudo-label maintenance policy: {cfg.name}")
