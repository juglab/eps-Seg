from __future__ import annotations

import random
from typing import Optional

from eps_seg.config.train import (
    CandidateSamplingConfig,
    InitialLabelSamplingConfig,
    PseudolabelAdmissionConfig,
    PseudolabelMaintenanceConfig,
)
from eps_seg.data.initial_label_sampling import (
    ClassBalancedSliceInitialLabelSamplingStrategy,
    ClassBalancedSubstackInitialLabelSamplingStrategy,
    FromCSVInitialLabelSamplingStrategy,
    InitialLabelSamplingStrategy,
)
from eps_seg.data.sampling import (
    CandidateSamplingStrategy,
    ClassWeightedTrainRegionSamplingStrategy,
    SamplingDomain,
    UniformTrainRegionSamplingStrategy,
)
from eps_seg.data.schedule_policies import (
    ConfidenceThresholdAdmissionPolicy,
    PseudolabelAdmissionPolicy,
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

    if cfg.name == "uniform_train_region":
        return UniformTrainRegionSamplingStrategy(domain=domain, rng=rng)
    if cfg.name == "class_weighted_train_region":
        return ClassWeightedTrainRegionSamplingStrategy(domain=domain, rng=rng, samples_per_class=samples_per_class)
    raise ValueError(f"Unknown candidate sampling strategy: {cfg.name}")


def build_initial_label_sampling_strategy(
    cfg: InitialLabelSamplingConfig | None,
    has_anchor_records: bool,
    has_train_substacks: bool,
) -> InitialLabelSamplingStrategy:
    """Build the configured initial-label sampling strategy.

    Args:
        cfg: Optional initial-label sampling configuration.
        has_anchor_records: Whether canonical labelled-voxel records are available.
        has_train_substacks: Whether cache-v2 train-substack descriptors are available.

    Returns:
        InitialLabelSamplingStrategy: Instantiated stage-0 initialization strategy.
    """

    strategy_name = cfg.name if cfg is not None else ("from_csv" if has_anchor_records else "class_balanced_substack")
    if strategy_name == "from_csv":
        return FromCSVInitialLabelSamplingStrategy()
    if not has_train_substacks:
        raise ValueError(
            f"Initial-label sampling strategy '{strategy_name}' requires train_substacks from cache-v2."
        )
    if strategy_name == "class_balanced_slice":
        return ClassBalancedSliceInitialLabelSamplingStrategy()
    if strategy_name == "class_balanced_substack":
        return ClassBalancedSubstackInitialLabelSamplingStrategy()
    raise ValueError(f"Unknown initial label sampling strategy: {strategy_name}")


def build_pseudolabel_admission_policy(cfg: PseudolabelAdmissionConfig) -> PseudolabelAdmissionPolicy:
    """Build the configured pseudo-label admission policy.

    Args:
        cfg: Pseudo-label admission configuration.

    Returns:
        PseudolabelAdmissionPolicy: Instantiated pseudo-label admission policy.
    """

    if cfg.name == "confidence_threshold":
        return ConfidenceThresholdAdmissionPolicy(confidence_min=cfg.confidence_min, confidence_max=cfg.confidence_max)
    raise ValueError(f"Unknown pseudo-label admission policy: {cfg.name}")


def build_pseudolabel_maintenance_policy(
    cfg: PseudolabelMaintenanceConfig,
    admission_policy: PseudolabelAdmissionPolicy,
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
    raise ValueError(f"Unknown pseudo-label maintenance policy: {cfg.name}")
