import numpy as np

from eps_seg.dataloaders.samplers import BalancedScheduledBatchSampler
from tests.fixtures_samplers import (
    DummyPseudoLabelDataset,
    batch_labels_and_stages,
    build_dummy_schedule,
)


def test_batches_are_class_balanced_from_current_labels():
    """
    Goal: every batch should be balanced over ``current_label``, regardless of
    how many items each class has in the full scheduler.
    """
    dataset = DummyPseudoLabelDataset(
        build_dummy_schedule(
            {
                0: {0: 30, 1: 30, 2: 30, 3: 30},
                1: {0: 70, 1: 70, 2: 70, 3: 70},
            }
        )
    )
    sampler = BalancedScheduledBatchSampler(
        dataset,
        batch_size=20,
        min_initial_label_fraction=0.10,
        seed=7,
        shuffle=True,
    )

    batch_indices = next(iter(sampler))
    labels, _, _ = batch_labels_and_stages(dataset, batch_indices)
    unique_labels, counts = np.unique(labels, return_counts=True)
    per_label = dict(zip(unique_labels.tolist(), counts.tolist()))

    assert per_label == {0: 5, 1: 5, 2: 5, 3: 5}


def test_batches_follow_stage_distribution_when_it_matches_the_minimum():
    """
    Goal: if the schedule stage distribution already satisfies the initial-label
    minimum, the batch should reproduce that stage distribution exactly.
    """
    dataset = DummyPseudoLabelDataset(
        build_dummy_schedule(
            {
                0: {0: 10, 1: 10, 2: 10, 3: 10},
                1: {0: 30, 1: 30, 2: 30, 3: 30},
                2: {0: 60, 1: 60, 2: 60, 3: 60},
            }
        )
    )
    sampler = BalancedScheduledBatchSampler(
        dataset,
        batch_size=20,
        min_initial_label_fraction=0.10,
        seed=11,
        shuffle=True,
    )

    batch_indices = next(iter(sampler))
    _, stages, label_sources = batch_labels_and_stages(dataset, batch_indices)
    unique_stages, counts = np.unique(stages, return_counts=True)
    per_stage = dict(zip(unique_stages.tolist(), counts.tolist()))

    assert per_stage == {0: 2, 1: 6, 2: 12}
    assert int((label_sources == 0).sum()) == 2


def test_minimum_initial_fraction_is_enforced_when_stage_zero_is_small():
    """
    Goal: if stage 0 is underrepresented in the scheduler, the sampler should
    still reserve at least the configured fraction of initial labels.
    """
    dataset = DummyPseudoLabelDataset(
        build_dummy_schedule(
            {
                0: {0: 5, 1: 5, 2: 5, 3: 5},
                1: {0: 25, 1: 25, 2: 25, 3: 25},
                2: {0: 75, 1: 75, 2: 75, 3: 75},
            }
        )
    )
    sampler = BalancedScheduledBatchSampler(
        dataset,
        batch_size=20,
        min_initial_label_fraction=0.20,
        seed=19,
        shuffle=True,
    )

    batch_indices = next(iter(sampler))
    _, stages, label_sources = batch_labels_and_stages(dataset, batch_indices)
    unique_stages, counts = np.unique(stages, return_counts=True)
    per_stage = dict(zip(unique_stages.tolist(), counts.tolist()))

    assert int((label_sources == 0).sum()) >= 4
    assert per_stage == {0: 4, 1: 4, 2: 12}


def test_sampling_is_reproducible_for_the_same_seed():
    """
    Goal: two samplers with the same seed and the same scheduler should emit the
    same sequence of batches.
    """
    dataset = DummyPseudoLabelDataset(
        build_dummy_schedule(
            {
                0: {0: 20, 1: 20, 2: 20, 3: 20},
                1: {0: 40, 1: 40, 2: 40, 3: 40},
                2: {0: 40, 1: 40, 2: 40, 3: 40},
            }
        )
    )
    sampler_a = BalancedScheduledBatchSampler(dataset, batch_size=16, min_initial_label_fraction=0.125, seed=123, shuffle=True)
    sampler_b = BalancedScheduledBatchSampler(dataset, batch_size=16, min_initial_label_fraction=0.125, seed=123, shuffle=True)

    batches_a = [batch for _, batch in zip(range(3), iter(sampler_a))]
    batches_b = [batch for _, batch in zip(range(3), iter(sampler_b))]

    assert batches_a == batches_b


def test_missing_intermediate_stages_do_not_break_stage_aware_sampling():
    """
    Goal: stage-aware sampling should still work when some stage indices are
    absent from the current scheduler.
    """
    dataset = DummyPseudoLabelDataset(
        build_dummy_schedule(
            {
                0: {0: 20, 1: 20, 2: 20, 3: 20},
                2: {0: 80, 1: 80, 2: 80, 3: 80},
            }
        )
    )
    sampler = BalancedScheduledBatchSampler(
        dataset,
        batch_size=20,
        min_initial_label_fraction=0.10,
        seed=31,
        shuffle=True,
    )

    batch_indices = next(iter(sampler))
    _, stages, label_sources = batch_labels_and_stages(dataset, batch_indices)

    assert set(np.unique(stages).tolist()) == {0, 2}
    assert int((label_sources == 0).sum()) >= 2


def test_sampler_rebuilds_after_scheduler_changes():
    """
    Goal: if the dataset scheduler changes and bumps ``sampling_version``, the
    sampler should rebuild its internal pools and reflect the new composition.
    """
    dataset = DummyPseudoLabelDataset(
        build_dummy_schedule(
            {
                0: {0: 20, 1: 20, 2: 20, 3: 20},
                1: {0: 80, 1: 80, 2: 80, 3: 80},
            }
        )
    )
    sampler = BalancedScheduledBatchSampler(
        dataset,
        batch_size=20,
        min_initial_label_fraction=0.10,
        seed=41,
        shuffle=True,
    )

    first_batch = next(iter(sampler))
    _, first_stages, _ = batch_labels_and_stages(dataset, first_batch)
    assert set(np.unique(first_stages).tolist()) == {0, 1}

    dataset.schedule = build_dummy_schedule(
        {
            0: {0: 20, 1: 20, 2: 20, 3: 20},
            2: {0: 80, 1: 80, 2: 80, 3: 80},
        }
    )
    dataset.sampling_version += 1

    second_batch = next(iter(sampler))
    _, second_stages, _ = batch_labels_and_stages(dataset, second_batch)
    assert set(np.unique(second_stages).tolist()) == {0, 2}
