from eps_seg.dataloaders.samplers import PseudoEpochDistributedParallelBatchSampler
from tests.fixtures_samplers import (
    DummyDataset,
    VersionedBatchSampler,
    collect_pseudoepoch,
    flatten_batch_ids,
)


def test_all_batches_are_eventually_seen_across_pseudoepochs():
    """
    Goal: when the dataset does not change, repeated pseudo-epochs should walk
    through the underlying sampler rather than restarting from the beginning.
    """
    dataset = DummyDataset()
    sampler = VersionedBatchSampler(
        dataset,
        {
            0: [[0], [1], [2], [3], [4]],
        },
    )
    wrapper = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(5)),
        sampler=sampler,
        num_replicas=1,
        rank=0,
        batches_per_pseudoepoch=2,
    )

    epoch_1 = flatten_batch_ids(collect_pseudoepoch(wrapper))
    epoch_2 = flatten_batch_ids(collect_pseudoepoch(wrapper))
    epoch_3 = flatten_batch_ids(collect_pseudoepoch(wrapper))

    assert epoch_1 == [0, 1]
    assert epoch_2 == [2, 3]
    assert epoch_3 == [4, 0]
    assert set(epoch_1 + epoch_2 + epoch_3) == {0, 1, 2, 3, 4}


def test_stale_sampler_mid_true_epoch_restarts_from_new_ordering():
    """
    Goal: if the sampler becomes stale while a true epoch is only partially
    consumed, the next pseudo-epoch should restart from the new ordering.
    """
    dataset = DummyDataset()
    sampler = VersionedBatchSampler(
        dataset,
        {
            0: [[0], [1], [2], [3], [4]],
            1: [[100], [101], [102], [103], [104]],
        },
    )
    wrapper = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(5)),
        sampler=sampler,
        num_replicas=1,
        rank=0,
        batches_per_pseudoepoch=2,
    )

    first_epoch = flatten_batch_ids(collect_pseudoepoch(wrapper))
    dataset.sampling_version = 1
    second_epoch = flatten_batch_ids(collect_pseudoepoch(wrapper))

    assert first_epoch == [0, 1]
    assert second_epoch == [100, 101]
    assert second_epoch != [2, 3]
    assert wrapper.current_true_epoch == 1


def test_replicas_receive_equal_batches_per_pseudoepoch():
    """
    Goal: every replica should receive the same number of batches in one
    pseudo-epoch when ``batches_per_pseudoepoch`` is divisible by ``num_replicas``.
    """
    dataset = DummyDataset()
    sampler = VersionedBatchSampler(
        dataset,
        {
            0: [[0], [1], [2], [3], [4], [5]],
        },
    )

    rank0 = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(6)),
        sampler=sampler,
        num_replicas=2,
        rank=0,
        batches_per_pseudoepoch=4,
    )
    rank1 = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(6)),
        sampler=sampler,
        num_replicas=2,
        rank=1,
        batches_per_pseudoepoch=4,
    )

    rank0_batches = flatten_batch_ids(collect_pseudoepoch(rank0))
    rank1_batches = flatten_batch_ids(collect_pseudoepoch(rank1))

    assert len(rank0_batches) == 2
    assert len(rank1_batches) == 2
    assert rank0_batches == [0, 2]
    assert rank1_batches == [1, 3]


def test_without_pseudoepochs_batches_are_partitioned_by_rank():
    """
    Goal: without pseudo-epochs, the wrapper should yield the underlying batches
    assigned to the current rank and drop any incomplete remainder.
    """
    dataset = DummyDataset()
    sampler = VersionedBatchSampler(
        dataset,
        {
            0: [[0], [1], [2], [3], [4]],
        },
    )

    rank0 = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(5)),
        sampler=sampler,
        num_replicas=2,
        rank=0,
        batches_per_pseudoepoch=None,
    )
    rank1 = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(5)),
        sampler=sampler,
        num_replicas=2,
        rank=1,
        batches_per_pseudoepoch=None,
    )

    rank0_batches = flatten_batch_ids(list(iter(rank0)))
    rank1_batches = flatten_batch_ids(list(iter(rank1)))

    assert len(rank0_batches) == 2
    assert len(rank1_batches) == 2
    assert rank0_batches == [0, 2]
    assert rank1_batches == [1, 3]
    assert 4 not in rank0_batches + rank1_batches


def test_len_matches_batches_per_replica_with_pseudoepochs():
    """
    Goal: ``len(wrapper)`` should report the number of batches seen by one
    replica during a pseudo-epoch.
    """
    dataset = DummyDataset()
    sampler = VersionedBatchSampler(
        dataset,
        {
            0: [[0], [1], [2], [3], [4], [5]],
        },
    )

    wrapper = PseudoEpochDistributedParallelBatchSampler(
        dataset=list(range(6)),
        sampler=sampler,
        num_replicas=3,
        rank=1,
        batches_per_pseudoepoch=6,
    )

    assert len(wrapper) == 2
