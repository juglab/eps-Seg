import random

import numpy as np
import pytest

from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.datasets import SemisupervisedDataset
from eps_seg.dataloaders.samplers import ModeAwareBalancedAnchorBatchSampler


def make_minimal_semisupervised_dataset(neighbor_samples_per_anchor=1):
    dataset = SemisupervisedDataset.__new__(SemisupervisedDataset)
    dataset.mode = "semisupervised"
    dataset.dim = 2
    dataset.offset = 0
    dataset.n_neighbors = 7
    dataset.neighbor_samples_per_anchor = neighbor_samples_per_anchor
    dataset.rng = random.Random(0)
    dataset.images = {
        "stack": np.arange(30 * 30, dtype=np.float32).reshape(1, 30, 30),
    }
    dataset.labels = {
        "stack": np.zeros((1, 30, 30), dtype=np.int64),
    }
    dataset.groups = [
        {
            "name": "stack",
            "coords": [
                (0, 10, 10),
                (0, 10, 11),
                (0, 10, 12),
                (0, 10, 13),
                (0, 11, 10),
                (0, 12, 10),
                (0, 13, 10),
                (0, 14, 10),
            ],
            "labels": [2, 0, 0, 0, 0, 0, 0, 0],
        }
    ]
    return dataset


def test_semisupervised_dataset_randomly_emits_one_neighbor_per_anchor():
    dataset = make_minimal_semisupervised_dataset(neighbor_samples_per_anchor=1)

    seen_neighbors = set()
    for _ in range(10):
        patches, labels, segments, coords = dataset[0]
        assert patches.shape[0] == 2
        assert segments.shape[0] == 2
        assert labels.tolist() == [2, -1]
        seen_neighbors.add(tuple(coords[1].tolist()))

    assert len(seen_neighbors) > 1


@pytest.mark.parametrize("neighbor_samples_per_anchor", [1, 3, 7])
def test_semisupervised_dataset_emits_requested_power_of_two_group_sizes(
    neighbor_samples_per_anchor,
):
    dataset = make_minimal_semisupervised_dataset(
        neighbor_samples_per_anchor=neighbor_samples_per_anchor
    )

    patches, labels, segments, coords = dataset[0]

    expected_group_size = 1 + neighbor_samples_per_anchor
    assert patches.shape[0] == expected_group_size
    assert segments.shape[0] == expected_group_size
    assert coords.shape[0] == expected_group_size
    assert labels.tolist() == [2] + [-1] * neighbor_samples_per_anchor


@pytest.mark.parametrize("neighbor_samples_per_anchor", [2, 4, 5, 6])
def test_train_config_rejects_non_ablation_neighbor_counts(
    neighbor_samples_per_anchor,
):
    with pytest.raises(ValueError, match="neighbor_samples_per_anchor must be one of"):
        TrainConfig(neighbor_samples_per_anchor=neighbor_samples_per_anchor)


def test_pseudolabel_neighbor_validation_is_opt_in():
    assert TrainConfig().validation_use_pseudolabel_neighbors is False
    assert TrainConfig(
        validation_use_pseudolabel_neighbors=True
    ).validation_use_pseudolabel_neighbors is True


class DummyNeighborDataset:
    mode = "semisupervised"
    unique_labels = np.array([0, 1])
    neighbor_samples_per_anchor = 1

    def __init__(self):
        self.groups = [{"labels": [0]} for _ in range(8)] + [
            {"labels": [1]} for _ in range(8)
        ]


@pytest.mark.parametrize(
    ("neighbor_samples_per_anchor", "expected_anchors_per_batch"),
    [(1, 256), (3, 128), (7, 64)],
)
def test_neighbor_sampler_uses_emitted_neighbor_count_for_patch_batch_size(
    neighbor_samples_per_anchor,
    expected_anchors_per_batch,
):
    dataset = DummyNeighborDataset()
    dataset.neighbor_samples_per_anchor = neighbor_samples_per_anchor
    sampler = ModeAwareBalancedAnchorBatchSampler(
        dataset,
        total_patches_per_batch=512,
        n_neighbors=7,
        seed=3,
        shuffle=False,
    )

    batch = next(iter(sampler))

    assert len(batch) == expected_anchors_per_batch


def test_neighbor_sampler_treats_total_patches_as_budget_when_not_divisible():
    dataset = DummyNeighborDataset()
    dataset.neighbor_samples_per_anchor = 3
    sampler = ModeAwareBalancedAnchorBatchSampler(
        dataset,
        total_patches_per_batch=510,
        n_neighbors=7,
        seed=3,
        shuffle=False,
    )

    batch = next(iter(sampler))

    assert len(batch) == 127
