import random

import numpy as np
import pytest
import torch

from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.datasets import SemisupervisedDataset
from eps_seg.dataloaders.samplers import ModeAwareBalancedAnchorBatchSampler
from eps_seg.models.lvae import LVAEModel


def make_minimal_semisupervised_dataset(neighbor_samples_per_anchor=1):
    dataset = SemisupervisedDataset.__new__(SemisupervisedDataset)
    dataset.mode = "semisupervised"
    dataset.dim = 2
    dataset.offset = 0
    dataset.n_neighbors = 7
    dataset.neighbor_samples_per_anchor = neighbor_samples_per_anchor
    dataset.neighbor_sampling_mode = "spatial"
    dataset.neighbor_label_mode = "pseudo"
    dataset.deterministic_neighbor_selection = False
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
    assert TrainConfig().resolved_validation_variant == "anchor_gt"
    assert TrainConfig(validation_use_pseudolabel_neighbors=True).resolved_validation_variant == "spatial_pl"
    assert TrainConfig(
        validation_use_pseudolabel_neighbors=True,
        validation_variant="random_gt",
    ).resolved_validation_variant == "random_gt"


def test_semisupervised_dataset_can_emit_neighbor_gt_labels():
    dataset = make_minimal_semisupervised_dataset(neighbor_samples_per_anchor=1)
    dataset.neighbor_label_mode = "gt"
    dataset.deterministic_neighbor_selection = True

    _, labels, _, coords = dataset[0]

    assert labels.tolist() == [2, 0]
    assert coords.tolist() == [[0, 10, 10], [0, 10, 11]]


def test_semisupervised_dataset_deterministic_neighbor_selection_is_stable():
    dataset = make_minimal_semisupervised_dataset(neighbor_samples_per_anchor=1)
    dataset.neighbor_label_mode = "pseudo"
    dataset.deterministic_neighbor_selection = True

    first = dataset[0][3].tolist()
    second = dataset[0][3].tolist()

    assert first == second == [[0, 10, 10], [0, 10, 11]]


def test_random_validation_neighbors_are_not_spatially_constrained():
    image = np.arange(1 * 30 * 30, dtype=np.float32).reshape(1, 30, 30)
    label = np.zeros((1, 30, 30), dtype=np.int64)
    records = [
        {"stack_name": "stack", "z": 0, "y": 10, "x": 10, "gt_label": 1, "z_start": 0, "z_stop": 1, "substack_id": 0, "coord_id": 0},
        {"stack_name": "stack", "z": 0, "y": 20, "x": 20, "gt_label": 2, "z_start": 0, "z_stop": 1, "substack_id": 0, "coord_id": 1},
    ]
    dataset = SemisupervisedDataset(
        images={"stack": image},
        labels={"stack": label},
        patch_size=2,
        label_size=1,
        mode="semisupervised",
        n_classes=3,
        radius=1,
        dim=2,
        seed=0,
        n_neighbors=1,
        neighbor_samples_per_anchor=1,
        neighbor_sampling_mode="random",
        neighbor_label_mode="gt",
        deterministic_neighbor_selection=True,
        coordinate_records=records,
    )

    _, labels, _, coords = dataset[0]

    assert labels.tolist() == [1, 2]
    assert coords.tolist() == [[0, 10, 10], [0, 20, 20]]


def test_validation_metric_targets_ignore_non_anchor_rows():
    labels = torch.tensor([1, 2, 3, 0])
    coords = torch.zeros(2, 2, 3, dtype=torch.long)

    targets = LVAEModel._validation_metric_targets("spatial_gt", labels, coords)

    assert targets.tolist() == [1, -1, 3, -1]
    assert torch.equal(LVAEModel._validation_metric_targets("anchor_gt", labels, coords), labels)


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
