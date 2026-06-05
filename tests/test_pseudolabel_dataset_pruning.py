import numpy as np
import pytest

from eps_seg.config.train import TrainConfig
from eps_seg.data.schedule import DataSchedule
from eps_seg.dataloaders.datasets import PseudoLabelDataset


def make_runtime_dataset(train_cfg: TrainConfig | None = None):
    images = {f"vol{i}": np.zeros((1, 16, 16), dtype=np.float32) for i in range(3)}
    labels = {}
    for i in range(3):
        yy, xx = np.indices((16, 16))
        labels[f"vol{i}"] = ((yy + xx + i) % 4)[None, ...].astype(np.int32)

    dataset = PseudoLabelDataset(
        images=images,
        labels=labels,
        patch_size=4,
        label_size=1,
        n_classes=4,
        ignore_lbl=-1,
        dim=2,
        seed=0,
        samples_per_class={},
        train_substacks=[
            {"substack_id": i, "stack_name": f"vol{i}", "z_start": 0, "z_stop": 1, "depth": 1}
            for i in range(3)
        ],
        train_cfg=train_cfg,
    )
    return dataset


def set_schedule(dataset, rows):
    fields = {
        "name_id": [],
        "coords": [],
        "current_label": [],
        "gt_label": [],
        "confidence": [],
        "label_source": [],
        "stage_index": [],
        "is_enabled": [],
        "stage_disabled": [],
        "consecutive_keep_failures": [],
        "last_predicted_label": [],
        "last_confidence": [],
    }
    for row in rows:
        for key in ("name_id", "current_label", "gt_label", "label_source", "stage_index", "stage_disabled", "consecutive_keep_failures", "last_predicted_label"):
            fields[key].append(int(row[key]))
        fields["coords"].append(tuple(int(v) for v in row["coords"]))
        fields["confidence"].append(float(row["confidence"]))
        fields["is_enabled"].append(bool(row["is_enabled"]))
        fields["last_confidence"].append(float(row["last_confidence"]))

    dataset._schedule = DataSchedule.from_mapping(
        mapping={
            "name_id": np.asarray(fields["name_id"], dtype=np.int32),
            "coords": np.asarray(fields["coords"], dtype=np.int32),
            "current_label": np.asarray(fields["current_label"], dtype=np.int32),
            "gt_label": np.asarray(fields["gt_label"], dtype=np.int32),
            "confidence": np.asarray(fields["confidence"], dtype=np.float32),
            "label_source": np.asarray(fields["label_source"], dtype=np.int32),
            "stage_index": np.asarray(fields["stage_index"], dtype=np.int32),
            "is_enabled": np.asarray(fields["is_enabled"], dtype=np.bool_),
            "stage_disabled": np.asarray(fields["stage_disabled"], dtype=np.int32),
            "consecutive_keep_failures": np.asarray(fields["consecutive_keep_failures"], dtype=np.int32),
            "last_predicted_label": np.asarray(fields["last_predicted_label"], dtype=np.int32),
            "last_confidence": np.asarray(fields["last_confidence"], dtype=np.float32),
        },
        seed=dataset.seed,
        stage_index=dataset.stage_index,
        sampling_version=dataset.sampling_version,
    )
    dataset._build_components()
    dataset.schedule.bump_version()
    dataset.sampling_version = dataset.schedule.sampling_version


def make_row(idx, *, name_id, coords, current_label, gt_label, label_source, stage_index, is_enabled=True, stage_disabled=-1, consecutive_keep_failures=0, confidence=1.0, last_predicted_label=None, last_confidence=None):
    if last_predicted_label is None:
        last_predicted_label = current_label
    if last_confidence is None:
        last_confidence = confidence
    return {
        "idx": idx,
        "name_id": name_id,
        "coords": coords,
        "current_label": current_label,
        "gt_label": gt_label,
        "confidence": confidence,
        "label_source": label_source,
        "stage_index": stage_index,
        "is_enabled": is_enabled,
        "stage_disabled": stage_disabled,
        "consecutive_keep_failures": consecutive_keep_failures,
        "last_predicted_label": last_predicted_label,
        "last_confidence": last_confidence,
    }


def test_gt_rows_are_never_pruned_and_pseudo_rows_disable_after_patience():
    dataset = make_runtime_dataset()
    set_schedule(
        dataset,
        [
            make_row(0, name_id=0, coords=(0, 4, 4), current_label=0, gt_label=0, label_source=0, stage_index=0),
            make_row(1, name_id=1, coords=(0, 5, 5), current_label=1, gt_label=1, label_source=1, stage_index=1, confidence=0.9),
        ],
    )
    dataset.schedule_engine.maintenance_policy.enable_pruning = True
    dataset.schedule_engine.maintenance_policy.pruning_patience = 1

    def evaluator(batch):
        preds = np.array([0, 3], dtype=np.int32)
        confs = np.array([0.2, 0.2], dtype=np.float32)
        return preds, confs

    disabled = dataset.schedule_engine.reevaluate_schedule(
        next_stage_idx=2,
        evaluator=evaluator,
        evaluation_batch_size=8,
    )

    assert disabled == 1
    assert bool(dataset.schedule["is_enabled"][0]) is True
    assert bool(dataset.schedule["is_enabled"][1]) is False
    assert int(dataset.schedule["stage_disabled"][1]) == 2


def test_successful_reevaluation_resets_failure_counter():
    dataset = make_runtime_dataset()
    set_schedule(
        dataset,
        [
            make_row(
                0,
                name_id=0,
                coords=(0, 4, 4),
                current_label=2,
                gt_label=2,
                label_source=1,
                stage_index=1,
                confidence=0.8,
                consecutive_keep_failures=1,
            ),
        ],
    )
    dataset.schedule_engine.maintenance_policy.enable_pruning = True
    dataset.schedule_engine.maintenance_policy.pruning_patience = 2

    def evaluator(batch):
        return np.array([2], dtype=np.int32), np.array([0.95], dtype=np.float32)

    disabled = dataset.schedule_engine.reevaluate_schedule(
        next_stage_idx=2,
        evaluator=evaluator,
        evaluation_batch_size=4,
    )

    assert disabled == 0
    assert int(dataset.schedule["consecutive_keep_failures"][0]) == 0
    assert bool(dataset.schedule["is_enabled"][0]) is True
    assert int(dataset.schedule["last_predicted_label"][0]) == 2
    assert float(dataset.schedule["last_confidence"][0]) == pytest.approx(0.95)


def test_add_pseudolabels_tops_up_to_target_active_count_after_pruning():
    dataset = make_runtime_dataset()
    set_schedule(
        dataset,
        [
            make_row(0, name_id=0, coords=(0, 4, 4), current_label=0, gt_label=0, label_source=0, stage_index=0),
            make_row(1, name_id=0, coords=(0, 5, 5), current_label=1, gt_label=1, label_source=1, stage_index=1, confidence=0.9),
            make_row(2, name_id=1, coords=(0, 6, 6), current_label=2, gt_label=2, label_source=1, stage_index=1, confidence=0.9),
        ],
    )
    dataset.schedule_engine.maintenance_policy.enable_pruning = True
    dataset.schedule_engine.maintenance_policy.pruning_patience = 1

    def fail_one(batch):
        preds = []
        confs = []
        for schedule_idx in batch["schedule_idx"].tolist():
            if schedule_idx == 1:
                preds.append(3)
                confs.append(0.2)
            else:
                preds.append(int(dataset.schedule["current_label"][schedule_idx]))
                confs.append(0.95)
        return np.asarray(preds, dtype=np.int32), np.asarray(confs, dtype=np.float32)

    dataset.schedule_engine.reevaluate_schedule(
        next_stage_idx=2,
        evaluator=fail_one,
        evaluation_batch_size=4,
    )

    original_sampler = dataset.schedule_engine.sampler
    candidate_coords = [("vol2", 0, 7, 7), ("vol1", 0, 8, 8), ("vol0", 0, 9, 9)]

    class _Sampler:
        def sample_batch(self, forbidden_coords, batch_size):
            return [coord for coord in candidate_coords if coord not in forbidden_coords][:batch_size]

    def evaluator(batch):
        return np.array([1, 2, 3], dtype=np.int32), np.array([0.95, 0.96, 0.97], dtype=np.float32)

    dataset.schedule_engine.sampler = _Sampler()
    accepted = dataset.schedule_engine.extend_schedule(
        stage_index=2,
        target_active_rows=4,
        evaluator=evaluator,
        evaluation_batch_size=4,
    )
    dataset.schedule_engine.sampler = original_sampler

    assert accepted == 3
    assert dataset.schedule.count_active_pseudolabels() == 4


def test_re_admitting_disabled_coordinate_creates_new_active_row():
    dataset = make_runtime_dataset()
    set_schedule(
        dataset,
        [
            make_row(0, name_id=0, coords=(0, 4, 4), current_label=0, gt_label=0, label_source=0, stage_index=0),
            make_row(
                1,
                name_id=1,
                coords=(0, 5, 5),
                current_label=1,
                gt_label=1,
                label_source=1,
                stage_index=1,
                is_enabled=False,
                stage_disabled=2,
                confidence=0.8,
                last_confidence=0.2,
            ),
        ],
    )

    original_sampler = dataset.schedule_engine.sampler
    candidate_coords = [("vol1", 0, 5, 5)]

    class _Sampler:
        def sample_batch(self, forbidden_coords, batch_size):
            return [coord for coord in candidate_coords if coord not in forbidden_coords][:batch_size]

    def evaluator(batch):
        return np.array([1], dtype=np.int32), np.array([0.97], dtype=np.float32)

    dataset.schedule_engine.sampler = _Sampler()
    accepted = dataset.schedule_engine.extend_schedule(
        stage_index=3,
        target_active_rows=1,
        evaluator=evaluator,
        evaluation_batch_size=4,
    )
    dataset.schedule_engine.sampler = original_sampler

    assert accepted == 1
    assert len(dataset.schedule["name_id"]) == 3
    assert bool(dataset.schedule["is_enabled"][1]) is False
    assert bool(dataset.schedule["is_enabled"][2]) is True
    assert tuple(dataset.schedule["coords"][1]) == tuple(dataset.schedule["coords"][2])


def test_active_learning_extension_stores_ground_truth_with_source_two():
    dataset = make_runtime_dataset(
        TrainConfig(
            training_regime="active_learning",
            candidate_sampling={"name": "uniform_coordinate_sampling"},
            schedule_admission={"name": "admit_all_with_gt"},
            pseudolabel_maintenance={"name": "noop"},
        )
    )
    set_schedule(
        dataset,
        [
            make_row(0, name_id=0, coords=(0, 4, 4), current_label=0, gt_label=0, label_source=0, stage_index=0),
        ],
    )

    original_sampler = dataset.schedule_engine.sampler
    candidate_coords = [("vol1", 0, 5, 5), ("vol2", 0, 6, 6)]

    class _Sampler:
        def sample_batch(self, forbidden_coords, batch_size):
            return [coord for coord in candidate_coords if coord not in forbidden_coords][:batch_size]

    def evaluator(batch):
        return np.array([3, 1], dtype=np.int32), np.array([0.15, 0.25], dtype=np.float32)

    dataset.schedule_engine.sampler = _Sampler()
    accepted = dataset.schedule_engine.extend_schedule(
        stage_index=1,
        target_active_rows=2,
        evaluator=evaluator,
        evaluation_batch_size=8,
        target_label_source=2,
    )
    dataset.schedule_engine.sampler = original_sampler

    assert accepted == 2
    acquired_indices = dataset.schedule.get_active_label_source_indices(2)
    assert len(acquired_indices) == 2
    np.testing.assert_array_equal(dataset.schedule["label_source"][acquired_indices], np.array([2, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        dataset.schedule["current_label"][acquired_indices],
        dataset.schedule["gt_label"][acquired_indices],
    )
    np.testing.assert_allclose(dataset.schedule["last_confidence"][acquired_indices], np.array([0.15, 0.25], dtype=np.float32))


def test_active_learning_noop_maintenance_keeps_acquired_rows_enabled():
    dataset = make_runtime_dataset(
        TrainConfig(
            training_regime="active_learning",
            candidate_sampling={"name": "uniform_coordinate_sampling"},
            schedule_admission={"name": "admit_all_with_gt"},
            pseudolabel_maintenance={"name": "noop"},
        )
    )
    set_schedule(
        dataset,
        [
            make_row(0, name_id=0, coords=(0, 4, 4), current_label=0, gt_label=0, label_source=0, stage_index=0),
            make_row(1, name_id=1, coords=(0, 5, 5), current_label=1, gt_label=1, label_source=2, stage_index=1, confidence=0.7),
        ],
    )

    disabled = dataset.schedule_engine.reevaluate_schedule(
        next_stage_idx=2,
        evaluator=lambda batch: (np.array([3], dtype=np.int32), np.array([0.01], dtype=np.float32)),
        evaluation_batch_size=4,
        target_label_source=2,
    )

    assert disabled == 0
    acquired_idx = dataset.schedule.get_active_label_source_indices(2)
    assert len(acquired_idx) == 1
    assert bool(dataset.schedule["is_enabled"][acquired_idx[0]]) is True
