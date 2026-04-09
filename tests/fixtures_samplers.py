import numpy as np


class DummyDataset:
    """
    Minimal dataset carrying only a version counter used by sampler tests.
    """

    def __init__(self):
        self.sampling_version = 0


class DummyPseudoLabelDataset:
    """
    Minimal in-memory dataset exposing only the scheduler interface needed by
    the staged scheduler sampler.
    """

    def __init__(self, schedule):
        self.schedule = schedule
        self.sampling_version = 0

    def get_initial_label_mask(self):
        return self.schedule["label_source"] == 0


class VersionedBatchSampler:
    """
    Small in-memory batch sampler whose ordering depends on the dataset version.
    """

    def __init__(self, dataset, version_to_batches):
        self.dataset = dataset
        self.version_to_batches = {
            int(version): [list(batch) for batch in batches]
            for version, batches in version_to_batches.items()
        }
        self._cached_version = None

    def is_stale(self):
        return self._cached_version != self.dataset.sampling_version

    def _current_batches(self):
        if self.dataset.sampling_version not in self.version_to_batches:
            raise KeyError(f"No batch ordering configured for version {self.dataset.sampling_version}")
        return self.version_to_batches[self.dataset.sampling_version]

    def __iter__(self):
        self._cached_version = self.dataset.sampling_version
        return iter(self._current_batches())

    def __len__(self):
        return len(self._current_batches())


def build_dummy_schedule(stage_label_counts):
    """
    Create a volatile scheduler table with the desired stage/class composition.

    ``stage_label_counts`` maps ``stage_index -> {current_label -> count}``.
    Stage 0 rows are marked as initial GT labels, all later stages as pseudo-labels.
    """
    name_id = []
    coords = []
    current_label = []
    gt_label = []
    confidence = []
    label_source = []
    stage_index = []

    coord_counter = 0
    for stage, label_counts in sorted(stage_label_counts.items()):
        for label, count in sorted(label_counts.items()):
            for _ in range(count):
                name_id.append(coord_counter % 3)
                coords.append((coord_counter, coord_counter + 1, coord_counter + 2))
                current_label.append(label)
                gt_label.append(label)
                confidence.append(1.0 if stage == 0 else 0.9)
                label_source.append(0 if stage == 0 else 1)
                stage_index.append(stage)
                coord_counter += 1

    return {
        "name_id": np.asarray(name_id, dtype=np.int32),
        "coords": np.asarray(coords, dtype=np.int32),
        "current_label": np.asarray(current_label, dtype=np.int32),
        "gt_label": np.asarray(gt_label, dtype=np.int32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "label_source": np.asarray(label_source, dtype=np.int32),
        "stage_index": np.asarray(stage_index, dtype=np.int32),
    }


def batch_labels_and_stages(dataset, batch_indices):
    labels = dataset.schedule["current_label"][batch_indices]
    stages = dataset.schedule["stage_index"][batch_indices]
    label_sources = dataset.schedule["label_source"][batch_indices]
    return labels, stages, label_sources


def collect_pseudoepoch(wrapper):
    """
    Materialize one pseudo-epoch from the wrapper into a plain list.
    """
    return [batch for batch in iter(wrapper)]


def flatten_batch_ids(batches):
    """
    Convert batches like ``[[0], [1]]`` into ``[0, 1]`` for easy assertions.
    """
    return [batch[0] for batch in batches]
