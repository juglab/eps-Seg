import random
import itertools
from torch.utils.data import Sampler


class ModeAwareBalancedAnchorBatchSampler(Sampler):
    """
    Yields balanced batches of anchor indices.
    Adapts to dataset.mode at the start of every epoch.
    - total_patches_per_batch is in *patch units* (e.g., 32).
    - Supervised: 1 patch per anchor
    - Semisupervised: 4 patches per anchor
    """

    def __init__(self, dataset, total_patches_per_batch=32, seed=42, shuffle=True):
        self.dataset = dataset
        self.total_patches_per_batch = total_patches_per_batch
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        # Build per-class pools once (anchors only)
        self.pools = {
            c: [i for i, g in enumerate(dataset.groups) if g["labels"][0] == c]
            for c in dataset.unique_vals
        }
        self.labels = [c for c, v in self.pools.items() if len(v) > 0]
        if not self.labels:
            raise ValueError("No anchors available in any class.")

        # cycling iterators for oversampling
        self._iters = None
        self._len_cached = None

    def _reset_iters(self):
        self._iters = {}
        for c in self.labels:
            pool = list(self.pools[c])
            if self.shuffle:
                self.rng.shuffle(pool)
            self._iters[c] = itertools.cycle(pool)

    def _compute_epoch_plan(self):
        # anchors-per-batch depends on current mode
        if self.dataset.mode == "semisupervised":
            assert (
                self.total_patches_per_batch % 8 == 0  # TODO
            ), "total_patches_per_batch must be divisible by 4 in semisupervised mode."
            anchors_per_batch = self.total_patches_per_batch // 8  # TODO
        else:
            anchors_per_batch = self.total_patches_per_batch

        # split anchors-per-batch across labels (balanced, round-robin remainder)
        base = anchors_per_batch // len(self.labels)
        rem = anchors_per_batch % len(self.labels)
        per_label_counts = {c: base for c in self.labels}
        for c in self.labels[:rem]:
            per_label_counts[c] += 1

        # epoch length heuristic: sized to the largest class before a full cycle
        max_class = max(len(self.pools[c]) for c in self.labels)
        num_batches = max(1, (max_class * len(self.labels)) // anchors_per_batch)

        return anchors_per_batch, per_label_counts, num_batches

    def __iter__(self):
        self._reset_iters()
        anchors_per_batch, per_label_counts, num_batches = self._compute_epoch_plan()
        self._len_cached = num_batches

        label_order = list(self.labels)
        if self.shuffle:
            self.rng.shuffle(label_order)

        for _ in range(num_batches):
            batch = []
            for c in label_order:
                take = per_label_counts[c]
                batch.extend(next(self._iters[c]) for _ in range(take))
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        # compute against current mode so progress bars don't go crazy after mode flip
        anchors_per_batch, _, num_batches = self._compute_epoch_plan()
        return num_batches


class BalancedBatchSampler(Sampler):
    """
    A custom sampler that generates balanced batches from a dataset by ensuring each batch
    contains a balanced number of samples from each label class.

    This sampler is useful when training models with imbalanced datasets, as it helps to
    maintain an equal representation of each class within each batch. The class ensures
    that samples from each label are included in the batch proportionally and handles
    scenarios where the number of samples for each label differs significantly.

    Attributes:
    -----------
    dataset : Dataset
        The dataset from which samples are drawn. The dataset should have a `patches_by_label`
        attribute, which is a dictionary mapping labels to indices of samples belonging to
        those labels.

    batch_size : int
        The total number of samples in each batch.

    label_to_indices : dict
        A dictionary that maps each label to a list of indices of samples that belong
        to that label.

    num_labels : int
        The number of unique labels in the dataset.

    samples_per_label : int
        The number of samples to include from each label in each batch.

    remaining_samples : int
        The number of extra samples to distribute across labels to fill the batch.

    max_batch : int
        The maximum number of batches that can be generated based on the size of the
        largest class and the number of samples per label.

    Methods:
    --------
    __init__(dataset, batch_size)
        Initializes the sampler with the dataset and batch size.

    __iter__()
        Returns an iterator that yields balanced batches of indices.

    __len__()
        Estimates the total number of batches that can be generated.
       pass
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # dictionary mapping labels to indices
        self.label_to_indices = dataset.patches_by_label
        for key in self.label_to_indices:
            random.shuffle(self.label_to_indices[key])

        # Determine number of labels
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = self.batch_size // self.num_labels
        self.remaining_samples = self.batch_size % self.num_labels
        self.max_batch = (
            max(len(indices) for indices in self.label_to_indices.values())
            // self.samples_per_label
        )

    def __iter__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        num_batches_generated = 0
        # Generate balanced batches
        while num_batches_generated < self.max_batch:
            batch = []
            for label, indices in self.label_to_indices.items():
                if len(indices) < self.samples_per_label:
                    indices = random.choices(indices, k=max_class_size)  # Oversample
                selected_indices = random.sample(indices, self.samples_per_label)
                batch.extend(selected_indices)

            if len(batch) < self.batch_size:
                # Handle any remaining spots in the batch
                remaining_indices = []
                for indices in self.label_to_indices.values():
                    remaining_indices.extend(indices)
                random.shuffle(remaining_indices)
                batch.extend(remaining_indices[: self.batch_size - len(batch)])

            if len(batch) == self.batch_size:
                random.shuffle(batch)
                num_batches_generated += 1
                yield batch
            # Return the batch and pause execution until the next batch is requested

    def __len__(self):
        # Estimate the length based on the largest class
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        return (max_class_size * self.num_labels) // self.batch_size


class CombinedBatchSampler(Sampler):
    """
    A custom sampler that generates batches containing 50% balanced labeled samples
    and 50% random samples. Inherits from BalancedBatchSampler.
    """

    def __init__(self, dataset, batch_size, labeled_ratio=0.50):
        """
        Initializes the CombinedBatchSampler.

        Parameters:
        -----------
        dataset : Dataset
            The dataset from which samples are drawn. Should have a `patches_by_label`
            attribute for labeled patches.
        batch_size : int
            The total number of samples in each batch.

        """
        self.label_to_indices = dataset.patches_by_label
        self.random_indices = range(int(len(dataset) * labeled_ratio), len(dataset))
        for key in self.label_to_indices:
            random.shuffle(self.label_to_indices[key])
        self.batch_size = batch_size
        self.small_batch_size = int(batch_size * labeled_ratio)
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = self.small_batch_size // self.num_labels
        self.remaining_samples = self.small_batch_size % self.num_labels
        self.max_batch = (
            max(len(indices) for indices in self.label_to_indices.values())
            // self.samples_per_label
        )

    def __iter__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        num_batches_generated = 0

        while num_batches_generated < self.max_batch:
            # Step 1: Sample 25% of the batch using balanced sampling from labeled indices
            balanced_batch = []
            for label, indices in self.label_to_indices.items():
                if len(indices) < self.samples_per_label:
                    indices = random.choices(indices, k=max_class_size)
                selected_indices = random.sample(indices, self.samples_per_label)
                balanced_batch.extend(selected_indices)

            # Fill up if the balanced batch is not full (due to class imbalance or fewer labeled samples)
            while len(balanced_batch) < self.small_batch_size:
                remaining_labeled = []
                for indices in self.label_to_indices.values():
                    remaining_labeled.extend(indices)
                random.shuffle(remaining_labeled)
                balanced_batch.extend(
                    remaining_labeled[: self.small_batch_size - len(balanced_batch)]
                )

            # Step 2: Sample 75% of the batch randomly from unlabeled indices
            random_unlabeled = random.sample(
                self.random_indices, self.batch_size - self.small_batch_size
            )

            # Combine balanced labeled and random unlabeled samples
            combined_batch = balanced_batch + random_unlabeled

            # Ensure the final batch size is correct
            if len(combined_batch) == self.batch_size:
                num_batches_generated += 1
                yield combined_batch

    def __len__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        return (max_class_size * self.num_labels) // self.small_batch_size


class UnsupervisedSampler(Sampler):
    """
    A custom sampler that generates batches containing random samples from the dataset.
    """

    def __init__(self, dataset, batch_size):
        """
        Initializes the UnsupervisedSampler.

        Parameters:
        -----------
        dataset : Dataset
            The dataset from which samples are drawn.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_batch = len(dataset) // batch_size

    def __iter__(self):
        indices = range(len(self.dataset))
        num_batches_generated = 0
        while num_batches_generated < self.max_batch:
            batch = random.sample(indices, self.batch_size)
            num_batches_generated += 1
            yield batch

    def __len__(self):
        return self.max_batch


class DynamicSampler(Sampler):
    def __init__(self, dataset, batch_size, labeled_ratio=0.25):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio

    def __iter__(self):
        if self.dataset.mode == "supervised":
            sampler = BalancedBatchSampler(self.dataset, self.batch_size)
        elif self.dataset.mode == "semisupervised":
            sampler = CombinedBatchSampler(
                self.dataset, self.batch_size, labeled_ratio=self.labeled_ratio
            )
        elif self.dataset.mode == "unsupervised":
            sampler = UnsupervisedSampler(self.dataset, self.batch_size)
        yield from iter(sampler)

    def __len__(self):
        return len(self.dataset) // self.batch_size


