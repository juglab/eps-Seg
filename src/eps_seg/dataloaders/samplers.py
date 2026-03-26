import random
import itertools
from torch.utils.data import BatchSampler, DistributedSampler
from typing import Optional
from typing import Iterator
import torch.distributed as dist

class ModeAwareBalancedAnchorBatchSampler(BatchSampler):
    """
        Yields balanced batches of anchor indices.
        Adapts to dataset.mode at the start of every epoch.
        - total_patches_per_batch is in *patch units* (e.g., 32).
        - Supervised: 1 patch per anchor
        - Semisupervised: 4 patches per anchor
    """

    def __init__(self, dataset, total_patches_per_batch=32, n_neighbors=7, seed=42, shuffle=True):
        self.dataset = dataset
        self.total_patches_per_batch = total_patches_per_batch
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        # Build per-class pools once (anchors only)
        self.pools = {
            c: [i for i, g in enumerate(dataset.groups) if g["labels"][0] == c]
            for c in dataset.unique_labels
        }
        self.labels = [c for c, v in self.pools.items() if len(v) > 0]
        if not self.labels:
            raise ValueError("No anchors available in any class.")

        # cycling iterators for oversampling
        self._iters = None
        self._len_cached = None

    def _reset_iters(self):
        """ 
            Resets the cycling iterators for each class pool. 
            Shuffles the pools if self.shuffle is True.
        """
        self._iters = {}
        for c in self.labels:
            pool = list(self.pools[c])
            if self.shuffle:
                self.rng.shuffle(pool)
            self._iters[c] = itertools.cycle(pool)

    def _compute_epoch_plan(self):
        # anchors-per-batch depends on current mode
        if self.dataset.mode == "semisupervised":
            # in semisupervised mode, dataset returns [anchor + 7 neighbors, ancor + 7 neighbors, ...]
            assert (
                self.total_patches_per_batch % (1 + self.n_neighbors) == 0  # TODO
            ), f"total_patches_per_batch must be divisible by {1 + self.n_neighbors} in semisupervised mode."
            anchors_per_batch = self.total_patches_per_batch // (1 + self.n_neighbors)  # TODO
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


class BalancedAnchorLabelBatchSampler(BatchSampler):
    """
        Yields class-balanced batches for PseudoLabelDataset-like schedules.

        The balancing label for every valid sample is derived from their respective anchors:
        - anchors are balanced by their own current label
        - neighbors are balanced by the current label of their anchor

        Pools are rebuilt at the start of every epoch from dataset.valid_indices so
        confidence-threshold updates, radius growth, and anchor elections are
        reflected immediately.
    """

    def __init__(self, dataset, batch_size=32, seed=42, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        self.labels = None
        self.pools = None
        self._iters = None

    def _anchor_labels_for_valid_indices(self):
        valid_indices = self.dataset.valid_indices
        if len(valid_indices) == 0:
            raise ValueError("No valid samples available in dataset.")

        anchor_indices = self.dataset.schedule["anchor_id"][valid_indices]
        anchor_labels = self.dataset.schedule["current_label"][anchor_indices]
        is_anchor = self.dataset.schedule["is_anchor"][valid_indices]
        current_labels = self.dataset.schedule["current_label"][valid_indices]

        # Prefer the sample's own label when it is a labeled anchor. This keeps the
        # sampler aligned with anchor elections once the schedule starts treating a
        # pseudo-label as a real anchor.
        return [
            int(current_label if is_anchor and current_label >= 0 else anchor_label)
            for current_label, anchor_label, is_anchor in zip(current_labels, anchor_labels, is_anchor)
        ]

    def _rebuild_pools(self):
        anchor_labels = self._anchor_labels_for_valid_indices()
        valid_indices = self.dataset.valid_indices

        pools = {}
        for dataset_idx, anchor_label in enumerate(anchor_labels):
            pools.setdefault(anchor_label, []).append(dataset_idx)

        self.pools = pools
        self.labels = [label for label, pool in self.pools.items() if len(pool) > 0]
        if not self.labels:
            raise ValueError("No valid samples available in any class.")

    def _reset_iters(self):
        self._iters = {}
        for label in self.labels:
            pool = list(self.pools[label])
            if self.shuffle:
                self.rng.shuffle(pool)
            self._iters[label] = itertools.cycle(pool)

    def _compute_epoch_plan(self):
        self._rebuild_pools()

        base = self.batch_size // len(self.labels)
        rem = self.batch_size % len(self.labels)
        per_label_counts = {label: base for label in self.labels}
        for label in self.labels[:rem]:
            per_label_counts[label] += 1

        max_class = max(len(self.pools[label]) for label in self.labels)
        num_batches = max(1, (max_class * len(self.labels)) // self.batch_size)
        return per_label_counts, num_batches

    def __iter__(self):
        per_label_counts, num_batches = self._compute_epoch_plan()
        self._reset_iters()

        label_order = list(self.labels)
        if self.shuffle:
            self.rng.shuffle(label_order)

        for _ in range(num_batches):
            batch = []
            for label in label_order:
                take = per_label_counts[label]
                batch.extend(next(self._iters[label]) for _ in range(take))
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        _, num_batches = self._compute_epoch_plan()
        return num_batches

class PseudoEpochDistributedParallelBatchSampler(DistributedSampler):
    """
        Wraps a ModeAwareBalancedAnchorBatchSampler to provide distributed sampling and pseudo-epoch capabilities.

        Psudo-epochs allow limiting the number of batches per epoch to a fixed number (batches_per_pseudoepoch),
        regardless of the total number of batches in the underlying sampler. 
        
        This sampler guarantees that
        1. Each replica gets an equal number of batches per pseudo-epoch.
        2. The underlying sampler's state is preserved across pseudo-epochs, so that the full dataset is eventually covered with enough pseudo-epochs.
        3. The sampler supports distributed training by ensuring each replica only returns batches with indices batch_idx % num_replicas == rank.

        Args:
            dataset (SemisupervisedDataset): The dataset to sample from.
            sampler (ModeAwareBalancedAnchorBatchSampler): The sampler to wrap for distributed sampling.
            num_replicas (Optional[int]): Number of processes in distributed training.
            rank (Optional[int]): Rank of the current process.
            shuffle (bool): Has no effect here, kept for compatibility.
            seed (int): Random seed for shuffling.
            drop_last (bool): Whether to drop the last incomplete batch.
            batches_per_pseudoepoch (Optional[int]): If provided, limits the number of batches per pseudo-epoch. If None, uses the full length of the underlying sampler.

    """
    def __init__(
        self,
        dataset,
        sampler: BatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = False,
        batches_per_pseudoepoch: Optional[int] = None,
    ) -> None:
        self.num_replicas, self.rank = self._fix_rank_replicas(num_replicas, rank)
        super().__init__(dataset, self.num_replicas, self.rank, False, seed, drop_last)
        assert not shuffle, "DistributedParallelBatchSampler does not support shuffling directly, please shuffle the underlying sampler."
        self.sampler = sampler
        self.current_true_epoch = 0
        self.batches_per_pseudoepoch = batches_per_pseudoepoch
        if self.batches_per_pseudoepoch is not None:
            assert self.batches_per_pseudoepoch % self.num_replicas == 0, "batches_per_pseudoepoch must be divisible by num_replicas to ensure every GPU gets equal number of batches."
        
        # Counter to track position in the underlying sampler across true-epochs
        self.next_te_idx = 0
        self.sampler_iter = None


    def _fix_rank_replicas(self, num_replicas: int, rank: int) -> tuple[int, int]:
        """
            Helper function to determine num_replicas and rank if they are not provided.
            Used mainly for debugging and non-distributed scenarios.
        """
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                # fallback to single-process behavior for local debugging / unit tests
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        return num_replicas, rank

    def __len__(self) -> int:
        """
            Returns the number of batches for this replica.
        """
        
        if self.batches_per_pseudoepoch is not None:
            # We are using pseudo-epochs, so the length of dataset in batch is actually batches_per_pseudoepoch.
            # If the dataset is smaller, it is resampled to match batches_per_pseudoepoch
            return self.batches_per_pseudoepoch // self.num_replicas
        else:
            # Dataset length is the real sampler length
            return len(self.sampler) // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """
            Yield indices for the current replica by filtering the underlying sampler's indices.
        """
        n_batches_this_replica = len(self) # e.g., 101 for 4 replicas with 404 total batches
        
        
        if self.batches_per_pseudoepoch is None:
            # Just return true epoch batches
            for batch_idx, batch in enumerate(self.sampler):
                # TODO: It would be better to resample so we don't discard last samples if they don't fit on num_replicas
                should_yield = batch_idx % self.num_replicas == self.rank and batch_idx < n_batches_this_replica * self.num_replicas
                if should_yield:
                    yield batch
        else:
            # This iteration goes over one pseudo-epoch.
            # Use a local pseudo-epoch index so each __iter__ call is self-contained.
            # Lightning may stop iteration after len(self) batches without exhausting the generator.
            pe_idx = 0
            while pe_idx < self.batches_per_pseudoepoch:
                if self.sampler_iter is None:
                    # This begins the first true epoch (and reshuffle data internally) if sampler.shuffle is True
                    self.sampler_iter = iter(self.sampler)
                try:
                    batch = next(self.sampler_iter)
                    # TODO: is the second condition ever violated?
                    should_yield = (pe_idx % self.num_replicas == self.rank and pe_idx < n_batches_this_replica * self.num_replicas)
                    pe_idx += 1
                    self.next_te_idx += 1
                    if should_yield:
                        yield batch

                except StopIteration:
                    # True epoch ended
                    # This begins the first true epoch (and reshuffle data internally) if sampler.shuffle is True
                    self.sampler_iter = iter(self.sampler)
                    self.next_te_idx = 0
                    self.current_true_epoch += 1
                    continue
