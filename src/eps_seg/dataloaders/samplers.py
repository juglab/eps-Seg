import random
import itertools
from torch.utils.data import BatchSampler, DistributedSampler
from eps_seg.dataloaders.datasets import SemisupervisedDataset
from typing import Optional
from typing import Iterator

class ModeAwareBalancedAnchorBatchSampler(BatchSampler):
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
            # in semisupervised mode, dataset returns [anchor + 7 neighbors, ancor + 7 neighbors, ...]
            assert (
                self.total_patches_per_batch % 8 == 0  # TODO
            ), "total_patches_per_batch must be divisible by 8 in semisupervised mode."
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
    

class DistributedParallelSampler(DistributedSampler):
    """
        Wraps a ModeAwareBalancedAnchorBatchSampler to provide distributed sampling capabilities.
        Each replica will only return batches with indices batch_idx % num_replicas == rank to achieve distributed training.

        Args:
            dataset (SemisupervisedDataset): The dataset to sample from.
            sampler (ModeAwareBalancedAnchorBatchSampler): The sampler to wrap for distributed sampling.
            num_replicas (Optional[int]): Number of processes in distributed training.
            rank (Optional[int]): Rank of the current process.
            shuffle (bool): Whether to shuffle the dataset at the start of each epoch.
            seed (int): Random seed for shuffling.
            drop_last (bool): Whether to drop the last incomplete batch.

    """
    def __init__(
        self,
        dataset: SemisupervisedDataset,
        sampler: ModeAwareBalancedAnchorBatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        num_replicas, rank = self._fix_rank_replicas(num_replicas, rank)
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
    
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

    def set_epoch(self, epoch: int) -> None:
        """
           This is called at the beginning of each epoch to set the epoch number for shuffling.
        """
        super().set_epoch(epoch)
        print(f"Set epoch called on DistributedParallelSampler with epoch {epoch}")
        self.sampler.set_epoch(epoch)
    
    def __iter__(self) -> Iterator[int]:
        """
            Yield indices for the current replica by filtering the underlying sampler's indices.
        """
        for batch_idx, batch in enumerate(self.sampler):
            if batch_idx % self.num_replicas == self.rank:
                yield batch