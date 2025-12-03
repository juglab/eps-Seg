import random
import itertools
from torch.utils.data import BatchSampler, DistributedSampler
from eps_seg.dataloaders.datasets import SemisupervisedDataset
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

    def __init__(self, dataset, total_patches_per_batch=32, seed=42, shuffle=True):
        self.dataset = dataset
        self.total_patches_per_batch = total_patches_per_batch
        self.seed = seed
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
        dataset: SemisupervisedDataset,
        sampler: ModeAwareBalancedAnchorBatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = False,
        batches_per_pseudoepoch: Optional[int] = None,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        num_replicas, rank = self._fix_rank_replicas(num_replicas, rank)
        assert not shuffle, "DistributedParallelBatchSampler does not support shuffling directly, please shuffle the underlying sampler."
        self.sampler = sampler
        self.current_true_epoch = 0
        self.batches_per_pseudoepoch = batches_per_pseudoepoch
        if self.batches_per_pseudoepoch is not None:
            assert self.batches_per_pseudoepoch % self.num_replicas == 0, "batches_per_pseudoepoch must be divisible by num_replicas to ensure every GPU gets equal number of batches."
        
        # Counters to track position when using pseudo-epochs
        self.next_pe_idx = 0 # To track position in the underlying sampler across pseudo-epochs
        self.next_te_idx = 0 # To track position in the underlying sampler across true-epochs
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
            # This iteration goes over a pseudo-epoch
            while self.next_pe_idx < self.batches_per_pseudoepoch:
                if self.sampler_iter is None:
                    # This begins the first true epoch (and reshuffle data internally) if sampler.shuffle is True
                    self.sampler_iter = iter(self.sampler)
                try:
                    batch = next(self.sampler_iter)
                    # TODO: It would be better to resample so we don't discard last samples if they don't fit on num_replicas
                    should_yield = (self.next_te_idx % self.num_replicas == self.rank and self.next_pe_idx < n_batches_this_replica * self.num_replicas)
                    self.next_pe_idx += 1
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

            # Pseudo-epoch ended
            self.next_pe_idx = 0