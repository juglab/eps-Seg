import random
import itertools
import numpy as np
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


class BalancedScheduledBatchSampler(BatchSampler):
    """
        Yields class-balanced batches for the staged scheduler dataset.

        Every batch satisfies two constraints:
        1. It is class balanced across the labels currently present in the scheduler.
        2. Its stage composition follows the relative distribution of the scheduler's
           ``stage_index`` values, while still enforcing a configurable minimum
           fraction of samples from the initial GT-labelled stage-0 pool.
    """

    def __init__(self, dataset, batch_size=32, min_initial_label_fraction=0.5, seed=42, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_initial_label_fraction = min_initial_label_fraction
        self.seed = seed
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        self.labels = None
        self.full_pools = None
        self.stage_label_pools = None
        self.stage_counts = None
        self.initial_stage_counts = None
        self.stages = None
        self._cached_version = None
        self._num_batches = None
        self._stage_targets = None
        self._label_targets = None
        self._stage_label_targets = None

    def _dataset_version(self):
        """Return the dataset version used to detect schedule changes."""
        return getattr(self.dataset, "sampling_version", 0)

    def is_stale(self):
        """Return whether the cached pools are out of date (i.e., Dataset composition has changed)."""
        return self._cached_version != self._dataset_version()

    def _build_pools(self, schedule_indices):
        """Group scheduler indices by their current label."""
        pools = {}
        active_schedule_indices = self.dataset.get_active_schedule_indices()
        for active_idx in schedule_indices:
            schedule_idx = int(active_schedule_indices[active_idx])
            label = int(self.dataset.schedule["current_label"][schedule_idx])
            pools.setdefault(label, []).append(int(active_idx))
        return pools

    def _build_stage_label_pools(self, schedule_indices):
        """Group scheduler indices first by stage and then by current label."""
        pools = {}
        active_schedule_indices = self.dataset.get_active_schedule_indices()
        for active_idx in schedule_indices:
            schedule_idx = int(active_schedule_indices[active_idx])
            stage = int(self.dataset.schedule["stage_index"][schedule_idx])
            label = int(self.dataset.schedule["current_label"][schedule_idx])
            pools.setdefault(stage, {}).setdefault(label, []).append(int(active_idx))
        return pools

    def _rebuild_pools(self):
        """Rebuild all cached pools from the current scheduler state."""
        active_schedule_indices = self.dataset.get_active_schedule_indices()
        schedule_indices = np.arange(len(active_schedule_indices))
        initial_mask = self.dataset.get_initial_label_mask()[active_schedule_indices]

        self.full_pools = self._build_pools(schedule_indices.tolist())
        self.stage_label_pools = self._build_stage_label_pools(schedule_indices.tolist())
        self.stage_counts = {
            int(stage): int(sum(len(pool) for pool in label_pools.values()))
            for stage, label_pools in self.stage_label_pools.items()
        }
        initial_stage_indices = np.where(initial_mask)[0].tolist()
        initial_stage_pools = self._build_stage_label_pools(initial_stage_indices)
        self.initial_stage_counts = {
            int(stage): int(sum(len(pool) for pool in label_pools.values()))
            for stage, label_pools in initial_stage_pools.items()
        }
        self.labels = sorted(label for label, pool in self.full_pools.items() if len(pool) > 0)
        self.stages = sorted(stage for stage, count in self.stage_counts.items() if count > 0)
        if not self.labels:
            raise ValueError("No valid samples available in any class.")
        if not any(count > 0 for count in self.initial_stage_counts.values()):
            raise ValueError("No initial GT-labelled samples available in the scheduler.")
        self._cached_version = self._dataset_version()

    def _reset_iters(self, pools):
        """Create cycling iterators for every pool, shuffling if requested."""
        iters = {}
        for key, pool_values in pools.items():
            if isinstance(pool_values, dict):
                nested_iters = self._reset_iters(pool_values)
                if nested_iters:
                    iters[key] = nested_iters
                continue
            if len(pool_values) == 0:
                continue
            pool = list(pool_values)
            if self.shuffle:
                self.rng.shuffle(pool)
            iters[key] = itertools.cycle(pool)
        return iters

    def _balanced_counts(self, total_items, keys):
        """Split a number of items as evenly as possible across the given keys."""
        keys = list(keys)
        if len(keys) == 0:
            return {}
        base = total_items // len(keys)
        rem = total_items % len(keys)
        counts = {key: base for key in keys}
        for key in keys[:rem]:
            counts[key] += 1
        return counts

    def _weighted_counts(self, total_items, weights):
        """Split a number of items according to the given weights."""
        keys = list(weights.keys())
        if total_items <= 0 or len(keys) == 0:
            return {key: 0 for key in keys}

        total_weight = float(sum(max(float(weights[key]), 0.0) for key in keys))
        if total_weight <= 0.0:
            return self._balanced_counts(total_items, keys)

        raw_counts = {key: total_items * max(float(weights[key]), 0.0) / total_weight for key in keys}
        counts = {key: int(np.floor(raw_counts[key])) for key in keys}
        remainder = total_items - sum(counts.values())
        if remainder > 0:
            ranked_keys = sorted(
                keys,
                key=lambda key: (raw_counts[key] - counts[key], -keys.index(key)),
                reverse=True,
            )
            for key in ranked_keys[:remainder]:
                counts[key] += 1
        return counts

    def _remove_from_counts(self, counts, total_to_remove, weights):
        """Remove items from a count dictionary while following the given weights."""
        remaining = int(total_to_remove)
        if remaining <= 0:
            return counts

        while remaining > 0:
            active_weights = {
                key: weights.get(key, 0.0)
                for key, count in counts.items()
                if count > 0
            }
            if not active_weights:
                raise ValueError("Cannot remove samples from stage targets without making them negative.")

            proposal = self._weighted_counts(remaining, active_weights)
            removed_this_round = 0
            for key, remove_count in proposal.items():
                if remove_count <= 0 or counts[key] <= 0:
                    continue
                actual_remove = min(int(remove_count), int(counts[key]))
                counts[key] -= actual_remove
                removed_this_round += actual_remove
            if removed_this_round == 0:
                fallback_key = max(active_weights.keys(), key=lambda key: (active_weights[key], counts[key], -key))
                counts[fallback_key] -= 1
                removed_this_round = 1
            remaining -= removed_this_round
        return counts

    def _compute_stage_targets(self):
        """Decide how many batch items should come from each scheduler stage."""
        stage_targets = self._weighted_counts(self.batch_size, self.stage_counts)

        # Enforce the minimum number of samples for the initial GT-labelled stages in a batch
        n_initial_min = int(np.ceil(self.batch_size * self.min_initial_label_fraction))
        n_initial_min = min(n_initial_min, self.batch_size)
        current_initial_target = sum(stage_targets.get(stage, 0) for stage in self.initial_stage_counts)

        # If the current distribution already meets the minimum requirement for initial stages, return it as is
        if current_initial_target >= n_initial_min:
            return stage_targets

        deficit = n_initial_min - current_initial_target

        # Add missing initial samples to meet the minimum requirement, distributing them according to the initial stage counts
        initial_additions = self._weighted_counts(deficit, self.initial_stage_counts)
        for stage, added_count in initial_additions.items():
            stage_targets[stage] = stage_targets.get(stage, 0) + int(added_count)

        # Remove the same number of samples from the non-initial stages to maintain the batch size
        non_initial_weights = {
            stage: count
            for stage, count in self.stage_counts.items()
            if stage not in self.initial_stage_counts
        }
        stage_targets = self._remove_from_counts(stage_targets, deficit, non_initial_weights)
        return stage_targets

    def _stage_available_labels(self):
        """List which labels are available inside each scheduler stage."""
        return {
            stage: sorted(label for label, pool in self.stage_label_pools[stage].items() if len(pool) > 0)
            for stage in self.stages
        }

    def _choose_label_for_stage(self, stage, available_labels, remaining_label_targets, stage_label_targets):
        """Pick the next label to draw for one stage while keeping class balance."""
        candidate_labels = [label for label in available_labels if remaining_label_targets[label] > 0]
        if not candidate_labels:
            candidate_labels = list(available_labels)
        return max(
            candidate_labels,
            key=lambda label: (
                remaining_label_targets[label],
                -stage_label_targets[stage][label],
                -self.labels.index(label),
            ),
        )

    def _repair_label_totals(self, stage_label_targets, label_targets, stage_available_labels):
        """Fix small count mismatches so global class totals match the batch plan."""
        label_totals = {
            label: sum(stage_label_targets[stage][label] for stage in self.stages)
            for label in self.labels
        }

        while True:
            deficit_labels = [label for label in self.labels if label_totals[label] < label_targets[label]]
            excess_labels = [label for label in self.labels if label_totals[label] > label_targets[label]]
            if not deficit_labels and not excess_labels:
                return stage_label_targets

            repaired = False
            for deficit_label in deficit_labels:
                for excess_label in excess_labels:
                    candidate_stages = [
                        stage
                        for stage in self.stages
                        if stage_label_targets[stage][excess_label] > 0 and deficit_label in stage_available_labels[stage]
                    ]
                    if not candidate_stages:
                        continue
                    chosen_stage = min(
                        candidate_stages,
                        key=lambda stage: (stage_label_targets[stage][deficit_label], stage),
                    )
                    stage_label_targets[chosen_stage][excess_label] -= 1
                    stage_label_targets[chosen_stage][deficit_label] += 1
                    label_totals[excess_label] -= 1
                    label_totals[deficit_label] += 1
                    repaired = True
                    break
                if repaired:
                    break

            if not repaired:
                raise ValueError(
                    "Unable to build a batch plan that is both class balanced and compatible "
                    "with the stage distribution of the current scheduler."
                )

    def _compute_stage_label_targets(self, stage_targets, label_targets):
        """Split each stage target into per-label targets for the current batch."""
        stage_available_labels = self._stage_available_labels()
        stage_label_targets = {
            stage: {label: 0 for label in self.labels}
            for stage in self.stages
        }
        remaining_label_targets = dict(label_targets)
        stage_order = sorted(self.stages, key=lambda stage: (len(stage_available_labels[stage]), stage))

        for stage in stage_order:
            for _ in range(stage_targets[stage]):
                chosen_label = self._choose_label_for_stage(
                    stage=stage,
                    available_labels=stage_available_labels[stage],
                    remaining_label_targets=remaining_label_targets,
                    stage_label_targets=stage_label_targets,
                )
                stage_label_targets[stage][chosen_label] += 1
                if remaining_label_targets[chosen_label] > 0:
                    remaining_label_targets[chosen_label] -= 1

        self._repair_label_totals(stage_label_targets, label_targets, stage_available_labels)
        return stage_label_targets

    def _compute_epoch_plan(self):
        """Build the cached batch plan and choose the number of batches per epoch."""
        self._rebuild_pools()
        max_class = max(len(self.full_pools[label]) for label in self.labels)
        num_batches = max(1, (max_class * len(self.labels)) // self.batch_size)
        # Compute the targets according to the stages
        stage_targets = self._compute_stage_targets()
        # Compute the label targets (balanced across all labels)
        label_targets = self._balanced_counts(self.batch_size, self.labels)
        # Compute the stage-label targets that satisfy both constraints
        stage_label_targets = self._compute_stage_label_targets(stage_targets, label_targets)
        self._stage_targets = stage_targets
        self._label_targets = label_targets
        self._stage_label_targets = stage_label_targets
        self._num_batches = num_batches
        return stage_label_targets, num_batches

    def __iter__(self):
        """Yield batches that follow the current class and stage plan."""
        stage_label_targets, num_batches = self._compute_epoch_plan()
        stage_label_iters = self._reset_iters(self.stage_label_pools)

        for _ in range(num_batches):
            batch = []
            stage_order = list(self.stages)
            label_order = list(self.labels)
            if self.shuffle:
                self.rng.shuffle(stage_order)
                self.rng.shuffle(label_order)
            for stage in stage_order:
                for label in label_order:
                    take = stage_label_targets[stage][label]
                    if take <= 0:
                        continue
                    batch.extend(next(stage_label_iters[stage][label]) for _ in range(take))
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        """Return the number of batches in the current epoch plan."""
        if self.is_stale() or self._num_batches is None:
            self._compute_epoch_plan()
        return self._num_batches


BalancedAnchorLabelBatchSampler = BalancedScheduledBatchSampler


class ClassBalancedScheduledBatchSampler(BatchSampler):
    """
        Yields class-balanced batches from all active scheduler rows.

        Unlike ``BalancedScheduledBatchSampler``, this sampler does not stratify
        by ``stage_index`` and does not enforce a minimum contribution from the
        initial labelled pool. It is intended as an active-learning control where
        the training set is treated as one active labelled pool.
    """

    def __init__(self, dataset, batch_size=32, seed=42, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        self.labels = None
        self.full_pools = None
        self._cached_version = None
        self._num_batches = None
        self._label_targets = None

    def _dataset_version(self):
        """Return the dataset version used to detect schedule changes."""
        return getattr(self.dataset, "sampling_version", 0)

    def is_stale(self):
        """Return whether the cached pools are out of date."""
        return self._cached_version != self._dataset_version()

    def _build_pools(self, active_indices):
        """Group active dataset indices by their current scheduler label."""
        pools = {}
        active_schedule_indices = self.dataset.get_active_schedule_indices()
        for active_idx in active_indices:
            schedule_idx = int(active_schedule_indices[active_idx])
            label = int(self.dataset.schedule["current_label"][schedule_idx])
            pools.setdefault(label, []).append(int(active_idx))
        return pools

    def _rebuild_pools(self):
        """Rebuild class pools from the currently active scheduler rows."""
        active_schedule_indices = self.dataset.get_active_schedule_indices()
        active_indices = np.arange(len(active_schedule_indices))
        self.full_pools = self._build_pools(active_indices.tolist())
        self.labels = sorted(label for label, pool in self.full_pools.items() if len(pool) > 0)
        if not self.labels:
            raise ValueError("No valid samples available in any class.")
        self._cached_version = self._dataset_version()

    def _reset_iters(self):
        """Create cycling iterators for every class pool, shuffling if requested."""
        iters = {}
        for label, pool_values in self.full_pools.items():
            if len(pool_values) == 0:
                continue
            pool = list(pool_values)
            if self.shuffle:
                self.rng.shuffle(pool)
            iters[label] = itertools.cycle(pool)
        return iters

    def _balanced_counts(self, total_items, keys):
        """Split a number of items as evenly as possible across the given keys."""
        keys = list(keys)
        if len(keys) == 0:
            return {}
        base = total_items // len(keys)
        rem = total_items % len(keys)
        counts = {key: base for key in keys}
        for key in keys[:rem]:
            counts[key] += 1
        return counts

    def _compute_epoch_plan(self):
        """Build the cached batch plan and choose the number of batches per epoch."""
        self._rebuild_pools()
        max_class = max(len(self.full_pools[label]) for label in self.labels)
        num_batches = max(1, (max_class * len(self.labels)) // self.batch_size)
        label_targets = self._balanced_counts(self.batch_size, self.labels)
        self._label_targets = label_targets
        self._num_batches = num_batches
        return label_targets, num_batches

    def __iter__(self):
        """Yield class-balanced batches from the active schedule."""
        label_targets, num_batches = self._compute_epoch_plan()
        label_iters = self._reset_iters()

        for _ in range(num_batches):
            batch = []
            label_order = list(self.labels)
            if self.shuffle:
                self.rng.shuffle(label_order)
            for label in label_order:
                take = label_targets[label]
                if take <= 0:
                    continue
                batch.extend(next(label_iters[label]) for _ in range(take))
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        """Return the number of batches in the current epoch plan."""
        if self.is_stale() or self._num_batches is None:
            self._compute_epoch_plan()
        return self._num_batches


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

        if hasattr(self.sampler, "is_stale") and self.sampler.is_stale():
            had_active_iterator = self.sampler_iter is not None
            self.sampler_iter = None
            self.next_te_idx = 0
            if self.batches_per_pseudoepoch is not None and had_active_iterator:
                self.current_true_epoch += 1
        
        
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
