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

    def __init__(self, 
                 dataset, 
                 total_patches_per_batch=32, 
                 seed=42, 
                 shuffle=True,
                 samples_per_pseudo_epoch=None):
        self.dataset = dataset
        self.total_patches_per_batch = total_patches_per_batch
        self.rng = random.Random(seed)
        self.shuffle = shuffle

        self.samples_per_pseudo_epoch = samples_per_pseudo_epoch
        self._pseudo_epoch_counter = 0
        self._samples_into_current_pseudo = 0
        self._samples_into_true_epoch = 0

        # Build per-class pools once (anchors only)
        # This assumes that dataset.groups contains groups that are formed by [anchor, neighbors...]
        self.pools = {
            c: [i for i, g in enumerate(dataset.groups) if g["labels"][0] == c]
            for c in dataset.unique_vals
        }
        self.labels = [c for c, v in self.pools.items() if len(v) > 0]
        if not self.labels:
            raise ValueError("No anchors available in any class.")

        # cycling iterators for oversampling
        self._iters = None
        # Caching len() result
        self._len_cached = None

    def _start_new_true_epoch(self):
        """
            When the dataset anchors are exhausted (a true epoch), reset all iterators.
        """
        self._samples_into_true_epoch = 0
        # Reshuffle class pools if shuffling is enabled
        self._reset_iters()

    def _reset_iters(self):
        """
            Reset cycling iterators for each class pool.
        """
        self._iters = {}
        for c in self.labels:
            pool = list(self.pools[c])
            if self.shuffle:
                self.rng.shuffle(pool)
            self._iters[c] = itertools.cycle(pool)

    def _compute_epoch_plan(self):
        """
            Compute how many anchors per batch, per class, and total number of batches for the current epoch.


            Returns:
                anchors_per_batch: int
                per_label_counts: dict[class_label] = count
                num_batches: int
        """

        # anchors-per-batch depends on current mode
        if self.dataset.mode == "semisupervised":
            assert (
                self.total_patches_per_batch % 8 == 0  # TODO
            ), "total_patches_per_batch must be divisible by 8 in semisupervised mode."
            # In semisupervised mode, each anchor has 7 neighbors?
            # Every 8 patches we have 1 anchor
            anchors_per_batch = self.total_patches_per_batch // 8  # TODO
        else:
            # In supervised mode, every sample is an anchor
            anchors_per_batch = self.total_patches_per_batch

        # split anchors-per-batch across labels (balanced, round-robin remainder)
        base = anchors_per_batch // len(self.labels)
        rem = anchors_per_batch % len(self.labels)
        per_label_counts = {c: base for c in self.labels}
        for c in self.labels[:rem]:
            per_label_counts[c] += 1

        # epoch length heuristic: sized to the largest class before a full cycle
        # 
        max_class = max(len(self.pools[c]) for c in self.labels)
        num_batches = max(1, (max_class * len(self.labels)) // anchors_per_batch)

        return anchors_per_batch, per_label_counts, num_batches

    def __iter__(self):
        # Reset at the start of each true epoch
        # TODO: Is it correct ???
        if self._samples_into_true_epoch == 0:
            self._start_new_true_epoch()
        

        # Dataset returns 

        # Start new pseudo-epoch 
        
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