import torch


def flex_collate(batch):
    # batch = list of (patches[M,1,H,W], labels[M], segs[M,1,H,W])
    patches = torch.cat([b[0] for b in batch], dim=0)  # [sum M, 1, H, W]
    labels = torch.cat([b[1] for b in batch], dim=0)  # [sum M]
    segs = torch.cat([b[2] for b in batch], dim=0)  # [sum M, 1, H, W]
    coords = torch.stack([b[3] for b in batch], dim=0) if len(batch[0]) > 3 else None

    return patches, labels, segs, coords


def ordered_collate_fn(batch):
    """
    Batch structure:
        - First N: anchors
        - Next 3N: neighbors grouped by anchor
    """
    anchor_patches = []
    anchor_labels = []
    neighbor_patches = []
    neighbor_labels = []
    anchor_meta = []
    neighbor_meta = []

    # torch.stack(patches),  # shape [4, 1, 64, 64]
    # anchor["label"],
    # {
    #     "neighbor_labels": torch.tensor(neighbor_labels, dtype=torch.long),
    #     "anchor_meta": anchor,
    #     "neighbor_meta": neighbors,
    # }

    for sample in batch:
        anchor_patches.append(sample[0])  # shape [1, 64, 64]
        anchor_labels.append(sample[1])
        anchor_meta.append(sample[2]["anchor_meta"])

        # neighbors: patches[1:] = 3 unlabeled patches
        neighbor_patches.extend(sample["patches"][1:])  # 3 x [1, 64, 64]
        neighbor_labels.extend(sample["neighbor_labels"])  # list of 3
        neighbor_meta.extend(sample["neighbor_meta"])

    return {
        "patches": torch.cat(
            anchor_patches + neighbor_patches, dim=0
        ),  # [4n, 1, 64, 64]
        "labels": torch.tensor(anchor_labels, dtype=torch.long),  # [n]
        "neighbor_labels": torch.tensor(neighbor_labels, dtype=torch.long),  # [3n]
        "anchor_meta": anchor_meta,
        "neighbor_meta": neighbor_meta,
    }