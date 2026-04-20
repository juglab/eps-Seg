import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SemisupervisedDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        patch_size=64,
        label_size=1,
        mode="semisupervised",
        n_classes=4,
        ignore_lbl=-1,
        indices_dict=None,
        radius=5,
        dim=2,
        seed=42,
        n_neighbors=7,
        samples_per_class: Optional[Dict[int, int]] = None,
        anchor_records: Optional[List[Dict[str, int]]] = None,
    ):
        self.patch_size = patch_size
        self.label_size = label_size
        self.offset = self.patch_size // 2 - self.label_size
        self.images = images
        self.labels = labels
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        self.unique_vals = np.array(range(n_classes))
        self.mode = mode
        self.indices_dict = indices_dict or {}
        self.anchor_records = anchor_records or []
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.samples_per_class = samples_per_class or {}
        self.default_samples_per_class: int = 1
        self.dim = dim
        self.groups = self._prepare_metadata()
        self.n_label_per_class = {
            c: len([g for g in self.groups if g["labels"][0] == c])
            for c in range(self.n_classes)
        }
        self.anchor_indices_by_label = {
            c: [i for i, g in enumerate(self.groups) if g["labels"][0] == c]
            for c in range(self.n_classes)
        }
        self._report_dataset_summary()

    def set_mode(self, mode: str):
        """Switch between supervised and semisupervised modes."""
        if mode not in ("supervised", "semisupervised"):
            raise ValueError("stage must be 'supervised' or 'semisupervised'")
        self.mode = mode
        self.groups = self._prepare_metadata()

    def set_radius(self, radius: int):
        if self.radius != radius:
            self.radius = radius
            self.groups = self._modify_metadata()

    def increase_radius(self):
        """Increase the radius for neighbor sampling."""
        self.radius += 1
        self.groups = self._modify_metadata()

    def _is_valid_coord(
        self,
        name,
        z,
        y,
        x,
        Z,
        H,
        W,
        z_start: Optional[int] = None,
        z_stop: Optional[int] = None,
    ):
        """
        Check if the coordinate (z, y, x) is valid:
        - full patch is inside bounds
        - center voxel is not ignore_lbl
        - if a substack range is provided, the center remains inside it
        """
        valid = self.offset <= y < H - self.offset - 1 and self.offset <= x < W - self.offset - 1

        if self.dim == 3:
            valid = valid and (self.offset <= z < Z - self.offset - 1)

        if z_start is not None and z_stop is not None:
            valid = valid and (z_start <= z < z_stop)

        in_cell = self.labels[name][z, y, x] != self.ignore_lbl

        # TODO: Implement mask consistency
        #unique_label_area = self.labels[name][z, x : x + self.label_size, y : y + self.label_size]
        #consistent_mask = np.all(unique_label_area == unique_label_area[0, 0])

        return valid and in_cell

    def __len__(self):
        return len(self.groups)

    def patch_at(self, img_stack, z, y, x):
        if self.dim == 2:
            p = img_stack[
                z,
                y - self.offset : y + self.offset + 2,
                x - self.offset : x + self.offset + 2,
            ]
            return torch.from_numpy(p).unsqueeze(0)
        p = img_stack[
            z - self.offset : z + self.offset + 2,
            y - self.offset : y + self.offset + 2,
            x - self.offset : x + self.offset + 2,
        ]
        return torch.from_numpy(p).unsqueeze(0)

    def __getitem__(self, idx):
        g = self.groups[idx]
        name = g["name"]
        img_vol = self.images[name]
        lbl_vol = self.labels[name]

        if self.mode == "supervised":
            cz, cy, cx = map(int, g["coords"][0])
            patch = self.patch_at(img_vol, cz, cy, cx).unsqueeze(0)
            label = torch.tensor([int(g["labels"][0])], dtype=torch.long)
            segment = self.patch_at(lbl_vol, cz, cy, cx).unsqueeze(0)
            return patch, label, segment, torch.tensor(g["coords"][0])

        coords = torch.tensor([tuple(map(int, xyz)) for xyz in g["coords"]])
        patches = torch.stack([self.patch_at(img_vol, cz, cy, cx) for (cz, cy, cx) in coords])
        labels = torch.tensor(
            [g["labels"][0]] + [-1] * self.n_neighbors,
            dtype=torch.long,
        )
        segments = torch.stack([self.patch_at(lbl_vol, cz, cy, cx) for (cz, cy, cx) in coords])
        return patches, labels, segments, coords

    def _prepare_metadata(self) -> List[dict]:
        if self.anchor_records:
            return self._prepare_metadata_from_anchors()
        return self._prepare_metadata_from_indices()

    def _prepare_metadata_from_indices(self) -> List[dict]:
        """
        Build metadata by sampling anchors from slice indices.

        This path is kept for compatibility, but the canonical-anchor cache uses
        the anchor-based path below.
        """
        groups: List[dict] = []

        desc = "Preparing supervised anchors from slice indices"
        for name, z_list in tqdm(self.indices_dict.items(), desc=desc, leave=False):
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape

            used_coords = set()
            for cz in z_list:
                stack = lbl[cz]

                for c in range(self.n_classes):
                    for cy, cx in self._sample_coords_for_class(stack, c):
                        if not self._is_valid_coord(name, cz, cy, cx, Z, H, W):
                            continue
                        if (cz, cy, cx) in used_coords:
                            continue

                        used_coords.add((int(cz), cy, cx))
                        neighbors = self._sample_neighbors(
                            name=name,
                            cz=cz,
                            cy=cy,
                            cx=cx,
                            Z=Z,
                            H=H,
                            W=W,
                            used_coords=used_coords,
                            lbl=lbl,
                            k=self.n_neighbors,
                            max_tries=100,
                        )

                        if self.mode == "supervised" or len(neighbors) == self.n_neighbors:
                            groups.append(
                                self._make_group_record(
                                    name=name,
                                    cz=cz,
                                    cy=cy,
                                    cx=cx,
                                    c=c,
                                    neighbors=neighbors,
                                )
                            )

        self._report_class_counts(groups)
        return groups

    def _prepare_metadata_from_anchors(self) -> List[dict]:
        """
        Build metadata from canonical anchor records cached by the datamodule.
        """
        groups: List[dict] = []

        for record in tqdm(self.anchor_records, desc="Preparing cached anchor groups", leave=False):
            name = record["stack_name"]
            cz = int(record["z"])
            cy = int(record["y"])
            cx = int(record["x"])
            c = int(record["gt_label"])
            z_start = int(record["z_start"])
            z_stop = int(record["z_stop"])

            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape
            if not self._is_valid_coord(name, cz, cy, cx, Z, H, W, z_start=z_start, z_stop=z_stop):
                continue

            used_coords = {(cz, cy, cx)}
            neighbors: List[Dict[str, int]] = []
            if self.mode == "semisupervised":
                neighbors = self._sample_neighbors(
                    name=name,
                    cz=cz,
                    cy=cy,
                    cx=cx,
                    Z=Z,
                    H=H,
                    W=W,
                    used_coords=used_coords,
                    lbl=lbl,
                    z_start=z_start,
                    z_stop=z_stop,
                    k=self.n_neighbors,
                    max_tries=100,
                )
                if len(neighbors) != self.n_neighbors:
                    continue

            groups.append(
                self._make_group_record(
                    name=name,
                    cz=cz,
                    cy=cy,
                    cx=cx,
                    c=c,
                    neighbors=neighbors,
                    substack_id=int(record["substack_id"]),
                    z_start=z_start,
                    z_stop=z_stop,
                    coord_id=int(record["coord_id"]),
                )
            )

        self._report_class_counts(groups)
        return groups

    def _modify_metadata(self) -> List[dict]:
        """Recompute metadata after changing radius."""
        if self.anchor_records:
            return self._prepare_metadata_from_anchors()

        for g in self.groups:
            name = g["name"]
            z = int(g["z"])
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape

            used_coords = set()
            _, cy, cx = g["coords"][0]

            used_coords.add((int(z), cy, cx))
            neighbors = self._sample_neighbors(
                name=name,
                cz=z,
                cy=cy,
                cx=cx,
                Z=Z,
                H=H,
                W=W,
                used_coords=used_coords,
                lbl=lbl,
                k=self.n_neighbors,
                max_tries=100,
            )

            if len(neighbors) == self.n_neighbors:
                modified_group = self._make_group_record(
                    name=name,
                    cz=z,
                    cy=cy,
                    cx=cx,
                    c=g["labels"][0],
                    neighbors=neighbors,
                )
                g["coords"] = modified_group["coords"]
                g["labels"] = modified_group["labels"]

        self._report_class_counts(self.groups)
        return self.groups

    def _sample_coords_for_class(
        self, stack: np.ndarray, c: int
    ) -> Iterable[Tuple[int, int]]:
        """Return up to N (y, x) coordinates for class c from a 2D label slice."""
        n_needed = self.samples_per_class.get(c, self.default_samples_per_class)
        label_coords = np.argwhere(stack == c)
        if len(label_coords) == 0 or n_needed <= 0:
            return []

        n_take = min(len(label_coords), n_needed)
        idx = self.rng.sample(range(len(label_coords)), n_take)
        sampled = label_coords[idx]
        return [(int(y), int(x)) for (y, x) in sampled]

    def _sample_neighbors(
        self,
        name: str,
        cz: int,
        cy: int,
        cx: int,
        Z: int,
        H: int,
        W: int,
        used_coords: set,
        lbl: np.ndarray,
        z_start: Optional[int] = None,
        z_stop: Optional[int] = None,
        k: int = 3,
        max_tries: int = 100,
    ) -> List[Dict[str, int]]:
        """Randomly sample up to k valid nearby coordinates within a disk."""
        neighbors: List[Dict[str, int]] = []
        tries = 0

        while len(neighbors) < k and tries < max_tries:
            dz = self.rng.randint(-self.radius, self.radius) if self.dim == 3 else 0
            dy = self.rng.randint(-self.radius, self.radius)
            dx = self.rng.randint(-self.radius, self.radius)

            if dx * dx + dy * dy + dz * dz > self.radius * self.radius:
                tries += 1
                continue
            if dx == 0 and dy == 0 and dz == 0:
                tries += 1
                continue

            nz, ny, nx = cz + dz, cy + dy, cx + dx
            coord = (nz, ny, nx)
            if coord in used_coords:
                tries += 1
                continue

            if self._is_valid_coord(
                name,
                nz,
                ny,
                nx,
                Z,
                H,
                W,
                z_start=z_start,
                z_stop=z_stop,
            ):
                used_coords.add(coord)
                neighbors.append(
                    {
                        "z": int(nz),
                        "y": int(ny),
                        "x": int(nx),
                        "label": int(lbl[nz, ny, nx].item()),
                    }
                )

            tries += 1

        return neighbors

    def _make_group_record(
        self,
        name: str,
        cz: int,
        cy: int,
        cx: int,
        c: int,
        neighbors: List[Dict[str, int]],
        substack_id: Optional[int] = None,
        z_start: Optional[int] = None,
        z_stop: Optional[int] = None,
        coord_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create the output dict for one (center + neighbors) group."""
        return {
            "name": name,
            "z": int(cz),
            "coords": [(int(cz), int(cy), int(cx))] + [(n["z"], n["y"], n["x"]) for n in neighbors],
            "labels": [int(c)] + [n["label"] for n in neighbors],
            "substack_id": substack_id,
            "z_start": z_start,
            "z_stop": z_stop,
            "coord_id": coord_id,
        }

    def _report_class_counts(self, groups: List[dict]) -> None:
        """Print class counts for centers and neighbors in tabular form."""
        centers = [g["labels"][0] for g in groups]
        neighbors = [lab for g in groups for lab in g["labels"][1:]]
        print(self._format_class_balance_table(centers, neighbors))

    def _report_dataset_summary(self) -> None:
        anchor_mode = "cached anchors" if self.anchor_records else "sampled indices"
        print(
            f"Dataset ready | mode={self.mode} | dim={self.dim}D | groups={len(self.groups)} "
            f"| neighbors={self.n_neighbors} | source={anchor_mode}"
        )

    def _format_class_balance_table(self, centers: List[int], neighbors: List[int]) -> str:
        anchor_counts = Counter(centers)
        neighbor_counts = Counter(neighbors)
        all_labels = sorted(set(anchor_counts) | set(neighbor_counts) | set(range(self.n_classes)))
        anchor_total = sum(anchor_counts.values())
        neighbor_total = sum(neighbor_counts.values())

        lines = ["Class balance:", f"{'label':>7} {'anchors':>10} {'anchor_%':>10} {'neighbors':>10} {'neighbor_%':>10}"]
        for label in all_labels:
            anchor_count = anchor_counts.get(label, 0)
            neighbor_count = neighbor_counts.get(label, 0)
            anchor_pct = 100.0 * anchor_count / anchor_total if anchor_total else 0.0
            neighbor_pct = 100.0 * neighbor_count / neighbor_total if neighbor_total else 0.0
            lines.append(
                f"{label:>7} {anchor_count:>10} {anchor_pct:>9.2f}% {neighbor_count:>10} {neighbor_pct:>9.2f}%"
            )
        lines.append(
            f"{'total':>7} {anchor_total:>10} {100.0 if anchor_total else 0.0:>9.2f}% "
            f"{neighbor_total:>10} {100.0 if neighbor_total else 0.0:>9.2f}%"
        )
        return "\n".join(lines)


class PredictionDataset(Dataset):
    """
        Yields 2D or 3D patches whose center has label != -1 and full patch is inside bounds.
        It optionally mask the image to a specific z slice for faster testing.

        image: (C,Z,H,W) or (Z,1,H,W)

            Args:
                images: dict of np.ndarrays
                    Key: image name,
                    Value: Image volume of shape (C,Z,H,W) or (Z,H,W)
                labels: dict of np.ndarrays
                    Key: image name,
                    Value: Label volume of shape (Z,H,W) or (1,Z,H,W)
                keys: List[str] (optional)
                    List of image names to keep ordering (if None, use keys from images dict).
                masks: Dict[str, Union[slice, None]] (optional)
                    Key: image name,
                    Value: If slice, only consider this z slice for patch extraction (for faster testing).
                    Constraint on label > -1 still apply. If None, consider all slices.
                patch_size: int
                    Size of the extracted patches (assumed cubic or square).
                dim: int
                    2 or 3 for 2D or 3D patches.
                ignore_lbl: int
                    Label value to ignore when extracting patches.
    """

    def __init__(
        self,
        images,
        labels,
        keys: List[str] = None,
        masks: Dict[str, Union[slice, None]] = None,
        patch_size=64,
        dim=2,
        ignore_lbl=-1,
    ):
        self.dim = dim
        self.ignore_lbl = ignore_lbl
        self.images = images
        self.labels = labels
        self.masks = masks

        self.ps = int(patch_size)
        assert self.ps % 2 == 0, "Patch size must be even; center is (ps/2-1, ps/2-1)."
        self.half = self.ps // 2  # 32 for 64x64 -> center at (31,31)
       
        self.images = self._fix_images_shape(self.images)
        self.labels = self._fix_images_shape(self.labels)
        self.masks = self._fix_mask_shape(self.masks)

        self.centers = []

        for k in (self.images.keys() if keys is None else keys):
            centers = self._get_valid_centers(
                mask=self.masks.get(k, None) if self.masks is not None else None,
                label=self.labels[k],
            )
            for c in centers:
                self.centers.append((k, c[0], c[1], c[2]))

    def _get_valid_centers(self, mask, label):
        """
            Get the valid centers for each image based on the mask and label.
            A valid center is one where:
                - label != ignore_lbl
                - full patch is inside bounds
                - mask is True (if provided)
        """
        final_mask = label != self.ignore_lbl
        C, Z, H, W = label.shape
        final_mask[:, :, :, : self.half] = False
        final_mask[:, :, :, W - self.half :] = False
        final_mask[:, :, : self.half, :] = False
        final_mask[:, :, H - self.half :, :] = False
        if self.dim == 3 and Z > self.half:
            final_mask[:, : self.half, :, :] = False
            final_mask[:, Z - self.half :, :, :] = False

        if mask is not None:
            mask_array = np.zeros_like(final_mask, dtype=bool)
            mask_array[mask] = True
            final_mask = final_mask & mask_array

        _, zs, ys, xs = np.where(final_mask)
        centers = np.stack([zs, ys, xs], axis=1).astype(np.int32)
        return centers

    def _fix_images_shape(self, images: dict) -> dict:
        for key, image in images.items():
            if image.ndim == 3:
                images[key] = image[None, ...]
        return images

    def _fix_mask_shape(self, masks: dict) -> dict:
        if masks is not None:
            for key, mask in masks.items():
                if mask is not None and len(mask) == 3:
                    masks[key] = (slice(None),) + mask
        return masks

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        """
            Returns a tuple:
            patch, center_label, coordinates of center and segment for compatibility with SemisupervisedDataset.
        """
        key, z, y, x = self.centers[idx]

        if self.dim == 3:
            z0, z1 = z - self.half, z + self.half
        else:
            z0, z1 = z, z + 1

        y0, y1 = y - self.half, y + self.half
        x0, x1 = x - self.half, x + self.half
        patch = self.images[key][:, z0:z1, y0:y1, x0:x1]
        patch = torch.from_numpy(patch).float()

        segment = self.labels[key][:, z0:z1, y0:y1, x0:x1]
        segment = torch.from_numpy(segment).long()

        center_label = torch.tensor(self.labels[key][:, z, y, x]).long()
        # Drop channel dim to get [B, D, H, W] in the DataLoader batches.
        # TODO: For multichannel this will return different shapes!
        if self.dim == 2:
            patch = patch.squeeze(-3)  # Return [H, W] to get [B, 1, H, W] in DataLoader (?)
            segment = segment.squeeze(-3)

        coords = torch.stack([torch.tensor(z), torch.tensor(y), torch.tensor(x)])
        return patch, center_label, segment, coords, key
