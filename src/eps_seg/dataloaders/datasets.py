import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Any

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
        seed = 42,
        n_neighbors=7,
    ):
        self.patch_size = patch_size
        self.label_size = label_size
        self.half = self.patch_size // 2 - self.label_size
        self.images = images
        self.labels = labels
        self.ignore_lbl = ignore_lbl
        self.n_classes = n_classes
        # Convert back to sorted numpy array if needed
        self.unique_vals = np.array(range(n_classes))
        self.mode = mode
        self.indices_dict = indices_dict or {}
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.samples_per_class: Dict[int, int] = {1: 2} # for class 1 (nucleus), we take 2 to compensate for rarity along z
        self.default_samples_per_class: int = 1  # We take 1 sample per class per slice
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

    def set_mode(self, mode: str):
        """Switch between supervised and semisupervised modes."""
        if mode not in ("supervised", "semisupervised"):
            raise ValueError("stage must be 'supervised' or 'semisupervised'")
        self.mode = mode

    def increase_radius(self):
        """Increase the radius for neighbor sampling."""
        self.radius += 1
        self.groups = self._modify_metadata()

    def _is_valid_coord(self, name, z, y, x, Z, H, W):
        valid = (
            self.half <= y < H - self.half - 1 and self.half <= x < W - self.half - 1
        )
        if self.dim == 3:
            if self.half <= z < Z - self.half - 1:
                valid = valid and True

        in_cell = self.labels[name][z, y, x] != self.ignore_lbl
        return valid and in_cell

    def __len__(self):
        return len(self.groups)

    def patch_at(self, img_stack, z, y, x):
        if self.dim == 2:
            p = img_stack[
                z,
                y - self.half : y + self.half + 2,
                x - self.half : x + self.half + 2,
            ]
            return torch.from_numpy(p).unsqueeze(0)  # [1, H, W]
        else:  # 3D
            p = img_stack[
                z - self.half : z + self.half + 2,
                y - self.half : y + self.half + 2,
                x - self.half : x + self.half + 2,
            ]
            return torch.from_numpy(p).unsqueeze(0)  # [1, Z, H, W]

    def __getitem__(self, idx):
        g = self.groups[idx]
        name, z = g["name"], int(g["z"])
        img_vol = self.images[name]
        lbl_vol = self.labels[name]

        if self.mode == "supervised":
            cy, cx = map(int, g["coords"][0])
            patch = self.patch_at(img_vol, z, cy, cx).unsqueeze(0)  # [1, Z, H, W]  <-- extra dim
            label = torch.tensor([int(g["labels"][0])], dtype=torch.long)  # [1]
            segment = self.patch_at(lbl_vol, z, cy, cx).unsqueeze(0)  # [1, Z, H, W]
            return patch, label, segment, torch.tensor(g["coords"][0])
        else:
            coords = torch.tensor([tuple(map(int, xy)) for xy in g["coords"]])
            patches = torch.stack([self.patch_at(img_vol, z, y, x) for (y, x) in coords])  # [4, 1, Z, H, W]
            labels = torch.tensor([g["labels"][0]] + [-1]*7, dtype=torch.long)  # [4]
            segments = torch.stack([self.patch_at(lbl_vol, z, y, x) for (y, x) in coords])  # [4, 1, Z, H, W]
            return patches, labels, segments, coords

    def _prepare_metadata(self) -> List[dict]:
        groups: List[dict] = []

        for name, z_list in self.indices_dict.items():
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape

            for cz in z_list:
                used_coords = set()
                stack = lbl[cz]

                for c in range(self.n_classes):
                    for cy, cx in self._sample_coords_for_class(stack, c):
                        if not self._is_valid_coord(name, cz, cy, cx, Z, H, W):
                            continue
                        if (cy, cx) in used_coords:
                            continue
                        
                        used_coords.add((cy, cx))
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
                            k=self.n_neighbors,  # Number of neighbors to sample
                            max_tries=100,
                        )

                        if len(neighbors) == self.n_neighbors:
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

    def _modify_metadata(self) -> List[dict]:
        """Recompute metadata after changing radius."""

        for g in self.groups:
            name, z = g["name"], int(g["z"])
            img = self.images[name]
            lbl = self.labels[name]
            Z, H, W = img.shape

            used_coords = set()
            cy, cx = g["coords"][0]

            used_coords.add((cy, cx))
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
                    z=z,
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
        """Return up to N (y, x) coordinates for class c from a 2D label stack."""

        n_needed = getattr(self, "samples_per_class", {}).get(
            c,
            getattr(
                self,
                "default_samples_per_class",
            ),  # TODO
        )

        label_coords = np.argwhere(stack == c)
        if len(label_coords) < n_needed:
            return []  # not enough to sample

        idx = self.rng.sample(range(len(label_coords)), n_needed)
        sampled = label_coords[idx]
        return [(int(y), int(x)) for (y, x) in sampled]

    def _sample_neighbors(
        self,
        name: str,
        cz: int,
        cy: int,
        cx: int,
        Z:int,
        H: int,
        W: int,
        used_coords: set,
        lbl: np.ndarray,
        k: int = 3,
        max_tries: int = 100,
    ) -> List[Dict[str, int]]:
        """Randomly sample up to k valid nearby coordinates within a disk (radius=self.radius)."""
        neighbors: List[Dict[str, int]] = []
        tries = 0

        while len(neighbors) < k and tries < max_tries:
            dy = self.rng.randint(-self.radius, self.radius)
            dx = self.rng.randint(-self.radius, self.radius)

            # reject outside disk or center itself
            if dx * dx + dy * dy > self.radius * self.radius or (dx == 0 and dy == 0):
                tries += 1
                continue

            ny, nx = cy + dy, cx + dx
            coord = (ny, nx)

            if coord in used_coords:
                tries += 1
                continue

            if self._is_valid_coord(name, cz, ny, nx, Z, H, W):
                used_coords.add(coord)
                neighbors.append(
                    {
                        "y": int(ny),
                        "x": int(nx),
                        "label": int(lbl[cz, ny, nx].item()),
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
    ) -> Dict[str, Any]:
        """Create the output dict for one (center + neighbors) group."""
        return {
            "name": name,
            "z": int(cz),
            "coords": [(int(cy), int(cx))] + [(n["y"], n["x"]) for n in neighbors],
            "labels": [int(c)] + [n["label"] for n in neighbors],
        }

    def _report_class_counts(self, groups: List[dict]) -> None:
        """Print class counts for centers and neighbors separately."""
        centers = [g["labels"][0] for g in groups]
        neighbors = [lab for g in groups for lab in g["labels"][1:]]

        for title, labs in (("anchors", centers), ("neighbors", neighbors)):
            counts = Counter(labs)
            for k in sorted(counts):
                print(f"  Class {k} ({title}): {counts[k]} samples")



class PredictionDataset(Dataset):
    """
    Yields only 64x64 patches whose center (31,31) has label != -1 for a fixed z slice.
    image: (Z,H,W)  (or (C,H,W) if you adapt for multichannel)
    label: (Z,H,W)
    """
    def __init__(self, image, label, z, patch_size=64):
        self.image = image
        self.label = label
        self.z = int(z)
        self.ps = int(patch_size)
        assert self.ps % 2 == 0, "Patch size must be even; center is (ps/2-1, ps/2-1)."
        self.half = self.ps // 2  # 32 for 64x64 -> center at (31,31)

        H, W = image[self.z].shape
        assert (H, W) == label[self.z].shape

        # Valid centers: label!= -1 and full patch inside bounds
        y0, y1 = self.half, H - self.half
        x0, x1 = self.half, W - self.half
        mask = (label[self.z] != -1)
        mask[:y0, :] = False
        mask[y1:, :] = False
        mask[:, :x0] = False
        mask[:, x1:] = False

        ys, xs = np.where(mask)
        self.centers = np.stack([ys, xs], axis=1).astype(np.int32)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        y, x = self.centers[idx]
        y0, y1 = y - self.half, y + self.half
        x0, x1 = x - self.half, x + self.half
        patch = self.image[self.z, y0:y1, x0:x1]              # (64,64)
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # (1,64,64)
        center_label = int(self.label[self.z, y, x])
        return {"patch": patch, "z": self.z, "y": int(y), "x": int(x), "center_label": center_label}