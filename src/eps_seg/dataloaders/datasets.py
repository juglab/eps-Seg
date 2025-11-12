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
        samples_per_class: Dict[int, int] = {1: 2},
    ):
        self.patch_size = patch_size
        self.label_size = label_size
        self.offset = self.patch_size // 2 - self.label_size
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
        self.samples_per_class = samples_per_class # for class 1 (nucleus), we take 2 to compensate for rarity along z
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
            self.offset <= y < H - self.offset - 1 and self.offset <= x < W - self.offset - 1
        )
        if self.dim == 3:
            valid = valid and (self.offset <= z < Z - self.offset - 1)

        in_cell = self.labels[name][z, y, x] != self.ignore_lbl
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
            return torch.from_numpy(p).unsqueeze(0)  # [1, H, W]
        else:  # 3D
            p = img_stack[
                z - self.offset : z + self.offset + 2,
                y - self.offset : y + self.offset + 2,
                x - self.offset : x + self.offset + 2,
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
            dz = self.rng.randint(-self.radius, self.radius) if self.dim ==3 else 0
            dy = self.rng.randint(-self.radius, self.radius)
            dx = self.rng.randint(-self.radius, self.radius)

            # reject outside disk or center itself
            if dx * dx + dy * dy + dz * dz > self.radius * self.radius or (dx == 0 and dy == 0 and (dz == 0 if self.dim==3 else False)):
                tries += 1
                continue

            nz, ny, nx = cz + dz, cy + dy, cx + dx
            coord = (nz, ny, nx)
            
            if coord in used_coords:
                tries += 1
                continue

            if self._is_valid_coord(name, nz, ny, nx, Z, H, W):
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
    ) -> Dict[str, Any]:
        """Create the output dict for one (center + neighbors) group."""
        return {
            "name": name,
            "z": int(cz),
            "coords": [(int(cz), int(cy), int(cx))] + [(n["z"], n["y"], n["x"]) for n in neighbors],
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
        Yields 2D or 3D patches whose center has label != -1 and full patch is inside bounds.
        Optionally, can produce a single z slice if z is provided.

        image: (C,Z,H,W) or (Z,1,H,W)

    """
    def __init__(self, image, label, z=None, patch_size=64, dim=2):
        self.dim = dim
        self.image = image
        self.label = label
        self.z = int(z) if z is not None else None
        self.ps = int(patch_size)
        assert self.ps % 2 == 0, "Patch size must be even; center is (ps/2-1, ps/2-1)."
        self.half = self.ps // 2  # 32 for 64x64 -> center at (31,31)

        if image.ndim == 3:
            Z, H, W = self.image.shape
            C = 1
            self.image = self.image[None, ...]  # add channel dim
        else:
            C, Z, H, W = self.image.shape

        if self.label.ndim == 3:
            self.label = self.label[None, ...]  # add channel dim

        assert self.image.shape[-dim:] == self.label.shape[-dim:], "Image and label must have same (Z),H,W"

        # Valid centers are:
        # 1) label != -1
        mask = (self.label != -1)
        # 2) full patch inside bounds
        mask[:, :, :, :self.half] = False
        mask[:, :, :, W - self.half:] = False
        mask[:, :, :self.half, :] = False
        mask[:, :, H - self.half:, :] = False
        mask[:, :self.half, :, :] = False
        mask[:, Z - self.half:, :, :] = False
        # If "fast_testing" on a single z slice:
        if self.z is not None:
            # Set all other z slices to False
            mask[:, :self.z, :, :] = False
            mask[:, self.z + 1:, :, :] = False


        _, zs, ys, xs = np.where(mask)
        self.centers = np.stack([zs, ys, xs], axis=1).astype(np.int32)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        z, y, x = self.centers[idx]

        if self.dim == 3:
            z0, z1 = z - self.half, z + self.half
        else:
            z0, z1 = z, z + 1

        y0, y1 = y - self.half, y + self.half
        x0, x1 = x - self.half, x + self.half
        patch = self.image[:, z0:z1, y0:y1, x0:x1]
        patch = torch.from_numpy(patch).float()
        
        center_label = int(self.label[:, z, y, x].item())
        # Drop channel dim to get [B, D, H, W] in the DataLoader batches.
        # TODO: For multichannel this will return different shapes!
        patch = patch.squeeze(0) 
        return {"patch": patch, "z": int(z), "y": int(y), "x": int(x), "center_label": center_label}