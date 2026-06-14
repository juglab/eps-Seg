import argparse
import json
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eps_seg.config.train import ExperimentConfig
from eps_seg.models import LVAEModel


PoolingMode = Literal["center", "mean", "flatten"]


def load_tiff_array(path: str | Path) -> np.ndarray:
    """Memory-map contiguous TIFFs and load compressed/tiled TIFFs normally."""

    try:
        return tiff.memmap(path, mode="r")
    except ValueError as error:
        if "not memory-mappable" not in str(error):
            raise
        print(f"TIFF is not memory-mappable; loading it into RAM: {path}")
        return tiff.imread(path)


class TestSliceDataset(Dataset):
    """Return one model patch for every evaluable pixel in a test slice."""

    def __init__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        slice_idx: int,
        patch_size: int,
        dim: int,
        n_classes: int,
    ):
        if patch_size % 2 != 0:
            raise ValueError("patch_size must be even.")
        if dim not in {2, 3}:
            raise ValueError("dim must be 2 or 3.")

        self.image = image[None, ...] if image.ndim == 3 else image
        label_volume = labels[0] if labels.ndim == 4 else labels
        if self.image.ndim != 4 or label_volume.ndim != 3:
            raise ValueError(
                "Expected image shape (Z,H,W) or (C,Z,H,W) and label shape "
                "(Z,H,W) or (1,Z,H,W)."
            )

        self.labels = label_volume
        self.slice_idx = int(slice_idx)
        self.patch_size = int(patch_size)
        self.half = self.patch_size // 2
        self.dim = int(dim)

        _, depth, height, width = self.image.shape
        if not 0 <= self.slice_idx < depth:
            raise ValueError(
                f"slice_idx={self.slice_idx} is outside image depth {depth}."
            )
        if self.dim == 3 and not self.half <= self.slice_idx < depth - self.half:
            raise ValueError(
                f"slice_idx={self.slice_idx} cannot support a {patch_size}^3 patch "
                f"inside image depth {depth}."
            )

        valid = (
            (self.labels[self.slice_idx] >= 0)
            & (self.labels[self.slice_idx] < n_classes)
        )
        valid[: self.half, :] = False
        valid[height - self.half :, :] = False
        valid[:, : self.half] = False
        valid[:, width - self.half :] = False
        self.yx = np.argwhere(valid).astype(np.int32)

        if len(self.yx) == 0:
            raise ValueError(
                f"No evaluable pixels found at slice {self.slice_idx}."
            )

    def __len__(self) -> int:
        return len(self.yx)

    def __getitem__(self, index: int):
        y, x = (int(value) for value in self.yx[index])
        if self.dim == 2:
            patch = self.image[
                :,
                self.slice_idx,
                y - self.half : y + self.half,
                x - self.half : x + self.half,
            ]
        else:
            patch = self.image[
                :,
                self.slice_idx - self.half : self.slice_idx + self.half,
                y - self.half : y + self.half,
                x - self.half : x + self.half,
            ]

        return (
            torch.from_numpy(np.asarray(patch, dtype=np.float32)),
            torch.tensor(int(self.labels[self.slice_idx, y, x]), dtype=torch.long),
            torch.tensor((self.slice_idx, y, x), dtype=torch.int32),
        )


def pool_latent_mus(
    mus: Sequence[torch.Tensor],
    mode: PoolingMode,
) -> torch.Tensor:
    """Convert each hierarchy's posterior mean into one feature vector per pixel."""

    pooled = []
    for layer_mu in mus:
        if layer_mu.ndim < 3:
            raise ValueError(
                f"Expected latent mean shape (B,C,...), got {tuple(layer_mu.shape)}."
            )
        if mode == "flatten":
            layer_features = layer_mu.flatten(start_dim=1)
        elif mode == "mean":
            layer_features = layer_mu.flatten(start_dim=2).mean(dim=2)
        elif mode == "center":
            center = tuple(size // 2 for size in layer_mu.shape[2:])
            layer_features = layer_mu[(slice(None), slice(None), *center)]
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        pooled.append(layer_features.float())

    if not pooled:
        raise ValueError("The model did not return latent means.")
    return torch.cat(pooled, dim=1)


def align_clusters_to_ground_truth(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    n_clusters: int,
    n_classes: int,
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    """Resolve arbitrary cluster IDs with a maximum-overlap Hungarian assignment."""

    clusters = np.asarray(cluster_labels, dtype=np.int64)
    gt = np.asarray(gt_labels, dtype=np.int64)
    valid = (
        (clusters >= 0)
        & (clusters < n_clusters)
        & (gt >= 0)
        & (gt < n_classes)
    )
    contingency = np.zeros((n_clusters, n_classes), dtype=np.int64)
    np.add.at(contingency, (clusters[valid], gt[valid]), 1)

    cluster_indices, class_indices = linear_sum_assignment(-contingency)
    mapping = {
        int(cluster_idx): int(class_idx)
        for cluster_idx, class_idx in zip(cluster_indices, class_indices)
    }
    aligned = np.full(clusters.shape, -1, dtype=np.int16)
    for cluster_idx, class_idx in mapping.items():
        aligned[clusters == cluster_idx] = class_idx
    return aligned, mapping, contingency


def dice_scores(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    n_classes: int,
) -> tuple[float, dict[int, float]]:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    valid = (target >= 0) & (target < n_classes)
    per_class = {}
    for class_idx in range(n_classes):
        pred_class = predicted[valid] == class_idx
        target_class = target[valid] == class_idx
        denominator = int(pred_class.sum() + target_class.sum())
        per_class[class_idx] = (
            float(2 * np.logical_and(pred_class, target_class).sum() / denominator)
            if denominator
            else float("nan")
        )
    finite = [score for score in per_class.values() if np.isfinite(score)]
    return (float(np.mean(finite)) if finite else float("nan")), per_class


def _resolve_checkpoint(exp: ExperimentConfig, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        checkpoint_path = exp.best_checkpoint_path("supervised").parent / checkpoint
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    for mode in ("semisupervised", "supervised"):
        checkpoint_path = exp.best_checkpoint_path(mode)
        if checkpoint_path.exists():
            return checkpoint_path
    raise FileNotFoundError(
        "Neither best_semisupervised.ckpt nor best_supervised.ckpt exists."
    )


def _resolve_slice_idx(dataset_cfg, test_key: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    key_idx = dataset_cfg.test_keys.index(test_key)
    configured = dataset_cfg.test_center_slices[key_idx]
    if configured is None:
        _, label_path = dataset_cfg.get_image_label_paths([test_key])[test_key]
        return int(tiff.memmap(label_path).shape[-3] // 2)
    return int(configured)


def _feature_chunks(features: np.ndarray, chunk_size: int):
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        yield start, stop, np.asarray(features[start:stop], dtype=np.float32)


def _transform_features(
    features: np.ndarray,
    *,
    scaler: StandardScaler | None,
    pca: PCA | None,
) -> np.ndarray:
    transformed = np.asarray(features, dtype=np.float32)
    if scaler is not None:
        transformed = scaler.transform(transformed).astype(np.float32, copy=False)
    if pca is not None:
        transformed = pca.transform(transformed).astype(np.float32, copy=False)
    return transformed


def fit_kmeans_streaming(
    features: np.ndarray,
    *,
    n_clusters: int,
    standardize: bool,
    pca_components: int | None,
    pca_fit_samples: int,
    batch_size: int,
    epochs: int,
    init_samples: int,
    seed: int,
    projected_features_path: str | Path | None = None,
) -> tuple[
    np.ndarray,
    MiniBatchKMeans,
    StandardScaler | None,
    PCA | None,
]:
    """Fit MiniBatchKMeans without materializing a second full feature matrix."""

    if batch_size <= 0:
        raise ValueError("K-means batch_size must be positive.")
    if epochs < 0:
        raise ValueError("K-means epochs cannot be negative.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if len(features) < n_clusters:
        raise ValueError(
            f"Cannot fit {n_clusters} clusters from only {len(features)} pixels."
        )
    if pca_components is not None:
        pca_components = int(pca_components)
        max_components = min(len(features), features.shape[1])
        if not 1 <= pca_components <= max_components:
            raise ValueError(
                f"pca_components must be between 1 and {max_components}; got "
                f"{pca_components}."
            )
        if batch_size < pca_components:
            raise ValueError(
                "K-means batch_size must be at least pca_components."
            )
        if pca_fit_samples < pca_components:
            raise ValueError(
                "pca_fit_samples must be at least pca_components."
            )

    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        for _, _, chunk in tqdm(
            _feature_chunks(features, batch_size),
            desc="Fitting feature scaler",
            total=(len(features) + batch_size - 1) // batch_size,
        ):
            scaler.partial_fit(chunk)

    rng = np.random.default_rng(seed)
    pca = None
    if pca_components is not None:
        pca_sample_count = min(int(pca_fit_samples), len(features))
        pca_indices = rng.choice(
            len(features),
            size=pca_sample_count,
            replace=False,
        )
        pca_fit_features = _transform_features(
            features[pca_indices],
            scaler=scaler,
            pca=None,
        )
        pca = PCA(
            n_components=pca_components,
            svd_solver="randomized",
            random_state=seed,
        )
        print(
            f"Fitting randomized PCA ({pca_components} components) on "
            f"{pca_sample_count:,} sampled pixels."
        )
        pca.fit(pca_fit_features)
        del pca_fit_features

    clustering_features = features
    clustering_scaler = scaler
    clustering_pca = pca
    projected_features = None
    if pca is not None and projected_features_path is not None:
        projected_features_path = Path(projected_features_path)
        projected_features_path.parent.mkdir(parents=True, exist_ok=True)
        projected_features = np.lib.format.open_memmap(
            projected_features_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(features), int(pca.n_components_)),
        )
        for start, stop, chunk in tqdm(
            _feature_chunks(features, batch_size),
            desc="Projecting PCA features",
            total=(len(features) + batch_size - 1) // batch_size,
        ):
            projected_features[start:stop] = _transform_features(
                chunk,
                scaler=scaler,
                pca=pca,
            )
        projected_features.flush()
        clustering_features = projected_features
        clustering_scaler = None
        clustering_pca = None

    sample_count = min(int(init_samples), len(clustering_features))
    if sample_count < n_clusters:
        raise ValueError(
            f"init_samples must provide at least {n_clusters} pixels; got "
            f"{sample_count}."
        )
    init_indices = rng.choice(
        len(clustering_features),
        size=sample_count,
        replace=False,
    )
    init_features = _transform_features(
        clustering_features[init_indices],
        scaler=clustering_scaler,
        pca=clustering_pca,
    )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        batch_size=batch_size,
        random_state=seed,
    )
    kmeans.fit(init_features)

    for epoch in range(epochs):
        progress = tqdm(
            _feature_chunks(clustering_features, batch_size),
            desc=f"Fitting K-means {epoch + 1}/{epochs}",
            total=(len(clustering_features) + batch_size - 1) // batch_size,
        )
        for _, _, chunk in progress:
            chunk = _transform_features(
                chunk,
                scaler=clustering_scaler,
                pca=clustering_pca,
            )
            kmeans.partial_fit(chunk)

    cluster_labels = np.empty(len(clustering_features), dtype=np.int32)
    inertia = 0.0
    for start, stop, chunk in tqdm(
        _feature_chunks(clustering_features, batch_size),
        desc="Assigning clusters",
        total=(len(clustering_features) + batch_size - 1) // batch_size,
    ):
        chunk = _transform_features(
            chunk,
            scaler=clustering_scaler,
            pca=clustering_pca,
        )
        chunk_labels = kmeans.predict(chunk)
        cluster_labels[start:stop] = chunk_labels
        residuals = chunk - kmeans.cluster_centers_[chunk_labels]
        inertia += float(np.square(residuals).sum(dtype=np.float64))
    kmeans.inertia_ = inertia
    if projected_features is not None:
        projected_features.flush()

    return cluster_labels, kmeans, scaler, pca


def sampled_silhouette_score(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    scaler: StandardScaler | None,
    pca: PCA | None,
    sample_size: int,
    seed: int,
) -> tuple[float, int]:
    if sample_size <= 0 or len(np.unique(cluster_labels)) < 2:
        return float("nan"), 0

    rng = np.random.default_rng(seed)
    sample_size = min(int(sample_size), len(features))
    indices = rng.choice(len(features), size=sample_size, replace=False)
    sampled_features = _transform_features(
        features[indices],
        scaler=scaler,
        pca=pca,
    )
    sampled_labels = cluster_labels[indices]
    n_sampled_clusters = len(np.unique(sampled_labels))
    if n_sampled_clusters < 2 or n_sampled_clusters >= sample_size:
        return float("nan"), sample_size
    return (
        float(silhouette_score(sampled_features, sampled_labels)),
        sample_size,
    )


def save_pca_artifacts(
    features: np.ndarray,
    *,
    scaler: StandardScaler | None,
    pca: PCA,
    output_dir: Path,
    batch_size: int,
    gt_labels: np.ndarray,
    coords: np.ndarray,
    direct_labels: np.ndarray,
    gt_slice: np.ndarray,
    metadata: dict,
) -> Path:
    """Save projected pixel features and enough context to recluster them."""

    pca_features_path = output_dir / "pca_features.npy"
    if pca_features_path.exists():
        projected = np.load(pca_features_path, mmap_mode="r")
        expected_shape = (len(features), int(pca.n_components_))
        if projected.shape != expected_shape:
            raise ValueError(
                f"Existing PCA feature shape {projected.shape} does not match "
                f"{expected_shape}."
            )
        del projected
    else:
        projected = np.lib.format.open_memmap(
            pca_features_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(features), int(pca.n_components_)),
        )
        for start, stop, chunk in tqdm(
            _feature_chunks(features, batch_size),
            desc="Saving PCA features",
            total=(len(features) + batch_size - 1) // batch_size,
        ):
            projected[start:stop] = _transform_features(
                chunk,
                scaler=scaler,
                pca=pca,
            )
        projected.flush()
        del projected

    transform_values = {
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "pca_explained_variance": pca.explained_variance_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_singular_values": pca.singular_values_,
    }
    if scaler is not None:
        transform_values.update(
            {
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "scaler_var": scaler.var_,
            }
        )
    np.savez_compressed(output_dir / "pca_transform.npz", **transform_values)

    np.savez_compressed(
        output_dir / "pca_features_metadata.npz",
        gt_labels=np.asarray(gt_labels, dtype=np.int16),
        coords=np.asarray(coords, dtype=np.int32),
        direct_labels=np.asarray(direct_labels, dtype=np.int16),
        gt_slice=np.asarray(gt_slice, dtype=np.int16),
        **metadata,
    )
    print(f"Saved reusable PCA features to {pca_features_path}")
    return pca_features_path


def _save_label_image(
    labels: np.ndarray,
    coords: np.ndarray,
    *,
    image_shape: tuple[int, int],
    output_path: Path,
) -> np.ndarray:
    image = np.full(image_shape, -1, dtype=np.int16)
    image[coords[:, 1], coords[:, 2]] = labels.astype(np.int16)
    tiff.imwrite(output_path, image)
    return image


def _save_summary_figure(
    *,
    gt_slice: np.ndarray,
    direct_image: np.ndarray,
    raw_cluster_image: np.ndarray,
    aligned_cluster_image: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("Ground truth", gt_slice),
        ("Trained classifier", direct_image),
        ("Raw K-means IDs", raw_cluster_image),
        ("Aligned K-means", aligned_cluster_image),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(
            np.ma.masked_where(image < 0, image),
            cmap="tab20",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_cluster_evaluation(
    *,
    output_dir: Path,
    cluster_labels: np.ndarray,
    kmeans: MiniBatchKMeans,
    gt_labels: np.ndarray,
    coords: np.ndarray,
    direct_labels: np.ndarray,
    gt_slice: np.ndarray,
    n_classes: int,
    n_clusters: int,
    silhouette: float,
    silhouette_samples: int,
    result_base: dict,
) -> Path:
    aligned_labels, cluster_mapping, contingency = align_clusters_to_ground_truth(
        cluster_labels,
        gt_labels,
        n_clusters=n_clusters,
        n_classes=n_classes,
    )
    kmeans_dice, kmeans_dice_per_class = dice_scores(
        aligned_labels,
        gt_labels,
        n_classes=n_classes,
    )
    direct_dice, direct_dice_per_class = dice_scores(
        direct_labels,
        gt_labels,
        n_classes=n_classes,
    )

    image_shape = tuple(int(size) for size in gt_slice.shape[-2:])
    raw_cluster_image = _save_label_image(
        cluster_labels,
        coords,
        image_shape=image_shape,
        output_path=output_dir / "kmeans_raw_clusters.tif",
    )
    aligned_cluster_image = _save_label_image(
        aligned_labels,
        coords,
        image_shape=image_shape,
        output_path=output_dir / "kmeans_aligned_labels.tif",
    )
    direct_image = _save_label_image(
        direct_labels,
        coords,
        image_shape=image_shape,
        output_path=output_dir / "model_direct_labels.tif",
    )
    _save_summary_figure(
        gt_slice=gt_slice,
        direct_image=direct_image,
        raw_cluster_image=raw_cluster_image,
        aligned_cluster_image=aligned_cluster_image,
        output_path=output_dir / "comparison.png",
    )

    mapping_rows = []
    for cluster_idx in range(n_clusters):
        class_idx = cluster_mapping.get(cluster_idx, -1)
        mapping_rows.append(
            {
                "cluster": cluster_idx,
                "mapped_class": class_idx,
                "cluster_size": int((cluster_labels == cluster_idx).sum()),
                "matched_pixels": (
                    int(contingency[cluster_idx, class_idx])
                    if class_idx >= 0
                    else 0
                ),
            }
        )
    pd.DataFrame(mapping_rows).to_csv(
        output_dir / "cluster_mapping.csv",
        index=False,
    )

    result = {
        **result_base,
        "n_pixels": len(gt_labels),
        "n_clusters": n_clusters,
        "kmeans_inertia": float(kmeans.inertia_),
        "silhouette_score": silhouette,
        "silhouette_samples": silhouette_samples,
        "adjusted_rand_score": float(
            adjusted_rand_score(gt_labels, cluster_labels)
        ),
        "normalized_mutual_info_score": float(
            normalized_mutual_info_score(gt_labels, cluster_labels)
        ),
        "kmeans_average_dice": kmeans_dice,
        "model_direct_average_dice": direct_dice,
    }
    for class_idx, score in kmeans_dice_per_class.items():
        result[f"kmeans_dice_class_{class_idx}"] = score
    for class_idx, score in direct_dice_per_class.items():
        result[f"model_direct_dice_class_{class_idx}"] = score

    results_path = output_dir / "results.csv"
    pd.DataFrame([result]).to_csv(results_path, index=False)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **result,
                "cluster_to_class_mapping": cluster_mapping,
                "contingency_matrix": contingency.tolist(),
            },
            f,
            indent=2,
        )

    print(f"K-means macro Dice: {kmeans_dice:.4f}")
    print(f"Direct model macro Dice: {direct_dice:.4f}")
    print(
        f"Silhouette score ({silhouette_samples:,} samples): "
        f"{silhouette:.4f}"
    )
    print(f"Wrote latent clustering evaluation to {output_dir}")
    return results_path


def evaluate_saved_pca_kmeans(
    *,
    exp_config_path: str | Path,
    pca_features_path: str | Path,
    output_dir: str | Path | None,
    n_clusters: int | None,
    kmeans_batch_size: int,
    kmeans_epochs: int,
    init_samples: int,
    silhouette_samples: int,
    seed: int,
) -> Path:
    exp = ExperimentConfig.from_yaml(Path(exp_config_path))
    _, dataset_cfg, _ = exp.get_configs()
    n_clusters = int(dataset_cfg.n_classes if n_clusters is None else n_clusters)

    pca_features_path = Path(pca_features_path)
    metadata_path = pca_features_path.with_name("pca_features_metadata.npz")
    if not pca_features_path.exists():
        raise FileNotFoundError(f"PCA features not found: {pca_features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"PCA feature metadata not found: {metadata_path}")

    features = np.load(pca_features_path, mmap_mode="r")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        gt_labels = metadata["gt_labels"]
        coords = metadata["coords"]
        direct_labels = metadata["direct_labels"]
        gt_slice = metadata["gt_slice"]
        result_base = {
            "experiment": exp.experiment_name,
            "checkpoint": str(metadata["checkpoint"].item()),
            "test_key": str(metadata["test_key"].item()),
            "slice_idx": int(metadata["slice_idx"].item()),
            "pooling": str(metadata["pooling"].item()),
            "standardized": bool(metadata["standardized"].item()),
            "feature_dim": int(metadata["feature_dim"].item()),
            "clustering_feature_dim": int(features.shape[1]),
            "pca_components": int(features.shape[1]),
            "pca_explained_variance_ratio": float(
                metadata["pca_explained_variance_ratio"].item()
            ),
            "reused_pca_features": str(pca_features_path),
        }

    if not (len(features) == len(gt_labels) == len(coords) == len(direct_labels)):
        raise ValueError("PCA features and metadata contain different pixel counts.")

    if output_dir is None:
        output_dir = pca_features_path.parent / f"kmeans_k{n_clusters}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_labels, kmeans, _, _ = fit_kmeans_streaming(
        features,
        n_clusters=n_clusters,
        standardize=False,
        pca_components=None,
        pca_fit_samples=0,
        batch_size=kmeans_batch_size,
        epochs=kmeans_epochs,
        init_samples=init_samples,
        seed=seed,
    )
    silhouette, actual_silhouette_samples = sampled_silhouette_score(
        features,
        cluster_labels,
        scaler=None,
        pca=None,
        sample_size=silhouette_samples,
        seed=seed,
    )
    return _write_cluster_evaluation(
        output_dir=output_dir,
        cluster_labels=cluster_labels,
        kmeans=kmeans,
        gt_labels=gt_labels,
        coords=coords,
        direct_labels=direct_labels,
        gt_slice=gt_slice,
        n_classes=int(dataset_cfg.n_classes),
        n_clusters=n_clusters,
        silhouette=silhouette,
        silhouette_samples=actual_silhouette_samples,
        result_base=result_base,
    )


def evaluate_latent_kmeans(
    *,
    exp_config_path: str | Path,
    checkpoint: str | None = None,
    test_key: str | None = None,
    slice_idx: int | None = None,
    output_dir: str | Path | None = None,
    pooling: PoolingMode = "center",
    batch_size: int | None = None,
    num_workers: int = 0,
    device: str = "auto",
    disable_amp: bool = False,
    n_clusters: int | None = None,
    standardize: bool = True,
    pca_components: int | None = None,
    pca_fit_samples: int = 10_000,
    kmeans_batch_size: int = 4096,
    kmeans_epochs: int = 3,
    init_samples: int = 50_000,
    silhouette_samples: int = 5_000,
    seed: int = 42,
    keep_features: bool = False,
    feature_temp_dir: str | Path | None = None,
) -> Path:
    exp = ExperimentConfig.from_yaml(Path(exp_config_path))
    train_cfg, dataset_cfg, model_cfg = exp.get_configs()
    checkpoint_path = _resolve_checkpoint(exp, checkpoint)

    test_key = test_key or dataset_cfg.test_keys[0]
    if test_key not in dataset_cfg.test_keys:
        raise ValueError(
            f"Unknown test_key={test_key!r}; expected one of {dataset_cfg.test_keys}."
        )
    slice_idx = _resolve_slice_idx(dataset_cfg, test_key, slice_idx)
    n_clusters = int(dataset_cfg.n_classes if n_clusters is None else n_clusters)
    batch_size = int(train_cfg.test_batch_size if batch_size is None else batch_size)
    if batch_size <= 0:
        raise ValueError("Inference batch_size must be positive.")

    image_path, label_path = dataset_cfg.get_image_label_paths([test_key])[test_key]
    image = load_tiff_array(image_path)
    labels = load_tiff_array(label_path)
    dataset = TestSliceDataset(
        image,
        labels,
        slice_idx=slice_idx,
        patch_size=int(model_cfg.img_shape[-1]),
        dim=int(model_cfg.conv_mult),
        n_classes=int(dataset_cfg.n_classes),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    model = LVAEModel.load_from_checkpoint(
        str(checkpoint_path),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        map_location=torch_device,
    )
    model.eval()
    model.to(torch_device)

    data_mean = float(model.model.data_mean.detach().cpu().item())
    data_std = float(model.model.data_std.detach().cpu().item())
    if not np.isfinite(data_std) or data_std == 0.0:
        raise ValueError(
            "The checkpoint does not contain a valid non-zero data_std buffer."
        )

    if output_dir is None:
        output_dir = (
            exp.outputs_dir
            / "latent_kmeans"
            / checkpoint_path.stem
            / f"{test_key}_z{slice_idx}_{pooling}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_parent = Path(feature_temp_dir) if feature_temp_dir is not None else output_dir
    temp_parent.mkdir(parents=True, exist_ok=True)
    temporary_context = (
        nullcontext(output_dir)
        if keep_features
        else tempfile.TemporaryDirectory(prefix="latent_kmeans_", dir=temp_parent)
    )
    with temporary_context as feature_dir:
        feature_path = Path(feature_dir) / "latent_features.npy"
        feature_array = None
        gt_labels = np.empty(len(dataset), dtype=np.int16)
        coords = np.empty((len(dataset), 3), dtype=np.int32)
        direct_labels = np.empty(len(dataset), dtype=np.int16)
        offset = 0

        amp_enabled = (
            torch_device.type == "cuda" and bool(train_cfg.amp) and not disable_amp
        )
        with torch.inference_mode():
            for patches, batch_gt, batch_coords in tqdm(
                loader,
                desc="Extracting latent means",
                total=len(loader),
            ):
                patches = patches.to(torch_device, non_blocking=True)
                patches = (patches - data_mean) / data_std
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                ):
                    outputs = model(
                        patches,
                        y=None,
                        validation_mode=False,
                        mask_input=False,
                    )
                batch_features = (
                    pool_latent_mus(outputs["mu"], pooling)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                if feature_array is None:
                    feature_dim = int(batch_features.shape[1])
                    estimated_gib = len(dataset) * feature_dim * 4 / (1024**3)
                    print(
                        f"Latent matrix: {len(dataset):,} pixels x {feature_dim:,} features "
                        f"(about {estimated_gib:.2f} GiB on disk)."
                    )
                    free_bytes = shutil.disk_usage(feature_path.parent).free
                    required_bytes = len(dataset) * feature_dim * 4
                    if free_bytes < int(required_bytes * 1.05):
                        raise OSError(
                            f"Insufficient temporary disk space in "
                            f"{feature_path.parent}: need about "
                            f"{required_bytes / (1024**3):.2f} GiB, have "
                            f"{free_bytes / (1024**3):.2f} GiB free."
                        )
                    feature_array = np.lib.format.open_memmap(
                        feature_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(dataset), feature_dim),
                    )

                stop = offset + len(batch_features)
                feature_array[offset:stop] = batch_features
                gt_labels[offset:stop] = batch_gt.numpy().reshape(-1)
                coords[offset:stop] = batch_coords.numpy()
                direct_labels[offset:stop] = (
                    outputs["class_probabilities"].argmax(dim=-1).cpu().numpy()
                )
                offset = stop

        if feature_array is None or offset != len(dataset):
            raise RuntimeError(
                f"Extracted {offset} feature rows for a dataset of size {len(dataset)}."
            )
        feature_array.flush()

        cluster_labels, kmeans, scaler, pca = fit_kmeans_streaming(
            feature_array,
            n_clusters=n_clusters,
            standardize=standardize,
            pca_components=pca_components,
            pca_fit_samples=pca_fit_samples,
            batch_size=kmeans_batch_size,
            epochs=kmeans_epochs,
            init_samples=init_samples,
            seed=seed,
            projected_features_path=(
                output_dir / "pca_features.npy"
                if pca_components is not None
                else None
            ),
        )
        silhouette_features = feature_array
        silhouette_scaler = scaler
        silhouette_pca = pca
        if pca is not None:
            silhouette_features = np.load(
                output_dir / "pca_features.npy",
                mmap_mode="r",
            )
            silhouette_scaler = None
            silhouette_pca = None
        silhouette, actual_silhouette_samples = sampled_silhouette_score(
            silhouette_features,
            cluster_labels,
            scaler=silhouette_scaler,
            pca=silhouette_pca,
            sample_size=silhouette_samples,
            seed=seed,
        )

        pca_explained_variance_ratio = (
            float(pca.explained_variance_ratio_.sum())
            if pca is not None
            else None
        )
        gt_slice = np.asarray(dataset.labels[slice_idx])
        if pca is not None:
            save_pca_artifacts(
                feature_array,
                scaler=scaler,
                pca=pca,
                output_dir=output_dir,
                batch_size=kmeans_batch_size,
                gt_labels=gt_labels,
                coords=coords,
                direct_labels=direct_labels,
                gt_slice=gt_slice,
                metadata={
                    "checkpoint": np.asarray(str(checkpoint_path)),
                    "test_key": np.asarray(test_key),
                    "slice_idx": np.asarray(slice_idx, dtype=np.int32),
                    "pooling": np.asarray(pooling),
                    "standardized": np.asarray(standardize),
                    "feature_dim": np.asarray(
                        feature_array.shape[1],
                        dtype=np.int32,
                    ),
                    "pca_explained_variance_ratio": np.asarray(
                        pca_explained_variance_ratio,
                        dtype=np.float64,
                    ),
                },
            )

        result_base = {
            "experiment": exp.experiment_name,
            "checkpoint": str(checkpoint_path),
            "test_key": test_key,
            "slice_idx": slice_idx,
            "pooling": pooling,
            "standardized": standardize,
            "n_pixels": len(dataset),
            "feature_dim": int(feature_array.shape[1]),
            "clustering_feature_dim": (
                int(pca.n_components_) if pca is not None else int(feature_array.shape[1])
            ),
            "pca_components": (
                int(pca.n_components_) if pca is not None else None
            ),
            "pca_fit_samples": (
                min(int(pca_fit_samples), len(dataset))
                if pca is not None
                else None
            ),
            "pca_explained_variance_ratio": pca_explained_variance_ratio,
            "reused_pca_features": None,
        }
        return _write_cluster_evaluation(
            output_dir=output_dir,
            cluster_labels=cluster_labels,
            kmeans=kmeans,
            gt_labels=gt_labels,
            coords=coords,
            direct_labels=direct_labels,
            gt_slice=gt_slice,
            n_classes=int(dataset_cfg.n_classes),
            n_clusters=n_clusters,
            silhouette=silhouette,
            silhouette_samples=actual_silhouette_samples,
            result_base=result_base,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster concatenated per-layer posterior means for every evaluable "
            "pixel in one test slice."
        )
    )
    parser.add_argument("--exp-config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path or filename. Defaults to best_semisupervised.ckpt, "
            "then best_supervised.ckpt."
        ),
    )
    parser.add_argument("--test-key", default=None)
    parser.add_argument("--slice-idx", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--reuse-pca-features",
        default=None,
        help=(
            "Path to a previously saved pca_features.npy. This skips checkpoint "
            "inference and PCA, allowing fast experiments with another k."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=["center", "mean", "flatten"],
        default="center",
        help=(
            "How each spatial latent map becomes a per-pixel vector. 'center' "
            "concatenates center-channel vectors; 'mean' spatially averages each "
            "layer; 'flatten' stores every latent value and can be very large."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "Optionally reduce concatenated latent features with sampled randomized "
            "PCA before K-means, for example 32. Silhouette is computed after PCA."
        ),
    )
    parser.add_argument(
        "--pca-fit-samples",
        type=int,
        default=10_000,
        help="Number of randomly selected pixels used to fit PCA.",
    )
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-epochs", type=int, default=3)
    parser.add_argument("--init-samples", type=int, default=50_000)
    parser.add_argument("--silhouette-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-features",
        action="store_true",
        help="Keep the potentially large latent_features.npy file.",
    )
    parser.add_argument(
        "--feature-temp-dir",
        default=None,
        help=(
            "Directory for the large temporary latent matrix. On Slurm, use "
            "$SLURM_TMPDIR or another node-local scratch path."
        ),
    )
    args = parser.parse_args()

    if args.reuse_pca_features is not None:
        if args.pca_components is not None:
            parser.error(
                "--pca-components cannot be combined with --reuse-pca-features."
            )
        evaluate_saved_pca_kmeans(
            exp_config_path=args.exp_config,
            pca_features_path=args.reuse_pca_features,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            kmeans_batch_size=args.kmeans_batch_size,
            kmeans_epochs=args.kmeans_epochs,
            init_samples=args.init_samples,
            silhouette_samples=args.silhouette_samples,
            seed=args.seed,
        )
    else:
        evaluate_latent_kmeans(
            exp_config_path=args.exp_config,
            checkpoint=args.checkpoint,
            test_key=args.test_key,
            slice_idx=args.slice_idx,
            output_dir=args.output_dir,
            pooling=args.pooling,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            disable_amp=args.disable_amp,
            n_clusters=args.n_clusters,
            standardize=not args.no_standardize,
            pca_components=args.pca_components,
            pca_fit_samples=args.pca_fit_samples,
            kmeans_batch_size=args.kmeans_batch_size,
            kmeans_epochs=args.kmeans_epochs,
            init_samples=args.init_samples,
            silhouette_samples=args.silhouette_samples,
            seed=args.seed,
            keep_features=args.keep_features,
            feature_temp_dir=args.feature_temp_dir,
        )


if __name__ == "__main__":
    main()
