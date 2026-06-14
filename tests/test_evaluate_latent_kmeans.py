import numpy as np
import pytest
import tifffile
import torch

from eps_seg.evaluate_latent_kmeans import (
    _transform_features,
    align_clusters_to_ground_truth,
    dice_scores,
    fit_kmeans_streaming,
    load_tiff_array,
    pool_latent_mus,
    save_pca_artifacts,
)


def test_pool_latent_mus_concatenates_layer_centers():
    lower = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
    upper = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

    features = pool_latent_mus([lower, upper], "center")

    assert features.shape == (2, 5)
    assert torch.equal(features[:, :2], lower[:, :, 2, 2])
    assert torch.equal(features[:, 2:], upper[:, :, 1, 1])


def test_pool_latent_mus_supports_mean_and_flatten():
    mus = [torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)]

    mean_features = pool_latent_mus(mus, "mean")
    flat_features = pool_latent_mus(mus, "flatten")

    assert mean_features.shape == (2, 3)
    assert flat_features.shape == (2, 12)
    assert torch.allclose(mean_features, mus[0].flatten(start_dim=2).mean(dim=2))


def test_cluster_alignment_recovers_permuted_ground_truth():
    gt = np.array([0, 0, 1, 1, 2, 2])
    clusters = np.array([2, 2, 0, 0, 1, 1])

    aligned, mapping, contingency = align_clusters_to_ground_truth(
        clusters,
        gt,
        n_clusters=3,
        n_classes=3,
    )

    assert mapping == {0: 1, 1: 2, 2: 0}
    assert np.array_equal(aligned, gt)
    assert contingency.sum() == len(gt)


def test_dice_scores_reports_macro_and_per_class_values():
    gt = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])

    average, per_class = dice_scores(predicted, gt, n_classes=2)

    assert per_class[0] == pytest.approx(2 / 3)
    assert per_class[1] == pytest.approx(4 / 5)
    assert average == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_dice_scores_counts_unmatched_clusters_as_errors():
    gt = np.array([0, 0, 1, 1])
    predicted = np.array([0, -1, 1, -1])

    average, per_class = dice_scores(predicted, gt, n_classes=2)

    assert per_class == pytest.approx({0: 2 / 3, 1: 2 / 3})
    assert average == pytest.approx(2 / 3)


def test_streaming_kmeans_supports_pca():
    rng = np.random.default_rng(4)
    features = np.concatenate(
        [
            rng.normal(-2.0, 0.1, size=(20, 6)),
            rng.normal(2.0, 0.1, size=(20, 6)),
        ]
    ).astype(np.float32)

    labels, kmeans, scaler, pca = fit_kmeans_streaming(
        features,
        n_clusters=2,
        standardize=True,
        pca_components=2,
        pca_fit_samples=20,
        batch_size=16,
        epochs=1,
        init_samples=20,
        seed=7,
    )

    assert labels.shape == (40,)
    assert len(np.unique(labels)) == 2
    assert kmeans.cluster_centers_.shape == (2, 2)
    assert scaler is not None
    assert pca is not None
    assert pca.n_components_ == 2


def test_save_pca_artifacts_writes_reusable_features(tmp_path):
    rng = np.random.default_rng(8)
    features = rng.normal(size=(12, 5)).astype(np.float32)
    _, _, scaler, pca = fit_kmeans_streaming(
        features,
        n_clusters=2,
        standardize=True,
        pca_components=3,
        pca_fit_samples=10,
        batch_size=6,
        epochs=0,
        init_samples=8,
        seed=2,
    )
    assert pca is not None

    output_path = save_pca_artifacts(
        features,
        scaler=scaler,
        pca=pca,
        output_dir=tmp_path,
        batch_size=6,
        gt_labels=np.arange(12) % 2,
        coords=np.column_stack(
            [np.zeros(12), np.arange(12), np.arange(12)]
        ),
        direct_labels=np.arange(12) % 2,
        gt_slice=np.zeros((12, 12)),
        metadata={
            "checkpoint": np.asarray("model.ckpt"),
            "test_key": np.asarray("test"),
            "slice_idx": np.asarray(4),
            "pooling": np.asarray("flatten"),
            "standardized": np.asarray(True),
            "feature_dim": np.asarray(5),
            "pca_explained_variance_ratio": np.asarray(
                pca.explained_variance_ratio_.sum()
            ),
        },
    )

    saved = np.load(output_path, mmap_mode="r")
    expected = _transform_features(features, scaler=scaler, pca=pca)
    assert np.allclose(saved, expected)
    assert (tmp_path / "pca_transform.npz").exists()
    assert (tmp_path / "pca_features_metadata.npz").exists()


def test_load_tiff_array_falls_back_for_compressed_tiff(tmp_path):
    expected = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    path = tmp_path / "compressed.tif"
    tifffile.imwrite(
        path,
        expected,
        compression="zlib",
        photometric="minisblack",
    )

    loaded = load_tiff_array(path)

    assert np.array_equal(loaded, expected)


def test_load_tiff_array_opens_contiguous_tiff_read_only(tmp_path):
    expected = np.arange(5 * 6, dtype=np.int8).reshape(5, 6)
    path = tmp_path / "read_only.tif"
    tifffile.imwrite(path, expected)
    path.chmod(0o444)

    loaded = load_tiff_array(path)

    assert isinstance(loaded, np.memmap)
    assert loaded.mode == "r"
    assert np.array_equal(loaded, expected)
