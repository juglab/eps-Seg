"""`prepare_data` must never rebuild a cache that already exists.

Every fixture here is synthetic and lives in `tmp_path`. No test in this file reads, writes or
even names a real dataset, cache or experiment config a rebuild triggered against a shared
cache is exactly the failure this guard exists to prevent, so the tests must not be able to
reach one even if the guard regresses.
"""

import os

import pytest
import yaml

from eps_seg.config.datasets import BetaSegDatasetConfig
from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.caching import CacheValidationError, DatasetCache
from eps_seg.dataloaders.datamodules import EPSSegDataModule

STUB_FILES = ("substacks.csv", "sampled_coords.csv", "fold_assignments.csv", "fold_stats.csv")


def _dataset_cfg(tmp_path, **overrides):
    fields = dict(
        type="BetaSegDatasetConfig",
        name="synthetic_ds",
        dim=2,
        data_dir=str(tmp_path / "data"),
        cache_root=str(tmp_path / "cache"),
        fold=0,
        max_folds=2,
        n_classes=4,
        patch_size=16,
        substacking=1,
        samples_per_class={0: 1, 1: 1, 2: 1, 3: 1},
        train_keys=["synthetic_a"],
        test_keys=["synthetic_b"],
    )
    fields.update(overrides)
    fields.pop("type", None)
    return BetaSegDatasetConfig(**fields)


def _build_synthetic_cache(cfg, train_cfg=None):
    """Hand-write a cache that `check_cache_dir` accepts. Manifest comes from build_manifest()."""
    cache = DatasetCache(cfg, train_cfg=train_cfg)
    root = cache.cache_dir
    root.mkdir(parents=True, exist_ok=True)
    for name in STUB_FILES:
        (root / name).write_text("stub\n", encoding="utf-8")
    with open(root / "manifest.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cache.build_manifest(), fh)
    for fold in range(cfg.max_folds):
        for sub in ("normalized", "labels"):
            d = root / f"fold_{fold}" / sub
            d.mkdir(parents=True, exist_ok=True)
            for key in cfg.train_keys:
                (d / f"{key}.tif").write_bytes(b"")
    return cache


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _datamodule(cfg, monkeypatch):
    """Datamodule whose build_cache is replaced by a recorder, so no build can ever run."""
    train_cfg = TrainConfig(use_wandb=False)
    dm = EPSSegDataModule(cfg=cfg, train_cfg=train_cfg)
    builds = []
    monkeypatch.setattr(dm.cache, "build_cache", lambda: builds.append(True))
    return dm, builds


def test_valid_cache_is_reused_and_never_rebuilt(tmp_path, monkeypatch):
    cfg = _dataset_cfg(tmp_path)
    cache = _build_synthetic_cache(cfg, TrainConfig(use_wandb=False))
    before = _snapshot(cache.cache_dir)

    dm, builds = _datamodule(cfg, monkeypatch)
    dm.prepare_data()

    assert builds == []
    assert _snapshot(cache.cache_dir) == before


def test_absent_cache_is_built(tmp_path, monkeypatch):
    cfg = _dataset_cfg(tmp_path)
    dm, builds = _datamodule(cfg, monkeypatch)
    assert not dm.cache_dir.exists()

    dm.prepare_data()

    assert builds == [True], "a cache that does not exist yet must still be built"


@pytest.mark.parametrize(
    "mutation",
    [
        {"seed": 1234},
        {"substacking": 7},
        {"max_patch_size": [8, 16, 16]},
        {"train_to_val_ratio": 0.5},
        {"samples_per_class": {0: 9, 1: 9, 2: 9, 3: 9}},
        {"train_keys": ["synthetic_a", "synthetic_c"]},
    ],
)
def test_mismatched_cache_raises_and_touches_nothing(tmp_path, monkeypatch, mutation):
    """Every manifest key that check_cache_dir compares must abort, not rebuild."""
    cfg = _dataset_cfg(tmp_path)
    cache = _build_synthetic_cache(cfg, TrainConfig(use_wandb=False))
    before = _snapshot(cache.cache_dir)

    changed = _dataset_cfg(tmp_path, **mutation)
    # the mutation must not move the cache folder, or this would test the "absent" path instead
    if DatasetCache(changed).cache_dir != cache.cache_dir:
        pytest.skip(f"{mutation} changes the cache folder name; covered by the absent-cache test")

    dm, builds = _datamodule(changed, monkeypatch)
    with pytest.raises(CacheValidationError) as excinfo:
        dm.prepare_data()

    assert "Refusing to rebuild" in str(excinfo.value)
    assert builds == [], "a mismatched cache must never be rebuilt"
    assert _snapshot(cache.cache_dir) == before, "the existing cache must be left byte-identical"


def test_missing_fold_artifact_raises_and_touches_nothing(tmp_path, monkeypatch):
    cfg = _dataset_cfg(tmp_path)
    cache = _build_synthetic_cache(cfg, TrainConfig(use_wandb=False))
    (cache.cache_dir / "fold_1" / "labels" / "synthetic_a.tif").unlink()
    before = _snapshot(cache.cache_dir)

    dm, builds = _datamodule(cfg, monkeypatch)
    with pytest.raises(CacheValidationError):
        dm.prepare_data()

    assert builds == []
    assert _snapshot(cache.cache_dir) == before


def test_empty_cache_directory_raises_rather_than_rebuilding(tmp_path, monkeypatch):
    """A half-built or emptied cache dir must fail loudly, not silently rebuild over itself."""
    cfg = _dataset_cfg(tmp_path)
    DatasetCache(cfg).cache_dir.mkdir(parents=True)

    dm, builds = _datamodule(cfg, monkeypatch)
    with pytest.raises(CacheValidationError):
        dm.prepare_data()

    assert builds == []


def test_guard_does_not_follow_symlinked_folds_into_a_rebuild(tmp_path, monkeypatch):
    """Mirrors the real hazard: fold_* symlinked into a parent cache, with a mismatched manifest."""
    parent_cfg = _dataset_cfg(tmp_path, name="synthetic_parent")
    parent = _build_synthetic_cache(parent_cfg, TrainConfig(use_wandb=False))

    child_cfg = _dataset_cfg(tmp_path, name="synthetic_child")
    child = DatasetCache(child_cfg).cache_dir
    child.mkdir(parents=True)
    for name in STUB_FILES:
        (child / name).write_text("stub\n", encoding="utf-8")
    with open(child / "manifest.yaml", "w", encoding="utf-8") as fh:
        bad = DatasetCache(child_cfg).build_manifest()
        bad["seed"] = 999  # deliberately mismatched
        yaml.safe_dump(bad, fh)
    for fold in range(child_cfg.max_folds):
        os.symlink(parent.cache_dir / f"fold_{fold}", child / f"fold_{fold}")

    parent_before = _snapshot(parent.cache_dir)
    dm, builds = _datamodule(child_cfg, monkeypatch)
    with pytest.raises(CacheValidationError):
        dm.prepare_data()

    assert builds == []
    assert _snapshot(parent.cache_dir) == parent_before, "the parent cache must be untouched"
