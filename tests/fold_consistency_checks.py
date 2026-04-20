from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from eps_seg.config.datasets import BaseEPSDatasetConfig
from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.datamodules import EPSSegDataModule


@dataclass
class FoldCheckContext:
    cfg: BaseEPSDatasetConfig
    cache_dir: Path
    substacks_df: pd.DataFrame
    sampled_coords_df: pd.DataFrame
    fold_assignments_df: pd.DataFrame
    fold_stats_df: pd.DataFrame


def build_context(dataset_config_path: str | Path, build_cache: bool = True) -> FoldCheckContext:
    cfg = BaseEPSDatasetConfig.from_yaml(Path(dataset_config_path))
    dm = EPSSegDataModule(
        cfg=cfg,
        train_cfg=TrainConfig(batch_size=32, batches_per_pseudoepoch=4, initial_radius=3),
    )

    if build_cache:
        dm.prepare_data()

    cache_dir = dm.cache_dir
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist.")

    substacks_df = pd.read_csv(cache_dir / "substacks.csv")
    sampled_coords_df = pd.read_csv(cache_dir / "sampled_coords.csv")
    fold_assignments_df = pd.read_csv(cache_dir / "fold_assignments.csv")
    fold_stats_df = pd.read_csv(cache_dir / "fold_stats.csv")

    return FoldCheckContext(
        cfg=cfg,
        cache_dir=cache_dir,
        substacks_df=substacks_df,
        sampled_coords_df=sampled_coords_df,
        fold_assignments_df=fold_assignments_df,
        fold_stats_df=fold_stats_df,
    )


def run_all_checks(ctx: FoldCheckContext) -> Dict[str, str]:
    checks = [
        ("cache files exist", check_cache_files_exist),
        ("substack descriptors are valid", check_substack_descriptors),
        ("substacks do not overlap within a stack", check_substacks_do_not_overlap),
        ("canonical coordinates are unique", check_canonical_coords_are_unique),
        ("canonical coordinates belong to their substacks", check_coords_within_substack_bounds),
        ("each fold assigns every substack exactly once", check_each_fold_assigns_every_substack_once),
        ("train and val substacks are disjoint", check_train_val_substacks_are_disjoint),
        ("each substack is val once and train otherwise", check_kfold_validation_occurrences),
        ("canonical coordinate union is identical across folds", check_canonical_union_same_for_each_fold),
        ("train and val coordinates are disjoint within each fold", check_train_val_coords_are_disjoint),
        ("every fold covers every source stack", check_each_fold_covers_all_source_stacks),
        ("fold statistics match cached coordinates", check_fold_stats_match_cached_counts),
    ]

    results: Dict[str, str] = {}
    for name, fn in checks:
        fn(ctx)
        results[name] = "PASS"
    return results


def check_cache_files_exist(ctx: FoldCheckContext) -> None:
    required = [
        ctx.cache_dir / "manifest.yaml",
        ctx.cache_dir / "substacks.csv",
        ctx.cache_dir / "sampled_coords.csv",
        ctx.cache_dir / "fold_assignments.csv",
        ctx.cache_dir / "fold_stats.csv",
    ]
    for fold in range(ctx.cfg.max_folds):
        for key in ctx.cfg.train_keys:
            required.append(ctx.cache_dir / f"fold_{fold}" / "normalized" / f"{key}.tif")
            required.append(ctx.cache_dir / f"fold_{fold}" / "labels" / f"{key}.tif")

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Missing cache files:\n" + "\n".join(missing)


def check_substack_descriptors(ctx: FoldCheckContext) -> None:
    df = ctx.substacks_df
    required_cols = {"substack_id", "stack_name", "z_start", "z_stop", "depth"}
    assert required_cols.issubset(df.columns), f"Missing substack columns: {required_cols - set(df.columns)}"
    assert df["substack_id"].is_unique, "substack_id must be unique."
    assert ((df["z_stop"] - df["z_start"]) == df["depth"]).all(), "Substack depth mismatch."
    assert (df["depth"] > 0).all(), "All substacks must have positive depth."


def check_substacks_do_not_overlap(ctx: FoldCheckContext) -> None:
    for stack_name, group in ctx.substacks_df.sort_values(["stack_name", "z_start"]).groupby("stack_name"):
        prev_stop = None
        for row in group.itertuples(index=False):
            if prev_stop is not None:
                assert row.z_start >= prev_stop, (
                    f"Overlapping substacks found in stack {stack_name}: "
                    f"previous stop={prev_stop}, current start={row.z_start}"
                )
            prev_stop = row.z_stop


def check_canonical_coords_are_unique(ctx: FoldCheckContext) -> None:
    df = ctx.sampled_coords_df
    assert df["coord_id"].is_unique, "coord_id must be unique."
    coord_cols = ["stack_name", "z", "y", "x"]
    assert not df.duplicated(coord_cols).any(), "Canonical coordinates contain duplicates."


def check_coords_within_substack_bounds(ctx: FoldCheckContext) -> None:
    merged = ctx.sampled_coords_df.merge(
        ctx.substacks_df[["substack_id", "stack_name", "z_start", "z_stop"]],
        on=["substack_id", "stack_name"],
        how="left",
        validate="many_to_one",
    )
    assert not merged[["z_start", "z_stop"]].isna().any().any(), "Some coordinates reference missing substacks."
    inside = (merged["z"] >= merged["z_start"]) & (merged["z"] < merged["z_stop"])
    assert inside.all(), "Some canonical coordinates fall outside the bounds of their substacks."


def check_each_fold_assigns_every_substack_once(ctx: FoldCheckContext) -> None:
    expected_substack_ids = set(ctx.substacks_df["substack_id"].tolist())
    for fold, group in ctx.fold_assignments_df.groupby("fold"):
        assigned_ids = set(group["substack_id"].tolist())
        assert assigned_ids == expected_substack_ids, f"Fold {fold} does not cover all substacks."
        counts = group["substack_id"].value_counts()
        assert (counts == 1).all(), f"Fold {fold} assigns some substacks more than once."


def check_train_val_substacks_are_disjoint(ctx: FoldCheckContext) -> None:
    for fold, group in ctx.fold_assignments_df.groupby("fold"):
        train_ids = set(group.loc[group["assigned_split"] == "train", "substack_id"])
        val_ids = set(group.loc[group["assigned_split"] == "val", "substack_id"])
        overlap = train_ids & val_ids
        assert not overlap, f"Fold {fold} has substacks assigned to both train and val: {sorted(overlap)}"


def check_kfold_validation_occurrences(ctx: FoldCheckContext) -> None:
    counts = (
        ctx.fold_assignments_df.groupby(["substack_id", "assigned_split"])
        .size()
        .unstack(fill_value=0)
    )

    assert "val" in counts.columns, "No validation assignments found in fold_assignments.csv."
    assert (counts["val"] == 1).all(), "Each substack must appear in validation exactly once."

    if ctx.cfg.max_folds > 1:
        assert "train" in counts.columns, "No training assignments found in fold_assignments.csv."
        expected_train_occurrences = ctx.cfg.max_folds - 1
        assert (counts["train"] == expected_train_occurrences).all(), (
            "Each substack must appear in training exactly max_folds - 1 times."
        )


def check_canonical_union_same_for_each_fold(ctx: FoldCheckContext) -> None:
    canonical_coords = set(
        map(tuple, ctx.sampled_coords_df[["stack_name", "z", "y", "x"]].itertuples(index=False, name=None))
    )
    for fold, group in ctx.fold_assignments_df.groupby("fold"):
        fold_substack_ids = set(group["substack_id"])
        fold_coords_df = ctx.sampled_coords_df[ctx.sampled_coords_df["substack_id"].isin(fold_substack_ids)]
        fold_coords = set(
            map(tuple, fold_coords_df[["stack_name", "z", "y", "x"]].itertuples(index=False, name=None))
        )
        assert fold_coords == canonical_coords, f"Fold {fold} does not recover the full canonical coordinate set."


def check_train_val_coords_are_disjoint(ctx: FoldCheckContext) -> None:
    merged = ctx.sampled_coords_df.merge(
        ctx.fold_assignments_df,
        on="substack_id",
        how="left",
        validate="many_to_many",
    )
    for fold, group in merged.groupby("fold"):
        train_coords = set(
            map(tuple, group.loc[group["assigned_split"] == "train", ["stack_name", "z", "y", "x"]].itertuples(index=False, name=None))
        )
        val_coords = set(
            map(tuple, group.loc[group["assigned_split"] == "val", ["stack_name", "z", "y", "x"]].itertuples(index=False, name=None))
        )
        overlap = train_coords & val_coords
        assert not overlap, f"Fold {fold} has coordinates appearing in both train and val."


def check_each_fold_covers_all_source_stacks(ctx: FoldCheckContext) -> None:
    expected_stacks = set(ctx.cfg.train_keys)
    merged = ctx.sampled_coords_df.merge(
        ctx.fold_assignments_df,
        on="substack_id",
        how="left",
        validate="many_to_many",
    )
    for fold, group in merged.groupby("fold"):
        train_stacks = set(group.loc[group["assigned_split"] == "train", "stack_name"])
        val_stacks = set(group.loc[group["assigned_split"] == "val", "stack_name"])
        assert train_stacks == expected_stacks, f"Fold {fold} train split does not cover all source stacks."
        assert val_stacks == expected_stacks, f"Fold {fold} val split does not cover all source stacks."


def check_fold_stats_match_cached_counts(ctx: FoldCheckContext) -> None:
    merged = ctx.sampled_coords_df.merge(
        ctx.fold_assignments_df,
        on="substack_id",
        how="left",
        validate="many_to_many",
    )
    stats_by_fold = ctx.fold_stats_df.set_index("fold")

    for fold, group in merged.groupby("fold"):
        stats_row = stats_by_fold.loc[fold]
        train_group = group[group["assigned_split"] == "train"]
        val_group = group[group["assigned_split"] == "val"]

        assert int(stats_row["n_train_coords"]) == len(train_group), f"Fold {fold} train coord count mismatch."
        assert int(stats_row["n_val_coords"]) == len(val_group), f"Fold {fold} val coord count mismatch."

        for class_idx in range(ctx.cfg.n_classes):
            train_count = int((train_group["gt_label"] == class_idx).sum())
            val_count = int((val_group["gt_label"] == class_idx).sum())
            assert int(stats_row[f"train_class_{class_idx}_count"]) == train_count, (
                f"Fold {fold} train class {class_idx} count mismatch."
            )
            assert int(stats_row[f"val_class_{class_idx}_count"]) == val_count, (
                f"Fold {fold} val class {class_idx} count mismatch."
            )

        for stack_name in ctx.cfg.train_keys:
            train_count = int((train_group["stack_name"] == stack_name).sum())
            val_count = int((val_group["stack_name"] == stack_name).sum())
            assert int(stats_row[f"train_stack_{stack_name}_count"]) == train_count, (
                f"Fold {fold} train stack {stack_name} count mismatch."
            )
            assert int(stats_row[f"val_stack_{stack_name}_count"]) == val_count, (
                f"Fold {fold} val stack {stack_name} count mismatch."
            )
