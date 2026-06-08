import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
import torch
from torchmetrics.classification import F1Score

from eps_seg.config.datasets import BaseEPSDatasetConfig, BetaSegDatasetConfig
from eps_seg.config.train import ExperimentConfig


def _resolve_default_slice_idx(
    dataset_cfg: BaseEPSDatasetConfig,
    key: str,
    gt_volume,
    override_slice_idx: Optional[int] = None,
) -> int:
    """
    Resolve the evaluation z-index for one test volume.

    Priority:
    1. CLI override
    2. dataset_cfg.test_center_slices for the matching key
    3. BetaSeg default slice 626
    4. middle z slice of the ground-truth volume
    """
    if override_slice_idx is not None:
        return int(override_slice_idx)

    if key in dataset_cfg.test_keys:
        key_idx = dataset_cfg.test_keys.index(key)
        if key_idx < len(dataset_cfg.test_center_slices):
            cfg_slice = dataset_cfg.test_center_slices[key_idx]
            if cfg_slice is not None:
                return int(cfg_slice)

    if isinstance(dataset_cfg, BetaSegDatasetConfig):
        return 626

    return int(gt_volume.shape[0] // 2)


def _compute_dice_rows(
    experiment_name: str,
    checkpoint_name: str,
    test_key: str,
    slice_idx: int,
    gt_slice,
    pred_slice,
    n_classes: int,
) -> dict:
    gt_arr = np.asarray(gt_slice)
    pred_arr = np.asarray(pred_slice)

    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch for test_key='{test_key}' at z={slice_idx}: "
            f"pred shape {pred_arr.shape}, gt shape {gt_arr.shape}."
        )

    gt_flat = gt_arr.reshape(-1)
    pred_flat = pred_arr.reshape(-1)

    valid_gt_mask = (gt_flat >= 0) & (gt_flat < n_classes)
    if not np.any(valid_gt_mask):
        raise ValueError(
            f"Ground-truth slice for test_key='{test_key}' at z={slice_idx} contains only ignore labels."
        )

    valid_pred_mask = np.isfinite(pred_flat) & (pred_flat >= 0) & (pred_flat < n_classes)
    valid_mask = valid_gt_mask & valid_pred_mask
    if not np.any(valid_mask):
        raise ValueError(
            f"Prediction slice for test_key='{test_key}' at z={slice_idx} contains no valid class labels "
            f"in [0, {n_classes - 1}] where GT is defined."
        )

    gtf = torch.as_tensor(gt_flat[valid_mask], dtype=torch.long)
    prf = torch.as_tensor(pred_flat[valid_mask], dtype=torch.long)

    dice_score = F1Score(
        num_classes=n_classes,
        average=None,
        task="multiclass",
    )
    dsc_per_class = dice_score(prf, gtf)
    avg_dsc = dsc_per_class.mean().item()
    dice_score.reset()

    row = {
        "experiment": experiment_name,
        "checkpoint": checkpoint_name,
        "test_key": test_key,
        "slice_idx": slice_idx,
        "average_dice_score": avg_dsc,
        "n_valid_voxels": int(valid_mask.sum()),
        "n_ignored_gt_voxels": int((~valid_gt_mask).sum()),
        "n_invalid_pred_voxels": int((valid_gt_mask & ~valid_pred_mask).sum()),
    }
    for class_idx, dsc in enumerate(dsc_per_class):
        row[f"dice_score_class_{class_idx}"] = dsc.item()
    return row


def _load_experiment_predictions(exp: ExperimentConfig) -> dict[str, Path]:
    predictions_root = exp.outputs_dir / "predictions"
    if not predictions_root.exists():
        return {}

    return {
        checkpoint_dir.name: checkpoint_dir
        for checkpoint_dir in sorted(predictions_root.iterdir())
        if checkpoint_dir.is_dir()
    }


def _plot_prediction_grid(
    out_path: Path,
    predictions_by_experiment: dict[str, dict[str, object]],
) -> None:
    if not predictions_by_experiment:
        return

    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.2,
        rc={"axes.linewidth": 0.5},
    )

    experiments = list(predictions_by_experiment.keys())
    checkpoints = sorted(
        {
            checkpoint_name
            for predictions in predictions_by_experiment.values()
            for checkpoint_name in predictions.keys()
        }
    )
    if not checkpoints:
        return

    fig, axes = plt.subplots(
        len(experiments),
        len(checkpoints),
        figsize=(4 * len(checkpoints), 4 * len(experiments)),
        squeeze=False,
    )

    for i, experiment_name in enumerate(experiments):
        experiment_predictions = predictions_by_experiment[experiment_name]
        for j, checkpoint_name in enumerate(checkpoints):
            ax = axes[i, j]
            pred_slice = experiment_predictions.get(checkpoint_name)
            if pred_slice is not None:
                ax.imshow(pred_slice)
            ax.set_title(f"{experiment_name}\n{checkpoint_name}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def evaluate(
    exp_root: str | Path,
    slice_idx: Optional[int] = None,
    results_dir: Optional[str | Path] = None,
) -> None:
    exp_root = Path(exp_root)
    ablation_name = exp_root.name
    exp_yaml_files = sorted(exp_root.glob("exp_*.yaml"))
    if not exp_yaml_files:
        raise FileNotFoundError(f"No experiment YAML files matching 'exp_*.yaml' were found in {exp_root}.")

    output_base = Path(results_dir) if results_dir is not None else exp_root
    out_result_folder = output_base / "results"
    out_result_folder.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    predictions_for_plot: dict[str, dict[str, object]] = {}

    for exp_yaml in exp_yaml_files:
        exp = ExperimentConfig.from_yaml(exp_yaml)
        _, dataset_cfg, _ = exp.get_configs()
        experiment_name = exp.experiment_name
        checkpoint_dirs = _load_experiment_predictions(exp)

        if not checkpoint_dirs:
            print(f"No prediction folders found for {experiment_name}. Skipping.")
            continue

        test_paths = dataset_cfg.get_image_label_paths(dataset_cfg.test_keys)
        predictions_for_plot.setdefault(experiment_name, {})

        for test_key in dataset_cfg.test_keys:
            _, gt_path = test_paths[test_key]
            gt_volume = tiff.imread(gt_path)
            eval_slice_idx = _resolve_default_slice_idx(
                dataset_cfg=dataset_cfg,
                key=test_key,
                gt_volume=gt_volume,
                override_slice_idx=slice_idx,
            )
            gt_slice = gt_volume[eval_slice_idx]

            for checkpoint_name, checkpoint_dir in checkpoint_dirs.items():
                pred_path = checkpoint_dir / f"{test_key}.tif"
                if not pred_path.exists():
                    print(
                        f"Prediction not found for experiment={experiment_name}, "
                        f"checkpoint={checkpoint_name}, test_key={test_key}."
                    )
                    continue

                pred_volume = tiff.imread(pred_path)
                pred_slice = pred_volume[eval_slice_idx]

                print(
                    f"Evaluating experiment={experiment_name}, checkpoint={checkpoint_name}, "
                    f"test_key={test_key}, slice={eval_slice_idx}"
                )
                try:
                    all_rows.append(
                        _compute_dice_rows(
                            experiment_name=experiment_name,
                            checkpoint_name=checkpoint_name,
                            test_key=test_key,
                            slice_idx=eval_slice_idx,
                            gt_slice=gt_slice,
                            pred_slice=pred_slice,
                            n_classes=dataset_cfg.n_classes,
                        )
                    )
                except ValueError as exc:
                    print(
                        f"Skipping experiment={experiment_name}, checkpoint={checkpoint_name}, "
                        f"test_key={test_key}, slice={eval_slice_idx}: {exc}"
                    )
                    continue

                # Store one representative slice per experiment/checkpoint for the summary figure.
                predictions_for_plot[experiment_name].setdefault(checkpoint_name, pred_slice)

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        raise RuntimeError(f"No predictions were evaluated under {exp_root}.")

    result_csv_path = out_result_folder / f"{ablation_name}_results.csv"
    results_df.to_csv(result_csv_path, index=False)
    print(f"Saved evaluation CSV to {result_csv_path}")

    figure_path = out_result_folder / f"{ablation_name}.png"
    _plot_prediction_grid(figure_path, predictions_for_plot)
    print(f"Saved prediction summary figure to {figure_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions for eps-Seg experiments.")
    parser.add_argument(
        "--exp_root",
        type=str,
        required=True,
        help="Path to the experiment root directory containing exp_*.yaml files.",
    )
    parser.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="Optional z slice to evaluate. If omitted, use the dataset-config center slice, BetaSeg default (626), or the middle slice.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional base directory where a results/ folder will be created. If omitted, use exp_root/results.",
    )
    args = parser.parse_args()
    evaluate(
        exp_root=args.exp_root,
        slice_idx=args.slice_idx,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
