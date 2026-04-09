import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from eps_seg.config.train import ExperimentConfig
from torchmetrics.classification import F1Score
import torch
import argparse

PRED_SLICE = 626
PRED_KEY = "high_c4.tif"
GT_TIFF = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_gt.tif"
INPUT_TIFF = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_source.tif"

LEGACY_CHECKPOINTS = ("best_supervised", "best_semisupervised")
CHECKPOINT_PATTERN = re.compile(
    r"^(?P<kind>best|last)_(?P<mode>supervised|semisupervised)(?:_K(?P<stage>\d+))?$"
)


def _checkpoint_sort_key(checkpoint_name: str) -> tuple[int, int, int]:
    """
    Return a stable sort key for prediction folders.

    Staged checkpoints are ordered by mode, then stage index, then best/last.
    Legacy checkpoint names are kept supported and sorted after staged folders of the
    same mode by assigning them stage -1.
    """
    match = CHECKPOINT_PATTERN.match(checkpoint_name)
    if match is None:
        return (99, 99_999, 99)

    mode_order = {"semisupervised": 0, "supervised": 1}
    kind_order = {"best": 0, "last": 1}
    stage_idx = int(match.group("stage")) if match.group("stage") is not None else -1
    return (
        mode_order[match.group("mode")],
        stage_idx,
        kind_order[match.group("kind")],
    )


def _discover_prediction_checkpoints(predictions_root: Path) -> list[str]:
    """
    Discover prediction subfolders produced by the current checkpoint naming scheme.

    The staged training code writes predictions under folders matching the checkpoint
    stem, e.g. ``best_semisupervised_K0``. Older runs may still use the legacy names
    ``best_supervised`` and ``best_semisupervised``. We support both schemes.
    """
    if not predictions_root.exists():
        return []

    discovered = []
    for child in predictions_root.iterdir():
        if not child.is_dir():
            continue
        if CHECKPOINT_PATTERN.match(child.name) is None:
            continue
        if not (child / PRED_KEY).exists():
            continue
        discovered.append(child.name)

    if discovered:
        return sorted(discovered, key=_checkpoint_sort_key)

    return [name for name in LEGACY_CHECKPOINTS if (predictions_root / name / PRED_KEY).exists()]

def evaluate(exp_root: str):
    ablation_name = str(Path(exp_root).name)
    exp_yaml_files = sorted(list(str(p) for p in Path(exp_root).glob("exp_*.yaml")))
    experiment_predictions_folders = {}
    experiment_checkpoint_names = {}
    for exp_yaml in exp_yaml_files:
        exp = ExperimentConfig.from_yaml(str(exp_yaml))
        predictions_root = exp.outputs_dir / "predictions"
        experiment_predictions_folders[exp.experiment_name] = predictions_root
        experiment_checkpoint_names[exp.experiment_name] = _discover_prediction_checkpoints(predictions_root)

    # Load ground truth and predictions
    input_image = tiff.imread(INPUT_TIFF)[PRED_SLICE]
    gt_image = tiff.imread(GT_TIFF)[PRED_SLICE]
    predictions = {}

    
    out_result_folder = Path("./results/") / ablation_name 
    out_result_folder.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(columns=["Experiment", "Checkpoint"])

    for experiment, pred_folder in experiment_predictions_folders.items():

        predictions[experiment] = {}
        checkpoint_names = experiment_checkpoint_names[experiment]

        for ckpt in checkpoint_names:
            pred_path = Path(pred_folder) / ckpt / PRED_KEY

            if pred_path.exists():
                print(f"Evaluating {experiment} at {ckpt}")
                pred_image = tiff.imread(pred_path)[PRED_SLICE]
                predictions[experiment][ckpt] = pred_image

                gtf = torch.tensor(gt_image.flatten())
                prf = torch.tensor(pred_image.flatten())
                mask = torch.logical_and(gtf != -1, prf != -1)
                dice_score = F1Score(num_classes=len(gtf[mask].unique()), average=None, task="multiclass", ignore_index=-1)
                dsc_per_class = dice_score(prf[mask].int(), gtf[mask].int())
                avg_dsc = dsc_per_class.mean().item()

                new_row = {
                    "Experiment": experiment,
                    "Checkpoint": ckpt,
                    "Average Dice Score": avg_dsc
                }
                for class_idx, dsc in enumerate(dsc_per_class):
                    new_row[f"DSC_Class_{class_idx}"] = dsc.item()

                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                
                results_df.to_csv(out_result_folder / f"results_{PRED_SLICE}.csv", index=False)

                dice_score.reset()
            else:
                print(f"Prediction not found for {experiment} at {ckpt}")

    # Save figures for each ckpt 
    all_checkpoints = sorted(
        {checkpoint for preds in predictions.values() for checkpoint in preds},
        key=_checkpoint_sort_key,
    )

    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.2,
        rc={"axes.linewidth": 0.5}
    )

    # Plot predictions in a table, row per experiment, column per checkpoint
    n_rows = len(experiment_predictions_folders)
    n_cols = max(1, len(all_checkpoints))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

    for i, (experiment, preds) in enumerate(predictions.items()):
        for j, ckpt in enumerate(all_checkpoints):
            ax = axes[i, j]
            if ckpt in preds:
                ax.imshow(preds[ckpt])
            ax.set_title(f"{experiment}\n{ckpt}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_result_folder / f"{ablation_name}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions Eps-Seg experiments.")
    parser.add_argument("--exp_root", type=str, required=True, help="Path to the root directory of the experiments.")
    args = parser.parse_args()
    evaluate(args.exp_root)
