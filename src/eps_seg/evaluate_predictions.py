import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff

from eps_seg.config.train import ExperimentConfig


PRED_SLICE = 626
PRED_KEY = "high_c4.tif"
GT_TIFF = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_gt.tif"

CHECKPOINT_PATTERN = re.compile(
    r"^(?P<kind>best|last)_(?P<mode>supervised|semisupervised)(?:_K(?P<stage>\d+))?$"
)


def checkpoint_sort_key(checkpoint_name: str) -> tuple[int, int, int]:
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


def discover_prediction_checkpoints(predictions_root: Path, pred_key: str) -> list[str]:
    if not predictions_root.exists():
        return []

    discovered = []
    for child in predictions_root.iterdir():
        if not child.is_dir():
            continue
        if CHECKPOINT_PATTERN.match(child.name) is None:
            continue
        if (child / pred_key).exists():
            discovered.append(child.name)
    return sorted(discovered, key=checkpoint_sort_key)


def selected_checkpoint(checkpoint_names: list[str]) -> str | None:
    if "best_semisupervised" in checkpoint_names:
        return "best_semisupervised"
    if "best_supervised" in checkpoint_names:
        return "best_supervised"
    return checkpoint_names[0] if checkpoint_names else None


def dice_scores(gt_image: np.ndarray, pred_image: np.ndarray) -> tuple[float, dict[int, float], int]:
    gt = gt_image.reshape(-1)
    pred = pred_image.reshape(-1)
    mask = (gt != -1) & (pred != -1)
    if not np.any(mask):
        return float("nan"), {}, 0

    gt_masked = gt[mask].astype(np.int64)
    pred_masked = pred[mask].astype(np.int64)
    labels = sorted(set(gt_masked.tolist()) | set(pred_masked.tolist()))
    per_class = {}
    for label in labels:
        gt_label = gt_masked == label
        pred_label = pred_masked == label
        denom = int(gt_label.sum() + pred_label.sum())
        if denom == 0:
            per_class[label] = float("nan")
        else:
            per_class[label] = float(2.0 * np.logical_and(gt_label, pred_label).sum() / denom)

    finite_scores = [value for value in per_class.values() if np.isfinite(value)]
    average = float(np.mean(finite_scores)) if finite_scores else float("nan")
    return average, per_class, int(mask.sum())


def read_slice(path: Path, pred_slice: int) -> np.ndarray:
    image = tiff.imread(path)
    if image.ndim < 3:
        return image
    return image[pred_slice]


def save_prediction_grid(
    predictions: dict[str, dict[str, np.ndarray]],
    checkpoint_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    if not predictions or not checkpoint_names:
        return

    sns.set_theme(context="paper", style="white", font_scale=1.0, rc={"axes.linewidth": 0.5})
    n_rows = len(predictions)
    n_cols = len(checkpoint_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(4, 3.2 * n_cols), max(4, 2.8 * n_rows)),
        squeeze=False,
    )

    for i, (experiment, preds) in enumerate(predictions.items()):
        for j, checkpoint in enumerate(checkpoint_names):
            ax = axes[i, j]
            if checkpoint in preds:
                ax.imshow(preds[checkpoint], interpolation="nearest", cmap="tab20")
            ax.set_title(f"{experiment}\n{checkpoint}", fontsize=7)
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_selected_barplot(results_df: pd.DataFrame, out_path: Path) -> None:
    selected = results_df[results_df["SelectedByRule"]].copy()
    if selected.empty:
        return

    selected = selected.sort_values("Average Dice Score", ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(selected))))
    sns.barplot(data=selected, y="Experiment", x="Average Dice Score", hue="Checkpoint", dodge=False, ax=ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Selected checkpoint prediction Dice")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate(
    exp_root: str,
    out_dir: str = "./results",
    pred_slice: int = PRED_SLICE,
    pred_key: str = PRED_KEY,
    gt_tiff: str = GT_TIFF,
) -> Path:
    exp_root_path = Path(exp_root)
    if not list(exp_root_path.glob("exp_*.yaml")) and (exp_root_path / "experiment_config").is_dir():
        exp_root_path = exp_root_path / "experiment_config"

    ablation_name = exp_root_path.name
    exp_yaml_files = sorted(exp_root_path.glob("exp_*.yaml"))
    if not exp_yaml_files:
        raise FileNotFoundError(
            f"No exp_*.yaml files found in {exp_root_path}. "
            "Pass the directory containing the experiment YAMLs, for example "
            "/group/jug/Sheida/Experiments/ablation_paper/experiment_config."
        )

    out_result_folder = Path(out_dir) / ablation_name
    out_result_folder.mkdir(parents=True, exist_ok=True)

    gt_image = read_slice(Path(gt_tiff), pred_slice)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    selected_predictions: dict[str, dict[str, np.ndarray]] = {}

    for exp_yaml in exp_yaml_files:
        exp = ExperimentConfig.from_yaml(str(exp_yaml))
        predictions_root = exp.outputs_dir / "predictions"
        checkpoint_names = discover_prediction_checkpoints(predictions_root, pred_key)
        selected = selected_checkpoint(checkpoint_names)
        predictions[exp.experiment_name] = {}

        if not checkpoint_names:
            rows.append(
                {
                    "Experiment": exp.experiment_name,
                    "Checkpoint": "",
                    "SelectedByRule": False,
                    "PredictionPath": "",
                    "Average Dice Score": np.nan,
                    "EvaluatedPixels": 0,
                    "Status": "missing_prediction",
                }
            )
            print(f"No predictions found for {exp.experiment_name}")
            continue

        for checkpoint in checkpoint_names:
            pred_path = predictions_root / checkpoint / pred_key
            print(f"Evaluating {exp.experiment_name} at {checkpoint}")
            pred_image = read_slice(pred_path, pred_slice)
            predictions[exp.experiment_name][checkpoint] = pred_image
            avg_dice, per_class, evaluated_pixels = dice_scores(gt_image, pred_image)

            row = {
                "Experiment": exp.experiment_name,
                "Checkpoint": checkpoint,
                "SelectedByRule": checkpoint == selected,
                "PredictionPath": str(pred_path),
                "Average Dice Score": avg_dice,
                "EvaluatedPixels": evaluated_pixels,
                "Status": "ok",
            }
            for class_idx, dice in per_class.items():
                row[f"DSC_Class_{class_idx}"] = dice
            rows.append(row)

        if selected is not None and selected in predictions[exp.experiment_name]:
            selected_predictions[exp.experiment_name] = {selected: predictions[exp.experiment_name][selected]}

    results_df = pd.DataFrame(rows)
    results_path = out_result_folder / f"results_{pred_slice}.csv"
    selected_results_path = out_result_folder / f"selected_results_{pred_slice}.csv"
    results_df.to_csv(results_path, index=False)
    if "SelectedByRule" in results_df.columns:
        results_df[results_df["SelectedByRule"]].to_csv(selected_results_path, index=False)
    else:
        pd.DataFrame().to_csv(selected_results_path, index=False)

    all_checkpoints = sorted(
        {checkpoint for preds in predictions.values() for checkpoint in preds},
        key=checkpoint_sort_key,
    )
    save_prediction_grid(
        predictions=predictions,
        checkpoint_names=all_checkpoints,
        out_path=out_result_folder / f"prediction_grid_{pred_slice}.png",
        title=f"{ablation_name} predictions",
    )
    save_prediction_grid(
        predictions=selected_predictions,
        checkpoint_names=sorted({next(iter(preds)) for preds in selected_predictions.values()}, key=checkpoint_sort_key),
        out_path=out_result_folder / f"selected_prediction_grid_{pred_slice}.png",
        title=f"{ablation_name} selected predictions",
    )
    save_selected_barplot(results_df, out_result_folder / f"selected_dice_{pred_slice}.png")

    print(f"Wrote evaluation results to {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EPS-Seg prediction TIFFs.")
    parser.add_argument("--exp_root", type=str, required=True, help="Directory containing exp_*.yaml files.")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory where evaluation outputs are written.")
    parser.add_argument("--pred_slice", type=int, default=PRED_SLICE)
    parser.add_argument("--pred_key", type=str, default=PRED_KEY)
    parser.add_argument("--gt_tiff", type=str, default=GT_TIFF)
    args = parser.parse_args()
    evaluate(
        exp_root=args.exp_root,
        out_dir=args.out_dir,
        pred_slice=args.pred_slice,
        pred_key=args.pred_key,
        gt_tiff=args.gt_tiff,
    )


if __name__ == "__main__":
    main()
