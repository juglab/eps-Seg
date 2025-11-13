from eps_seg.dataloaders.datasets import PredictionDataset
from torch.utils.data import DataLoader
from eps_seg.models.lvae import LVAEModel
from eps_seg.config.train import ExperimentConfig, TrainConfig
from eps_seg.config.datasets import BaseEPSDatasetConfig
from eps_seg.config.models import BaseEPSModelConfig
from eps_seg.dataloaders.datamodules import BetaSegTrainDataModule
from eps_seg.dataloaders.samplers import ModeAwareBalancedAnchorBatchSampler
from eps_seg.dataloaders.utils import flex_collate
import lightning as L
import torch
import numpy as np
from pathlib import Path
import tifffile
import tqdm
from typing import Literal, Union, Tuple, List
from torchmetrics.classification import F1Score
import pandas as pd
import os
from dotenv import load_dotenv
import argparse

def get_dataloader(dataset_config: BaseEPSDatasetConfig, 
                   model_config: BaseEPSModelConfig,
                   train_config: TrainConfig,
                   split: Literal["train", "val", "test"], 
                   batch_size: int, 
                   predict_on_slice: Union[int, None] = None,
                   num_workers: int = 0,
                   dataset_stats: Union[Tuple[float], None] = None) -> DataLoader:
        """
            Returns a DataLoader for the specified dataset split.

            Args:
                dataset_config (BaseEPSDatasetConfig): Configuration for the dataset.
                model_config (BaseEPSModelConfig): Configuration for the model.
                train_config ()
                split (Literal["train", "val", "test"]): The dataset split to load
                batch_size (int): The batch size for the DataLoader.
                predict_on_slice (Union[int, None]): If specified, predicts on a single slice (2D) at this z-index.
                num_workers (int): Number of worker processes for data loading.
                dataset_stats: mean and std of the dataset for testing phase. Must be provided for testing phase since the dataset doesn't normalize them on its own.

            Returns:
                DataLoader: DataLoader for the specified dataset split.
                Data shape: shape of the data in case of test split
        """
        data_shape = None
        if split in ["train", "val"]:
            # Overwrite train config batch size for prediction
            tconfig = train_config.model_copy()
            tconfig.batch_size = batch_size

            train_dm = BetaSegTrainDataModule(
            cfg=dataset_config,
            train_cfg=tconfig)
            train_dm.setup("predict")
            # TODO: We JUST need a test dataloader implemented in the DataModule to avoid all this...
            if split == "train":
                return DataLoader(
                    train_dm.train_dataset,
                    batch_sampler=ModeAwareBalancedAnchorBatchSampler(
                        train_dm.train_dataset,
                        total_patches_per_batch=train_dm.train_cfg.batch_size,
                        shuffle=False,
                    ),
                    collate_fn=flex_collate,
                    ), data_shape
            else:
                return DataLoader(
                    train_dm.train_dataset,
                    batch_sampler=ModeAwareBalancedAnchorBatchSampler(
                        train_dm.val_dataset,
                        total_patches_per_batch=train_dm.train_cfg.batch_size,
                        shuffle=False,
                    ),
                    collate_fn=flex_collate,
                    ), data_shape
        elif split == "test":
            # Load test volume:
            test_img_fp, test_lbl_fp = dataset_config.get_test_paths()
            assert dataset_stats is not None, "For testing, dataset should be normalized with mu and std from the model"
            if len(test_img_fp) > 1:
                raise NotImplementedError("Prediction on multiple test volumes is not implemented yet.")
            
            test_img = tifffile.imread(test_img_fp[0])
            test_lbl = tifffile.imread(test_lbl_fp[0])
            data_shape = test_lbl.shape
            if predict_on_slice is not None:
                data_shape = list(data_shape)
                data_shape[-3] = 1 # Remove z dimension for 2D prediction
                data_shape = tuple(data_shape)

            dataset = PredictionDataset(
                image=test_img,
                label=test_lbl,
                z = predict_on_slice,
                dim=dataset_config.dim,
                patch_size=model_config.img_shape[0],
                normalize_stats=dataset_stats
            )
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True), data_shape

def predict(exp_config: ExperimentConfig, 
            batch_size: int, 
            limit_batches: Union[int, None] = None, 
            predict_on_slice: Union[int, None] = None,
            splits: List[Literal["train", "val", "test"]] = ["test", "train", "val"]):
    """
        Predict with eps-seg model using the provided experiment configuration.

        Results will be stored in the experiment's outputs and results folders.

        Args:
            exp_config (ExperimentConfig): The experiment configuration.
            batch_size (int): The batch size for prediction.
            limit_batches (int, optional): If specified, limits the number of batches to predict on. (Used for debugging)
            predict_on_slice (int, optional): If specified, predicts on a single slice (2D) at this z-index.


    """
    train_config, dataset_config, model_config = exp_config.get_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_results_csv = exp_config.results_dir
    if out_results_csv.exists():
        results_df = pd.read_csv(out_results_csv)
        print(f"Loaded existing results from: {out_results_csv}")
    else:
        results_df = pd.DataFrame(columns=["exp_name", "model_ckpt", "split", "datetime", "predict_on_slice"])
    # Predict for all models in the folder.

    all_ckpts = sorted(list(exp_config.best_checkpoint_path("supervised").parent.glob("*.ckpt")))

    for model_ckpt in all_ckpts:
        print(f"Predicting with model: {model_ckpt}")
        model = LVAEModel.load_from_checkpoint(model_ckpt,
                                            model_cfg=model_config,
                                            train_cfg=train_config).to(device)
        model.eval()

        for split in splits:
            dataloader, data_shape = get_dataloader(dataset_config, 
                                        model_config,
                                        train_config,
                                        split, 
                                        batch_size=batch_size, 
                                        predict_on_slice=predict_on_slice, 
                                        dataset_stats=(model.model.data_mean.cpu().numpy(), model.model.data_std.cpu().numpy()),
                                        num_workers=os.cpu_count()
                                        )
            
            predictor = L.Trainer(limit_predict_batches=limit_batches,
                                  precision = "16-mixed" if train_config.amp else 32,)
            preds = predictor.predict(model=model,
                                            dataloaders=[dataloader])
            # TODO: train/val and prediction dataset have different formats. Handle it in predictions

            if split == "test":
                if predict_on_slice is None:
                    out_tiff_fp = exp_config.outputs_dir / split / f"{model_ckpt.stem}.tif"
                else:
                    out_tiff_fp = exp_config.outputs_dir / split / f"{model_ckpt.stem}_slice_{predict_on_slice}.tif"
                out_tiff = np.zeros(data_shape, dtype=np.uint8) - 1  # Initialize with -1 for unlabeled
            else:
                # For train/val we store preds / gt / coords in a csv
                out_csv_fp = exp_config.outputs_dir / split / f"{model_ckpt.stem}_predictions.csv"
                preds_list = []

            # TODO: This should be moved in a test function inside the model. I'm doing it here to ensure multigpu doesn't interfere
            dice_acc = F1Score(num_classes=model_config.n_components, average=None, task="multiclass", ignore_index=-1) 

            for model_out, in_batch in preds:
                in_patch, labels, _, coords = in_batch

                pred_labels = torch.argmax(model_out["class_probabilities"], dim=-1)

                dice_acc.update(pred_labels.cpu(), labels.cpu())
                # Reconstruct full volume
                xc = coords[:, -1].cpu().numpy()
                yc = coords[:, -2].cpu().numpy()
                
                if split == "test":
                    zc = coords[:, -3].cpu().numpy() if predict_on_slice is None else np.array([0,]*len(xc))
                    out_tiff[zc, yc, xc] = pred_labels.cpu().numpy().astype(np.uint8)
                else:
                    for i in range(len(xc)):
                        preds_list.append({
                            "x": xc[i],
                            "y": yc[i],
                            "z": coords[:, -3].cpu().numpy()[i],
                            "pred_label": pred_labels.cpu().numpy()[i],
                            "true_label": labels.cpu().numpy()[i]
                        })

            if split == "test":
                out_tiff_fp.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(out_tiff_fp, out_tiff)
                print(f"Saved predicted tiff to: {out_tiff_fp}")
            else:
                out_csv_fp.parent.mkdir(parents=True, exist_ok=True)
                preds_df = pd.DataFrame(preds_list)
                preds_df.to_csv(out_csv_fp, index=False)
                print(f"Saved predictions csv to: {out_csv_fp}")

            # Compute metrics
            dice_scores = dice_acc.compute()
            dice_acc.reset()
            
            this_results_df = pd.DataFrame({
                "exp_name": [exp_config.experiment_name],
                "model_name": [train_config.model_name],
                "model_ckpt": [model_ckpt.name],
                "split": [split],
                "datetime": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
                "predict_on_slice": [predict_on_slice],
                **{f"dice_class_{i}": [dice_scores[i].item()] for i in range(len(dice_scores))},
                "mean_dice_score": [dice_scores.mean().item()]
            })
            results_df = pd.concat([results_df, this_results_df], ignore_index=True)
            out_results_csv.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_results_csv, index=False)

            print(f"Dice scores on {split} set: {dice_scores}")
def main():
    # Allows to be run as: python -m eps_seg.train --exp_config path/to/exp_config.yaml --env_file path/to/.env
    parser = argparse.ArgumentParser(description="Train EPS-Seg Model")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to experiment configuration YAML file")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file with environment variables")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--predict_on_slice", type=int, default=None, help="If specified, predicts on a single slice (2D) at this z-index.")

    args = parser.parse_args()
    print("Loading experiment config from:", args.exp_config)
    print("Loading environment variables from:", args.env_file)
    load_dotenv(args.env_file)
    exp_config = ExperimentConfig.from_yaml(args.exp_config)

    predict(exp_config, 
            batch_size=args.batch_size, 
            predict_on_slice=args.predict_on_slice)

if __name__ == "__main__":
   main()