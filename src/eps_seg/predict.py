import argparse
from eps_seg.dataloaders.datasets import PredictionDataset
from torch.utils.data import DataLoader
from eps_seg.models.lvae import LVAEModel
from eps_seg.config.train import ExperimentConfig, TrainConfig
from eps_seg.config.datasets import BaseEPSDatasetConfig
from eps_seg.config.models import BaseEPSModelConfig
from eps_seg.dataloaders.datamodules import BetaSegTrainDataModule
import lightning as L
import torch
import numpy as np
from pathlib import Path
import tifffile
import tqdm
from typing import Literal, Union, Tuple
from dotenv import load_dotenv

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
        """
        dim = dataset_config.dim
        if split == "train":
            # Overwrite train config batch size for prediction
            tconfig = train_config.model_copy()
            tconfig.batch_size = batch_size

            train_dm = BetaSegTrainDataModule(
            cfg=dataset_config,
            train_cfg=tconfig)
            train_dm.setup("predict")

            # FIXME: Be aware that this will return shuffled data!
            return train_dm.train_dataloader()
        elif split == "val":
            tconfig = train_config.model_copy()
            tconfig.batch_size = batch_size

            train_dm = BetaSegTrainDataModule(
            cfg=dataset_config,
            train_cfg=tconfig)
            train_dm.setup("predict")

            # FIXME: Be aware that this will return shuffled data!
            return train_dm.val_dataloader()

        elif split == "test":
            # Load test volume:
            test_img_fp, test_lbl_fp = dataset_config.get_test_paths()
            assert dataset_stats is not None, "For testing, dataset should be normalized with mu and std from the model"
            if len(test_img_fp) > 1:
                raise NotImplementedError("Prediction on multiple test volumes is not implemented yet.")
            
            test_img = tifffile.imread(test_img_fp[0])
            test_lbl = tifffile.imread(test_lbl_fp[0])

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
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def try_batch_size(model, model_config, batch_size, channels, device):
    x_shape = (batch_size, ) + (channels, ) + tuple(model_config.img_shape)
    y_shape = (batch_size, 1)
    batch = {"patch": torch.zeros(x_shape).to(device)}
    model.predict_step(batch, batch_idx=0, dataloader_idx=0)

def predict(exp_config: ExperimentConfig, batch_size: int = None, predict_on_slice: Union[int, None] = None):
    """
        Predict with eps-seg model using the provided experiment configuration.

        Results will be stored in the experiment's outputs and results folders.

        Args:
            exp_config (ExperimentConfig): The experiment configuration.
            batch_size (int, optional): The batch size for prediction. If None, it will be determined automatically.
            predict_on_slice (int, optional): If specified, predicts on a single slice (2D) at this z-index.


    """
    train_config, dataset_config, model_config = exp_config.get_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predict for all models in the folder.
    for model_ckpt in exp_config.best_checkpoint_path("supervised").parent.glob("*.ckpt"):
        print(f"Predicting with model: {model_ckpt}")
        model = LVAEModel.load_from_checkpoint(model_ckpt,
                                            model_cfg=model_config,
                                            train_cfg=train_config).to(device)
        model.eval()
        channels = dataset_config.n_channels

        # Finding correct batch_size
        if batch_size is None:
            # Find maximum batch size that fits in memory
            batch_size = 1
            while True:
                try:
                    try_batch_size(model, model_config, batch_size, channels, device)
                except RuntimeError:
                    print(f"Batch size {batch_size} too large, using {batch_size // 2}")
                    batch_size //= 2
                    break
                print(f"Batch size {batch_size} fits, trying {batch_size * 2}")
                batch_size *= 2

        
        for split in ["test", "train", "val"]:
            dataloader = get_dataloader(dataset_config, 
                                        model_config,
                                        train_config,
                                        split, 
                                        batch_size, 
                                        predict_on_slice=predict_on_slice, 
                                        dataset_stats=(model.model.data_mean.cpu().numpy(), model.model.data_std.cpu().numpy()),
                                        )
            
            predictor = L.Trainer()
            predictions = predictor.predict(model=model,
                                            dataloaders=[dataloader])
            
            # TODO: Implement saving and result computation
            print("Predicted")


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

    predict(exp_config, batch_size=args.batch_size, predict_on_slice=args.predict_on_slice)

if __name__ == "__main__":
   main()