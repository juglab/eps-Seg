import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from eps_seg.models import LVAEModel
from eps_seg.dataloaders.datamodules import EPSSegDataModule
from eps_seg.training.callbacks import EarlyStoppingWithPatiencePropagation, SemiSupervisedModeCallback, ThresholdSchedulerCallback, RadiusSchedulerCallback
from eps_seg.config.train import ExperimentConfig
from dotenv import load_dotenv
import torch 
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import pandas as pd
import os
import tifffile as tiff
import numpy as np
import zarr
from eps_seg.utils.outputs import zarr_to_tiff
from typing import Dict, Set, Tuple

class PredictionWriterCallback(BasePredictionWriter):
    """
        Callback to write prediction outputs to Zarr and TIFF files after prediction.
    """
    def __init__(self, exp_config: ExperimentConfig, ckpt_path: Path):
        super().__init__(write_interval="batch")
        self.ckpt_path = ckpt_path
        self.exp_config = exp_config
        _, dataset_config, _ = exp_config.get_configs()
        self.test_keys = dataset_config.test_keys
        self._writers: Dict[str, zarr.Array] = {} # Dict[test_key, zarr.Array]
        self.out_dir = self.exp_config.outputs_dir / "predictions" / self.ckpt_path.stem
        os.makedirs(self.out_dir, exist_ok=True)

        self.chunks = (64, 128, 128)  # Z, Y, X chunk size for zarr arrays
        
        self.rank_chunks: Dict[str, Dict[int, Set[Tuple[int, int, int]]]] = {tk: {} for tk in self.test_keys} # Dict[test_key, Dict[rank, Set[Tuple[cz, cy, cx]]]]
        self.final_out_paths: Dict[str, Path] = {tk: self.out_dir / f"{tk}.zarr" for tk in self.test_keys} 

        self._get_rank_out_path = lambda rank, test_key: self.out_dir / f"rank_{rank}_{test_key}.zarr"
    
    def setup(self, trainer, pl_module, stage):
        # Called on each rank before prediction starts
        if stage != "predict":
            return
        
        # Get volume shapes from datamodule
        dm = trainer.datamodule
        pred_shapes = {tk: trainer.datamodule.data["test_images"][tk].shape for tk in dm.cfg.test_keys}  # Dict[test_key, (Z, Y, X)]
        
        rank = trainer.global_rank

        self.rank_chunks[rank] = set()
        self.rank_paths: Dict[str, Path] = {} # Dict[test_key, Path]
        
        for test_key in self.test_keys:
             # Initialize zarr arrays for this rank and test_key
            self.rank_paths[test_key] = self._get_rank_out_path(rank, test_key)
            self.rank_chunks[test_key][rank] = set()

            self._writers[test_key] = zarr.open(
                self.rank_paths[test_key],
                mode='w',
                shape=pred_shapes[test_key],
                dtype=np.int8, # Support -1 label
                fill_value=-1,
                chunks=self.chunks,
            )

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        
        preds = prediction["preds"].cpu().numpy().astype(np.int8)  # (B,)
        coords = prediction["coords"].cpu().numpy().astype(np.intp)  # (B,3)
        test_keys = prediction["keys"] # Tuple containing corresponding filenames for each prediction in the batch

        for i_tk, tk in enumerate(test_keys):
            zi, yi, xi = coords[i_tk]  # (3,)
            self._writers[tk][zi, yi, xi] = preds[i_tk]    

            # Track written chunks for merging
            cz = zi // self.chunks[0]
            cy = yi // self.chunks[1]
            cx = xi // self.chunks[2]
            self.rank_chunks[tk][trainer.global_rank].add((cz, cy, cx))
        
    def on_predict_end(self, trainer, pl_module):
        """
            Only on rank 0, merge the ranks outputs and write.
        """
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Wait for all ranks to finish writing
            torch.distributed.barrier()
        for tk in self.test_keys:
            # Save written chunks for this rank and test_key
            np.save(self.out_dir / f"rank_{trainer.global_rank}_chunks_{tk}.npy", 
                    np.array(list(self.rank_chunks[tk][trainer.global_rank]), dtype=np.int32))

        if trainer.is_global_zero:
            print("Merging rank outputs...")
            self.merge(trainer)
    
    def merge(self, trainer):
        """
            Merge all ranks zarr outputs into a final zarr array.
            Should be ran ONLY on rank 0.
        """
        if not trainer.is_global_zero:
            return
        print("Merging Zarr ranks...")

        for tk in self.test_keys:
            
            rank0 = zarr.open(self._get_rank_out_path(0, tk), mode="r")
    
            final = zarr.open(
                self.final_out_paths[tk],
                mode="w",
                shape=rank0.shape,
                chunks=rank0.chunks,
                dtype=rank0.dtype,
                fill_value=-1,
            )
            final[:] = rank0[:]

            for rank in range(1, trainer.world_size):
                z = zarr.open(self._get_rank_out_path(rank, tk), mode="r")
                chunks = np.load(self.out_dir / f"rank_{rank}_chunks_{tk}.npy")

                for cz, cy, cx in chunks:
                    z0 = cz * self.chunks[0]
                    y0 = cy * self.chunks[1]
                    x0 = cx * self.chunks[2]

                    sel = (
                        slice(z0, z0 + self.chunks[0]),
                        slice(y0, y0 + self.chunks[1]),
                        slice(x0, x0 + self.chunks[2]),
                    )

                    data = z[sel]
                    if not np.any(data != -1):
                        continue

                    out = final[sel]
                    mask = data != -1
                    out[mask] = data[mask]
                    final[sel] = out
                
                 # Convert final zarr to tiff
            print(f"Converting {tk} Zarr to TIFF...")
            zarr_to_tiff(self.final_out_paths[tk])
            print(f"Saved final prediction TIFF to: {self.final_out_paths[tk].with_suffix('.tif')}")
       

class TestWriterCallback(L.Callback):
    """
        Callback to write test outputs and metrics to a CSV file after testing.
    """
    def __init__(self, exp_config: ExperimentConfig, ckpt_path: Path):
        super().__init__()
        self.exp_config = exp_config
        self.out_csv_path = exp_config.results_csv_path
        self.ckpt_path = ckpt_path
        
    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)
        if stage == "test":
            # Prepare results CSV
            if self.out_csv_path.exists():
                self.results_df = pd.read_csv(self.out_csv_path)
                print(f"Loaded existing results from: {self.out_csv_path}")
            else:
                self.results_df = pd.DataFrame(columns=["exp_name", "model_ckpt", "datetime"])
            # Prepare outputs csv
            self.out_preds_csv = self.exp_config.outputs_dir / "test_predictions" / (self.ckpt_path.stem+".csv")
            os.makedirs(self.out_preds_csv.parent, exist_ok=True)
            self.predictions_df = pd.DataFrame()
                
    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)

        # Collect metrics
        metrics = trainer.callback_metrics
        result_entry = {
            "exp_name": self.exp_config.experiment_name,
            "model_ckpt": self.ckpt_path.name,
            "datetime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                result_entry[key] = value.item()
            else:
                result_entry[key] = value
        self.results_df = pd.concat([self.results_df, pd.DataFrame([result_entry])], ignore_index=True)
        os.makedirs(self.out_csv_path.parent, exist_ok=True)
        self.results_df.to_csv(self.out_csv_path, index=False)

        # Save predictions
        all_outputs = pl_module.current_test_outputs
        print(f"Writing test predictions to: {self.out_preds_csv}")
        for batch_idx, output in enumerate(all_outputs):
            write_dict = {}
            class_probs = output["class_probabilities"].cpu().numpy()
            true_labels = output["labels"].cpu().numpy()
            pred_labels = output["preds"].cpu().numpy()
            coords = output["coords"].cpu().numpy()
            write_dict["coords_x"] = coords[:, -1]
            write_dict["coords_y"] = coords[:, -2]
            write_dict["coords_z"] = coords[:, -3]
            write_dict["true_label"] = true_labels.squeeze()
            write_dict["pred_label"] = pred_labels.squeeze()
            for i in range(class_probs.shape[1]):
                write_dict[f"class_prob_{i}"] = class_probs[:, i]
            self.predictions_df = pd.concat([self.predictions_df, pd.DataFrame(write_dict)], ignore_index=True)
        self.predictions_df.to_csv(self.out_preds_csv, index=False)
        print(f"Saved test predictions to: {self.out_preds_csv}")

def test_predict(exp_config: ExperimentConfig, 
                 predict: bool = False, 
                 test: bool = False, 
                 batch_size: int = None,
                 models: list[str] = ["all"]):
    """
        Run prediction and/or testing on an EPS-Seg model based on the provided experiment configuration.
        Args:
            exp_config (ExperimentConfig): The experiment configuration object containing paths to training, dataset, and model configs.
            predict (bool): If True, run prediction.
            test (bool): If True, run testing.
            batch_size (int): Batch size for prediction/testing (overrides config if provided).
    """
    train_config, dataset_config, model_config = exp_config.get_configs()

    if batch_size is not None:
        print(f"Overriding batch size to {batch_size} for prediction/testing...")
        train_config.test_batch_size = batch_size
    
    MODES = [] + (["predict"] if predict else []) + (["test"] if test else [])
    
    # Check what checkpoints to use
    CKPTS_PATHS = []
    ckpt_folder = exp_config.best_checkpoint_path("supervised").parent
    if "all" in models:
        CKPTS_PATHS += sorted(list(ckpt_folder.glob("*.ckpt")))
    for ckpt_name in models:
        if ckpt_name == "all":
            continue
        ckpt_path = ckpt_folder / ckpt_name
        if ckpt_path.exists():
            CKPTS_PATHS.append(ckpt_path)
        else:
            print(f"Warning: Checkpoint {ckpt_path} does not exist and will be skipped.")
    
    # Run prediction and/or testing           
    dm = EPSSegDataModule(cfg=dataset_config, train_cfg=train_config)
    for mode in MODES:
        for ckpt_path in CKPTS_PATHS:
            print(f"Running {mode} with checkpoint: {ckpt_path}")
            
            model = LVAEModel.load_from_checkpoint(str(ckpt_path),
                                        model_cfg=model_config,
                                        train_cfg=train_config)
            model.eval()
            devices = 1 if mode == "test" else "auto"
            # We can't use DDP for testing because with default sampler it duplicates data to match gpus
            strategy = "ddp" if mode == "predict" and torch.cuda.device_count() > 1 else "auto"
            callbacks = [TestWriterCallback(exp_config=exp_config, ckpt_path=ckpt_path)] if mode == "test" \
                   else [PredictionWriterCallback(exp_config=exp_config, ckpt_path=ckpt_path)]
            
            trainer = L.Trainer(devices=devices, 
                                accelerator="gpu", 
                                precision = "16-mixed" if train_config.amp else 32,
                                strategy=strategy,
                                callbacks=callbacks,
                                #limit_test_batches=4,
                                #limit_predict_batches=12,
                                )
            if mode == "test":
                trainer.test(model=model, datamodule=dm)
            elif mode == "predict":
                trainer.predict(model=model, 
                                datamodule=dm,
                                return_predictions=False, # Prevents gathering all predictions in memory. We write them on the fly via callback.
                                )

def main():
    parser = argparse.ArgumentParser(description="Predict / Test EPS-Seg Model")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to experiment configuration YAML file")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file with environment variables")
    parser.add_argument("--predict", action="store_true", help="If set, run prediction")
    parser.add_argument("--test", action="store_true", help="If set, run testing")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for prediction/testing (overrides config)")
    parser.add_argument("--models", type=str, nargs='+', default=["all"], help="List of checkpoints names (without path) to test/predict. Use 'all' to test/predict all available checkpoints.")

    args = parser.parse_args()
    print("Loading experiment config from:", args.exp_config)
    print("Loading environment variables from:", args.env_file)
    load_dotenv(args.env_file)
    assert args.predict or args.test, "At least one of --predict or --test must be set"
    exp_config = ExperimentConfig.from_yaml(args.exp_config)
    
    test_predict(exp_config, 
                 predict=args.predict, 
                 test=args.test, 
                 batch_size=args.batch_size,
                 models=args.models)

if __name__ == "__main__":
    main()