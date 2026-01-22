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
        self._writers: Dict[str, zarr.Array] = {} # Dict[test_key, zarr.Array], zarr writers for each test_key for the current rank
        self.out_dir = self.exp_config.outputs_dir / "predictions" / self.ckpt_path.stem
        os.makedirs(self.out_dir, exist_ok=True)
        
        # TODO: We could gain even more speed by tuning chunk sizes based rank batch sizes and volume sizes
        self.chunks = (64, 128, 128)  # Z, Y, X chunk size for zarr arrays
        
        # Zarr chunks written by this rank for each test_key
        self.rank_chunks: Dict[str, Set[Tuple[int, int, int]]] = {tk: set() for tk in self.test_keys} # Dict[test_key, Set[(cz, cy, cx)]]
        # Corresponding cache Zarr output paths for each rank and test_key
        self._get_rank_out_path = lambda rank, test_key: self.out_dir / f"rank_{rank}_{test_key}.zarr"
        # Final output Zarr paths for each test_key
        self.final_out_paths: Dict[str, Path] = {tk: self.out_dir / f"{tk}.zarr" for tk in self.test_keys} 
    
    def setup(self, trainer, pl_module, stage):
        # Called on each rank before prediction starts
        if stage != "predict":
            return
        
        # Get volume shapes from datamodule
        dm = trainer.datamodule
        pred_shapes = {tk: trainer.datamodule.data["test_images"][tk].shape for tk in dm.cfg.test_keys}  # Dict[test_key, (C, Z, Y, X)]
        
        rank = trainer.global_rank

        self.rank_paths: Dict[str, Path] = {} # Dict[test_key, Path]
        
        for test_key in self.test_keys:
             # Initialize zarr arrays for this rank and test_key
            self.rank_paths[test_key] = self._get_rank_out_path(rank, test_key)

            self._writers[test_key] = zarr.open(
                self.rank_paths[test_key],
                mode='w',
                shape=pred_shapes[test_key][-3:], # (Z,Y,X), ignores channels
                dtype=np.int8, # Support -1 label
                fill_value=-1,
                chunks=self.chunks,
            )

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        
        preds = prediction["preds"].cpu().numpy().astype(np.int8)  # (B,)
        coords = prediction["coords"].cpu().numpy().astype(np.intp)  # (B,3)
        test_keys = np.asarray(prediction["keys"]) # Tuple containing corresponding filenames for each prediction in the batch

        # A single batch may contain samples from different test volumes
        # Group predictions by test_key so we can write them in a single operation

        for tk in np.unique(test_keys):
            mask = test_keys == tk
            preds_k = preds[mask]
            coords_k = coords[mask]

            zi, yi, xi = coords_k.T  # (N,)

            # TODO: This could be optimized by writing every chunk at once, instead of every batch at a time
            self._writers[tk][zi, yi, xi] = preds_k
        
            # Track written chunks for merging later
            cz = zi // self.chunks[0]
            cy = yi // self.chunks[1]
            cx = xi // self.chunks[2]

            # This updates chunks without duplicates
            unique_chunks = np.unique(np.column_stack((cz, cy, cx)), axis=0)
            self.rank_chunks[tk].update(map(tuple, unique_chunks))

                
        
    def on_predict_end(self, trainer, pl_module):
        """
            Only on rank 0, merge the ranks outputs and write.
        """
        
        
        for tk in self.test_keys:
            # Save written chunks for this rank and test_key
            np.save(self.out_dir / f"rank_{trainer.global_rank}_chunks_{tk}.npy", 
                    np.array(list(self.rank_chunks[tk]), dtype=np.int32))
       
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Wait for all ranks to finish writing
            torch.distributed.barrier()

        if trainer.is_global_zero:
            print("Merging rank outputs...")
            self.merge(trainer)
    
    def merge(self, trainer):
        """
            Merge all ranks zarr outputs into a final zarr array.
            Should be ran ONLY on rank 0 and after each rank has finished writing..
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
        self.out_preds_csv = self.exp_config.outputs_dir / "test_predictions" / (self.ckpt_path.stem+".csv") 
        self._predictions_csv_header_written = False

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
                        
            os.makedirs(self.out_preds_csv.parent, exist_ok=True)
            self._predictions_csv_header_written = False

    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """
            Writes predictions for the current batch to a CSV file.
        """
        # Save a batch of predictions to CSV
        
        class_probs = outputs["class_probabilities"].cpu().numpy()
        true_labels = outputs["labels"].cpu().numpy()
        pred_labels = outputs["preds"].cpu().numpy()
        coords = outputs["coords"].cpu().numpy()
        test_keys = outputs["keys"]

        write_dict = {
            "test_key": test_keys,
            "coords_x": coords[:, -1],
            "coords_y": coords[:, -2],
            "coords_z": coords[:, -3],
            "true_label": true_labels.squeeze(),
            "pred_label": pred_labels.squeeze(),
        }

        for i in range(class_probs.shape[1]):
            write_dict[f"class_prob_{i}"] = class_probs[:, i]

        df = pd.DataFrame(write_dict)
        # Append to predictions dataframe
        df.to_csv(
            self.out_preds_csv,
            mode="a",
            header=not self._predictions_csv_header_written,
            index=False,
        )
        self._predictions_csv_header_written = True
        

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        """
            Collects metrics and saves to results CSV.
        """

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