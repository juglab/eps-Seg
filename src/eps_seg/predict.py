import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from eps_seg.models import LVAEModel
from eps_seg.dataloaders.datamodules import BetaSegTrainDataModule
from eps_seg.training.callbacks import EarlyStoppingWithPatiencePropagation, SemiSupervisedModeCallback, ThresholdSchedulerCallback, RadiusSchedulerCallback
from eps_seg.config.train import ExperimentConfig
from dotenv import load_dotenv
import torch 
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import pandas as pd
import os

class PredictionWriterCallback(BasePredictionWriter):
    def __init__(self, exp_config: ExperimentConfig, ckpt_path: Path):
        super().__init__(write_interval="epoch")
        self.ckpt_path = ckpt_path
        self.exp_config = exp_config
    
    # TODO: Continue
    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)
        if stage == "predict":
            self.out_dir = self.exp_config.outputs_dir / "predictions" / self.ckpt_path.stem
            os.makedirs(self.out_dir, exist_ok=True)



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
    dm = BetaSegTrainDataModule(cfg=dataset_config, train_cfg=train_config)
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
                                #limit_predict_batches=4,
                                )
            if mode == "test":
                trainer.test(model=model, datamodule=dm)
            elif mode == "predict":
                trainer.predict(model=model, datamodule=dm)

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