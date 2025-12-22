# eps-Seg
A hierarchical variational autoencoder based method for semantic segmentation of Electron Microscopy data.


## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:juglab/eps-Seg.git
    cd eps-Seg
    ```
2. Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create a virtual environment and install the package:
   ```bash
    uv sync
    source .venv/bin/activate
    uv pip install -e . # Editable install for development
   ```

## Usage

To train a new model on the provided datasets:

1. Create a folder in the `experiments/` directory (or a directory of your choice).

2. Create a configuration file (YAML format) for the experiment. This should include:
   - `project_name`: A name for the projectm, used for logging.
   - `dataset_cfg_path`: Path to a YAML file containing dataset configuration.
   - `train_cfg_path`: Path to a YAML file containing training configuration.
   - `model_cfg_path`: Path to a YAML file containing model configuration. (Optional; if not provided, default model parameters will be used.)
   
   Configuration paths can be either absolute or relative to the experiment YAML file.

3. Create a dataset configuration file in the location specified by `dataset_cfg_path`. It should contain (example for 2D BetaSeg dataset):
   ```yaml
      type: "BetaSegDatasetConfig" # Type of dataset configuration, from eps_seg.config.datasets. Check the parameters of the class for more details.
      dim: 2
      data_dir: "/path/to/data/"
      cache_dir: "/path/to/cache/" # Directory to store cached patches, used to speed up loading and reproducibility. We STRONGLY recommend to use it unless you have very limited disk space.
      enable_cache: True
      train_keys: ["high_c1", "high_c2", "high_c3"] # Which filenames to use for training
      test_keys: ["high_c4"] # Which filenames to use for testing
      seed: 42 # seed for data splitting, only used when no cache is available
      patch_size: 64
   ```
4. Create a training configuration file in the location specified by `train_cfg_path`. It must contain at least the `model_name` parameter.
5. From the root of the repository, run the training script with the experiment configuration file:
   ```bash
   python src/eps_seg/train.py --exp_config experiments/your_experiment/exp_config.yaml --env_file path/to/.env
   ```
   Where `--env_file` is optional and can be used to specify environment variables, such as WandB API keys:
   ```bash
   WANDB_API_KEY=your_wandb_api_key
   ```

### Predict and Testing using an Experiment Configuration

To run predictions using a trained model and an experiment configuration file, you can use the `predict.py` script. Here's how to do it:

1. Ensure you have a trained model checkpoint available. By default, it is saved in `<experiment_yaml_root>/checkpoints/<experiment_filename>/<model_name>/best_[supervised|semisupervised].ckpt`.
2. Run the prediction script with the experiment configuration file:
   ```bash
      uv run src/eps_seg/predict.py --exp_config experiments/your_experiment/exp_config.yaml --batch_size BATCH_SIZE [--predict || --test]
   ```
   By default, the script predicts all the checkpoints in the experiment checkpoints folder. You can use the `--models` argument to specify a list of model checkpoint names (e.g., `last.ckpt`) to predict.

The predict script will save the predicted segmentations in the `predictions/` folder inside the experiment root directory, following a similar structure to the checkpoints folder.

If `--test` is specified, the outputs will be saved as CSV files containing the pixelwise predictions and the corresponding ground truth labels for the test set. It will also compute and log the Dice score for each class and the average Dice score across all classes and save it in the `results/` folder inside the experiment root directory.
If `--predict` is specified, the outputs will be saved as a Zarr file and a TIFF file containing the full predicted segmentation for each test volume.

### Defining your own Dataset

To define your own dataset, you need to create a new class that inherits from `BaseEPSDatasetConfig` in `src/eps_seg/config/datasets.py` and create the corresponding YAML file to pass to the training script.
This class should follow all the parameters defined in the base class. To let the model know about your local file structure, you will need to override the `get_paths(stage: str) -> Dict[str, Tuple(Path, Path)]` method, which takeas as input a `stage` (i.e., ) maps the dataset keys (i.e., filenames specified in `train_keys` and `test_keys`) to their actual file paths on disk. Everything else (data loading, patch extraction, caching, etc.) will be handled by DataModule and Dataset classes.

For example

```python
class MyDatasetConfig(BaseEPSDatasetConfig):
   ...
   def get_image_label_paths(self, keys: List[str]) -> Dict[str, Tuple[Path, Path]]:
         """
               Return the image and label file paths for the specified keys.
               For BetaSeg, we have [data_dir]/[key]/[key]_source.tif and [data_dir]/[key]/[key]_gt.tif
         """
         paths = {}
         for key in keys:
               img_path = Path(self.data_dir) / key / f"{key}_source.tif"
               lbl_path = Path(self.data_dir) / key / f"{key}_gt.tif"
               paths[key] = (img_path, lbl_path)
         return paths
```

Then, after creating the corresponding YAML file (e.g., `mydataset.yaml`) with the following content:

```yaml
type: "MyDatasetConfig" # Name of the class you created
```
you can use it as any other dataset configuration.
