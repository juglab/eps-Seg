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

To train a new model:
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
      cache_dir: "/path/to/cache/" # Directory to store cached patches, used to speed up loading and reproducibility.
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

### Predict using an Experiment Configuration

To run predictions using a trained model and an experiment configuration file, you can use the `predict.py` script. Here's how to do it:

1. Ensure you have a trained model checkpoint available. By default, it is saved in `<experiment_yaml_root>/checkpoints/<experiment_filename>/<model_name>/best_[supervised|semisupervised].ckpt`.
2. Run the prediction script with the experiment configuration file:
   ```bash
      uv run src/eps_seg/predict.py --exp_config experiments/your_experiment/exp_config.yaml --batch_size BATCH_SIZE
   ```
   To predict on just one slice, use the `--predict_on_slice Z_INDEX` argument.
This will create two folders next to the experiment configuration file:
- `outputs/<experiment_name>/<model_name>/<split>/<checkpoint_name>/`, containing the predicted segmentation as .tif file for testing and a csv with pixelwise predictions and coordinates for training/validation.
- `results/<experiment_name>/<model_name>/<checkpoint_name>/`, containing evaluation metrics in a csv file, containing Dice scores for each checkoint and split.