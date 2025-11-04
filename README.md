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
1. Create a folder in the `experiments/` directory.
2. Create a configuration file (YAML format) for the experiment. This must include at least: 
   - `project_name`: A name for the projectm, used for logging.
   - `dataset_cfg_path`: Path to a YAML file containing dataset configuration.
   - `train_cfg_path`: Path to a YAML file containing training configuration.
3. Create a dataset configuration file in the location specified by `dataset_cfg_path`. It will contain parameters such as data paths.
4. Create a training configuration file in the location specified by `train_cfg_path`. It must contain at least the `model_name` parameter.
5. From the root of the repository, run the training script with the experiment configuration file:
   ```bash
   python -m eps_seg.train --exp_config experiments/your_experiment/exp_config.yaml --env_file path/to/.env
   ```
   Where `--env_file` is optional and can be used to specify environment variables, such as WandB API keys:
   ```bash
   WANDB_API_KEY=your_wandb_api_key
   ```


  