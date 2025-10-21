import os
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings

warnings.filterwarnings("ignore")
# We import all our dependencies.
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.lvae import LadderVAE
from boilerplate.dataloader import SemisupervisedDataset, ModeAwareBalancedAnchorBatchSampler, flex_collate
import training
from tqdm import tqdm
import tifffile as tiff




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("--image", type=str, help="Path to input image")
parser.add_argument("--labels", type=str, help="Path to input label")
parser.add_argument(
    "--directory_path", type=str, default="/group/jug/Sheida/HVAE/segmentation/65/"
)
parser.add_argument("--contrastive_learning", type=bool, default=True)
parser.add_argument("--mode", type=str, default="supervised")
parser.add_argument("--labeled_ratio", type=float, default=1)
parser.add_argument("--stochastic_block_type", type=str, default="mixture")
parser.add_argument("--conditional", type=bool, default=True)
parser.add_argument("--condition_type", type=str, default="mlp")
parser.add_argument("--sample_ratio", type=int, default=20)
parser.add_argument("--num_latents", type=int, default=5)
parser.add_argument("--blocks_per_layer", type=int, default=5)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--beta", type=float, default=1e-1)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--initial_mask_size", type=int, default=1)
parser.add_argument("--final_mask_size", type=int, default=1)
parser.add_argument("--initial_label_size", type=int, default=1)
parser.add_argument("--final_label_size", type=int, default=1)
parser.add_argument("--step_interval", type=int, default=10)
parser.add_argument("--load_checkpoint", type=bool, default=False)

args = parser.parse_args()

# If --image and --labels are provided, use them directly
if args.image and args.labels:
    imgs = {"plugin": tiff.imread(args.image).astype(np.float16)}
    lbls = {"plugin": tiff.imread(args.labels).astype(np.float16)}
    keys = ["plugin"]
else:
    # fallback to hardcoded files
    data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
    keys = ["high_c1", "high_c2", "high_c3"]
    img_paths = [os.path.join(data_dir + key + f"/{key}_source.tif") for key in keys]
    lbl_paths = [os.path.join(data_dir + key + f"/{key}_gt.tif") for key in keys]
    imgs = {key: tiff.imread(path).astype(np.float16) for key, path in zip(keys, img_paths)}
    lbls = {key: tiff.imread(path).astype(np.float16) for key, path in zip(keys, lbl_paths)}


use_wandb = True

patch_size = 64

gaussian_noise_std = None

model_name = "segmentation"
directory_path = args.directory_path

# Model-specific
load_checkpoint = args.load_checkpoint
checkpoint = directory_path + "model_supervised/best.net"

noiseModel = None

# Training-specific
batch_size = 1024
lr = 3e-5
max_epochs = 1000
num_latents = args.num_latents
z_dims = [32] * int(num_latents)
blocks_per_layer = args.blocks_per_layer
batchnorm = True
free_bits = 0.0

alpha = args.alpha  # weight of the inpainting loss
beta = args.beta  # weight of the KL loss
gamma = args.gamma  # weight of the contrastive loss

initial_mask_size = args.initial_mask_size
final_mask_size = args.final_mask_size
initial_label_size = args.initial_label_size
final_label_size = args.final_label_size
step_interval = args.step_interval

contrastive_learning = args.contrastive_learning
margin = 50  # distance for negative pairs in contrastive learning
lambda_contrastive = 0.5  # weight of the positive pairs in contrastive learning
# (1-lambda_contrastive is the weight of the negative pairs)

mode = args.mode  # 'supervised' or 'semisupervised' or 'unsupervised'
labeled_ratio = args.labeled_ratio  # ratio of labeled data in semisupervised mode
stochastic_block_type = args.stochastic_block_type  # 'normal' or 'mixture'
conditional = args.conditional  # True for conditional LVAE (conditioned on gt label)
condition_type = args.condition_type  # 'mlp' or 'transformer'
assert (conditional == True and condition_type != None) or conditional == False
n_components = 4  # number of components for prior
n_classes = 4  # number of classes in the dataset
# train data
data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
keys = ["high_c1", "high_c2", "high_c3"]

img_paths = [os.path.join(data_dir + key + f"/{key}_source.tif") for key in keys]
lbl_paths = [os.path.join(data_dir + key + f"/{key}_gt.tif") for key in keys]
imgs = {key: tiff.imread(path).astype(np.float16) for key, path in zip(keys, img_paths)}
lbls = {key: tiff.imread(path).astype(np.float16) for key, path in zip(keys, lbl_paths)}
train_idx, val_idx = {}, {}
np.random.seed(42)
for key in keys:
    # Create a mask for valid indices where labels are not all -1
    # -1 indicates outside of the cell
    
    valid_indices = np.where(~np.all(lbls[key] == -1, axis=(1, 2)))[0]
    total_samples = valid_indices.shape[0]
    np.random.shuffle(valid_indices)  # Shuffles in place

    # Compute split index
    split_idx = int(0.85 * total_samples)

    # Split the indices
    train_idx[key] = valid_indices[:split_idx]
    val_idx[key] = valid_indices[split_idx:]

# compute mean and std of the data
all_elements = np.concatenate([imgs[key][train_idx[key]].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements.astype(np.float32))

sample_ratio = args.sample_ratio

# normalizing the data
for key in tqdm(keys, "Normalizing data"):
    imgs[key] = (imgs[key] - data_mean) / data_std

train_set = SemisupervisedDataset(
    images=imgs,
    labels=lbls,
    patch_size=patch_size,
    label_size=initial_label_size,
    mode=mode,
    n_classes=n_classes,
    ignore_lbl=-1,
    ratio=labeled_ratio,
    indices_dict=train_idx,
)

val_set = SemisupervisedDataset(
    images=imgs,
    labels=lbls,
    patch_size=patch_size,
    label_size=initial_label_size,
    mode='supervised',
    n_classes=n_classes,
    ignore_lbl=-1,
    ratio=labeled_ratio,
    indices_dict=val_idx,
)

train_loader = DataLoader(
    train_set,
    batch_sampler=ModeAwareBalancedAnchorBatchSampler(
        train_set, total_patches_per_batch=batch_size, shuffle=True
    ),
    collate_fn=flex_collate,
)
val_loader = DataLoader(
    val_set,
    batch_sampler=ModeAwareBalancedAnchorBatchSampler(
        val_set, total_patches_per_batch=batch_size, shuffle=False
    ),
    collate_fn=flex_collate,
)


img_shape = (64, 64)

if load_checkpoint:
    model = torch.load(checkpoint, weights_only=False)
    model.update_mode("semisupervised")

else:
    model = LadderVAE(
        z_dims=z_dims,
        blocks_per_layer=blocks_per_layer,
        data_mean=data_mean,
        data_std=data_std,
        noiseModel=noiseModel,
        conv_mult=2,
        device=device,
        batchnorm=batchnorm,
        stochastic_skip=True,
        free_bits=free_bits,
        img_shape=img_shape,
        grad_checkpoint=True,
        mask_size=initial_mask_size,
        contrastive_learning=contrastive_learning,
        margin=margin,
        lambda_contrastive=lambda_contrastive,
        stochastic_block_type=stochastic_block_type,
        conditional=conditional,
        condition_type=condition_type,
        n_components=n_components,
        training_mode=mode,
        labeled_ratio=labeled_ratio,
    ).cuda()
print(model)
model.train()  # Model set in training mode

training.train_network(
    model=model,
    lr=lr,
    max_epochs=max_epochs,
    directory_path=directory_path,
    batch_size=batch_size,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    train_loader=train_loader,
    val_loader=val_loader,
    gaussian_noise_std=gaussian_noise_std,
    model_name=model_name,
    gradient_scale=256,
    use_wandb=use_wandb,
    max_grad_norm=1,
    initial_label_size=initial_label_size,
    final_label_size=final_label_size,
    initial_mask_size=initial_mask_size,
    final_mask_size=final_mask_size,
    step_interval=step_interval,
)
