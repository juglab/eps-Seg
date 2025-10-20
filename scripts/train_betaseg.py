# mlproject/scripts/train_betaseg.py
import os, argparse, numpy as np, torch, tifffile as tiff
from torch.utils.data import DataLoader
from tqdm import tqdm

from eps_seg.modules.lvae import LadderVAE
from eps_seg.dataloaders.datasets import SemisupervisedDataset
from eps_seg.dataloaders.samplers import ModeAwareBalancedAnchorBatchSampler
from eps_seg.dataloaders.utils import flex_collate

from eps_seg.config.train import TrainConfig
from eps_seg.train import Trainer
from eps_seg.train.callbacks import (
                                            ModelCheckpoint,
                                            EarlyStopping,
                                            ReduceLROnPlateauStep,
                                            LabelMaskSizeScheduler,
                                            ThresholdScheduler,
                                            PlateauActions,
                                            NaNDetector,
                                            WandbLogger,
                                    )


def build_data(args):
    # use provided image/labels or default
    if args.image and args.labels:
        imgs = {"plugin": tiff.imread(args.image).astype(np.float16)}
        lbls = {"plugin": tiff.imread(args.labels).astype(np.float16)}
        keys = ["plugin"]
    else:
        data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
        keys = ["high_c1", "high_c2", "high_c3"]
        img_paths = [os.path.join(data_dir + k + f"/{k}_source.tif") for k in keys]
        lbl_paths = [os.path.join(data_dir + k + f"/{k}_gt.tif") for k in keys]
        imgs = {k: tiff.imread(p).astype(np.float16) for k, p in zip(keys, img_paths)}
        lbls = {k: tiff.imread(p).astype(np.float16) for k, p in zip(keys, lbl_paths)}

    # split by slice index where label not all -1
    train_idx, val_idx = {}, {}
    rng = np.random.default_rng(42)
    for k in keys:
        valid = np.where(~np.all(lbls[k] == -1, axis=(1, 2)))[0]
        rng.shuffle(valid)
        split = int(0.85 * len(valid))
        train_idx[k] = valid[:split]
        val_idx[k] = valid[split:]

    # global mean/std computed on train
    all_elements = np.concatenate([imgs[k][train_idx[k]].ravel() for k in keys])
    data_mean = np.mean(all_elements)
    data_std = np.std(all_elements.astype(np.float32))
    for k in tqdm(keys, "Normalizing data"):
        imgs[k] = (imgs[k] - data_mean) / data_std

    patch_size = 64
    n_classes = 4

    train_set = SemisupervisedDataset(
        images=imgs,
        labels=lbls,
        patch_size=patch_size,
        label_size=args.initial_label_size,
        mode=args.mode,
        n_classes=n_classes,
        ignore_lbl=-1,
        ratio=args.labeled_ratio,
        indices_dict=train_idx,
    )
    val_set = SemisupervisedDataset(
        images=imgs,
        labels=lbls,
        patch_size=patch_size,
        label_size=args.initial_label_size,
        mode="supervised",
        n_classes=n_classes,
        ignore_lbl=-1,
        ratio=args.labeled_ratio,
        indices_dict=val_idx,
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=ModeAwareBalancedAnchorBatchSampler(
            train_set, total_patches_per_batch=args.batch_size, shuffle=True
        ),
        collate_fn=flex_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=ModeAwareBalancedAnchorBatchSampler(
            val_set, total_patches_per_batch=args.batch_size, shuffle=False
        ),
        collate_fn=flex_collate,
    )
    return train_loader, val_loader, data_mean, data_std


def build_model(args, data_mean, data_std, device):
    z_dims = [32] * int(args.num_latents)
    model = LadderVAE(
        z_dims=z_dims,
        blocks_per_layer=args.blocks_per_layer,
        data_mean=data_mean,
        data_std=data_std,
        noiseModel=None,
        conv_mult=2,
        device=device,
        batchnorm=True,
        stochastic_skip=True,
        free_bits=0.0,
        img_shape=(64, 64),
        grad_checkpoint=True,
        mask_size=args.initial_mask_size,
        contrastive_learning=args.contrastive_learning,
        margin=50,
        lambda_contrastive=0.5,
        stochastic_block_type=args.stochastic_block_type,
        conditional=args.conditional,
        condition_type=args.condition_type,
        n_components=4,
        training_mode=args.mode,
        labeled_ratio=args.labeled_ratio,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument(
        "--directory_path", type=str, default="./results/test_refactoring/"
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--use_wandb", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, data_mean, data_std = build_data(args)

    if args.load_checkpoint:
        model_path = os.path.join(args.directory_path, "model_supervised", "best.net")
        model = torch.load(model_path, weights_only=False).to(device)
        if hasattr(model, "update_mode"):
            model.update_mode("semisupervised")
    else:
        model = build_model(args, data_mean, data_std, device)

    cfg = TrainConfig(
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        amp=True,
        gradient_scale=256,
        max_grad_norm=1.0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_wandb=args.use_wandb,
    )

    model_folder = os.path.join(
        args.directory_path, f"model_{getattr(model, 'training_mode', 'train')}/"
    )
    os.makedirs(model_folder, exist_ok=True)

    # callbacks: mirror your current behaviors
    cbs = [
        WandbLogger(
            use_wandb=cfg.use_wandb,
            project="refactor_test",
            config={
                "learning rate": cfg.lr,
                "epochs": cfg.max_epochs,
                "batch size": cfg.batch_size,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "gamma": cfg.gamma,
            },
        ),
        ModelCheckpoint(
            dirpath=model_folder, monitor="val_total", mode="min", top_k=1, save_last=True
        ),
        EarlyStopping(monitor="val_total", mode="min", patience=51, min_delta=1e-6),
        ReduceLROnPlateauStep(monitor="val_total"),
        LabelMaskSizeScheduler(
            args.initial_label_size,
            args.final_label_size,
            args.initial_mask_size,
            args.final_mask_size,
            args.step_interval,
        ),
        ThresholdScheduler(
            start=0.50, max_val=0.99, step=0.005, only_in_mode="semisupervised"
        ),
        PlateauActions(dirpath=model_folder),
        NaNDetector(),
    ]

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        callbacks=cbs,
        gaussian_noise_std=None,
        directory_path=model_folder,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
