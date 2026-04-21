from __future__ import annotations

import argparse
from pathlib import Path

from eps_seg.config.train import ExperimentConfig
from eps_seg.dataloaders.datamodules import EPSSegDataModule


def run_semisupervised_datamodule_test(
    exp_config_path: str | Path,
    epochs: int = 200,
    iterate_val_every: int = 1,
    increase_radius_every: int = 20,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> None:
    exp_config = ExperimentConfig.from_yaml(Path(exp_config_path))
    train_cfg, dataset_cfg, _ = exp_config.get_configs()

    dataset_cfg.mode = "semisupervised"
    dm = EPSSegDataModule(cfg=dataset_cfg, train_cfg=train_cfg)
    dm.prepare_data()
    dm.setup("fit")
    dm.set_mode("semisupervised")

    print("Semisupervised datamodule test")
    print(f"Experiment config : {exp_config_path}")
    print(f"Dataset           : {dataset_cfg.name}")
    print(f"Fold              : {dataset_cfg.fold}/{dataset_cfg.max_folds - 1}")
    print(f"Epochs            : {epochs}")
    print(f"Initial radius    : {train_cfg.initial_radius}")
    print(f"Max radius        : {train_cfg.max_radius}")
    print(f"Increase every    : {increase_radius_every} epoch(s)")
    print(f"Max train batches : {max_train_batches}")
    print(f"Max val batches   : {max_val_batches}")

    for epoch in range(epochs):
        print(
            f"\n[Test] Epoch {epoch + 1}/{epochs} | "
            f"radius={dm.train_dataset.radius} | "
            f"train_groups={len(dm.train_dataset)} | val_groups={len(dm.val_dataset)}"
        )

        train_loader = dm.train_dataloader()
        train_batches_seen = 0
        for train_batches_seen, _batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and train_batches_seen >= max_train_batches:
                break
        print(f"[Test] Train batches iterated: {train_batches_seen}")

        if iterate_val_every > 0 and epoch % iterate_val_every == 0:
            val_loader = dm.val_dataloader()
            val_batches_seen = 0
            for val_batches_seen, _batch in enumerate(val_loader, start=1):
                if max_val_batches is not None and val_batches_seen >= max_val_batches:
                    break
            print(f"[Test] Val batches iterated  : {val_batches_seen}")

        if (
            increase_radius_every > 0
            and (epoch + 1) % increase_radius_every == 0
            and dm.train_dataset.radius < train_cfg.max_radius
        ):
            next_radius = min(dm.train_dataset.radius + 1, train_cfg.max_radius)
            print(f"[Test] Increasing radius to {next_radius}")
            dm.set_radius(next_radius)

    print("\nSemisupervised datamodule test completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a semisupervised datamodule test for a configurable number of epochs."
    )
    parser.add_argument(
        "exp_config",
        type=str,
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of synthetic epochs to iterate.",
    )
    parser.add_argument(
        "--iterate-val-every",
        type=int,
        default=1,
        help="Iterate the validation dataloader every N epochs. Set to 0 to skip validation iteration.",
    )
    parser.add_argument(
        "--increase-radius-every",
        type=int,
        default=10,
        help="Increase the train dataset radius every N epochs. Set to 0 to keep radius fixed.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on train batches per synthetic epoch.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per synthetic epoch.",
    )

    args = parser.parse_args()
    run_semisupervised_datamodule_test(
        exp_config_path=args.exp_config,
        epochs=args.epochs,
        iterate_val_every=args.iterate_val_every,
        increase_radius_every=args.increase_radius_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    main()
