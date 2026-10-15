"""A run writes its metrics to a local CSV as well as to the primary logger.

The point is to be able to read a running experiment from disk instead of querying W&B. So:
  * the CSV sink is on by default and is additive -- the primary logger is unchanged,
  * turning it off restores the exact previous return value (a single logger),
  * the CSV lands under the experiment's own log dir, named per stage,
  * Lightning accepts the pair and actually writes a readable metrics.csv.

Fully synthetic: no dataset config, no cache, no eps_seg model.
"""

import csv
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from eps_seg.config.train import TrainConfig
from eps_seg.train import CSV_FLUSH_EVERY_N_STEPS, build_logger


class _ExpStub:
    """The three attributes build_logger actually reads."""

    def __init__(self, log_dir: Path, name="exp_demo", project="proj"):
        self.experiment_name = name
        self.project_name = project
        self._log_dir = log_dir

    def get_log_dir(self) -> Path:
        return self._log_dir


def _cfg(**overrides) -> TrainConfig:
    fields = dict(use_wandb=False)      # never touch the network in a test
    fields.update(overrides)
    return TrainConfig(**fields)


def test_csv_logger_is_on_by_default():
    assert TrainConfig(use_wandb=False).use_csv_logger is True


def test_returns_primary_plus_csv(tmp_path):
    loggers = build_logger(_ExpStub(tmp_path), _cfg(), "supervised", None)
    assert isinstance(loggers, list) and len(loggers) == 2
    assert isinstance(loggers[0], TensorBoardLogger)
    assert isinstance(loggers[1], CSVLogger)
    assert loggers[1]._flush_logs_every_n_steps == CSV_FLUSH_EVERY_N_STEPS


def test_disabling_restores_a_single_logger(tmp_path):
    logger = build_logger(_ExpStub(tmp_path), _cfg(use_csv_logger=False), "supervised", None)
    assert not isinstance(logger, list)
    assert isinstance(logger, TensorBoardLogger)


def test_csv_lands_under_the_experiment_log_dir(tmp_path):
    _, csv_logger = build_logger(_ExpStub(tmp_path), _cfg(), "supervised", None)
    assert Path(csv_logger.save_dir) == tmp_path
    assert csv_logger.name == "exp_demo_supervised"


def test_stage_naming_reaches_the_csv_sink(tmp_path):
    _, staged = build_logger(_ExpStub(tmp_path), _cfg(), "semisupervised", 3)
    assert staged.name == "exp_demo_semisupervised_K3"
    _, tagged = build_logger(_ExpStub(tmp_path), _cfg(), "supervised", "U40")
    assert tagged.name == "exp_demo_supervised_U40"


def test_both_sinks_name_the_same_run(tmp_path):
    primary, csv_logger = build_logger(_ExpStub(tmp_path), _cfg(), "supervised", None)
    assert primary.name == csv_logger.name


class _TinyModule(L.LightningModule):
    """Minimal module that logs one epoch-level metric, like the real validation hook does."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        self.log("val/total_loss", loss)
        self.log("val/dice_score_mean", torch.tensor(0.5))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_trainer_accepts_the_pair_and_writes_a_readable_csv(tmp_path):
    loggers = build_logger(_ExpStub(tmp_path), _cfg(), "supervised", None)
    data = TensorDataset(torch.randn(8, 4), torch.randn(8, 1))
    loader = DataLoader(data, batch_size=4)

    trainer = L.Trainer(
        logger=loggers,
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(_TinyModule(), train_dataloaders=loader, val_dataloaders=loader)

    metrics = list(tmp_path.glob("exp_demo_supervised/version_*/metrics.csv"))
    assert len(metrics) == 1, f"expected one metrics.csv, found {metrics}"
    with metrics[0].open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "metrics.csv is empty"
    assert "val/total_loss" in rows[0]
    assert "val/dice_score_mean" in rows[0]
    # the monitored metrics are present for both epochs, which is what makes a run readable mid-flight
    epochs = {r["epoch"] for r in rows if r.get("val/total_loss")}
    assert epochs == {"0", "1"}, epochs


def test_csv_sink_does_not_change_checkpointing_or_seeding_fields():
    """The flag must be inert outside logging."""
    on, off = TrainConfig(use_wandb=False), TrainConfig(use_wandb=False, use_csv_logger=False)
    ignored = {"use_csv_logger"}
    a = {k: v for k, v in on.model_dump().items() if k not in ignored}
    b = {k: v for k, v in off.model_dump().items() if k not in ignored}
    assert a == b
