"""One training run saves a `best` checkpoint per monitored metric.

Monitors should be compared WITHOUT retraining, so:
  * the primary metric keeps its historic filename (downstream tools unchanged),
  * each extra metric gets a distinct sibling with the correct direction,
  * early stopping is bound to exactly one metric, independent of checkpointing,
  * the per-metric ModelCheckpoints cannot restore each other's state on a resume.
"""

from pathlib import Path

import pytest

from eps_seg.config.train import TrainConfig, metric_mode, metric_tag
from eps_seg.train import build_monitor_callbacks

CVPR_METRICS = dict(
    monitored_metric="val/total_loss",
    monitored_metric_mode="min",
    monitored_metrics=["val/CE_epoch", "val/dice_score_mean"],
    early_stopping_metric="val/dice_score_mean",
)


def _cfg(**overrides):
    fields = dict(use_wandb=False, early_stopping_patience=20)
    fields.update(overrides)
    return TrainConfig(**fields)


@pytest.mark.parametrize("metric,expected", [
    ("val/total_loss", "min"), ("val/total_loss_epoch", "min"), ("val/CE_epoch", "min"),
    ("val/dice_score_mean", "max"), ("val/dice_score_class_2", "max"),
])
def test_metric_mode_direction(metric, expected):
    assert metric_mode(metric) == expected


@pytest.mark.parametrize("metric,expected", [
    ("val/total_loss", "total_loss"), ("val/CE_epoch", "CE"),
    ("val/dice_score_mean", "dice_score_mean"), ("val/total_loss_epoch", "total_loss"),
])
def test_metric_tag_is_filename_safe(metric, expected):
    assert metric_tag(metric) == expected
    assert "/" not in metric_tag(metric)


def test_single_metric_behaviour_is_unchanged(tmp_path):
    """Back-compat: with no extra metrics, exactly one checkpoint at the historic filename."""
    cfg = _cfg(monitored_metric="val/CE_epoch", monitored_metric_mode="min")
    best = tmp_path / "best_supervised.ckpt"
    ckpts, early = build_monitor_callbacks(cfg, best)

    assert len(ckpts) == 1
    assert ckpts[0].filename == "best_supervised"
    assert ckpts[0].monitor == "val/CE_epoch"
    assert early.monitor == "val/CE_epoch"


def test_three_metrics_produce_three_distinct_checkpoints(tmp_path):
    cfg = _cfg(**CVPR_METRICS)
    best = tmp_path / "best_supervised.ckpt"
    ckpts, early = build_monitor_callbacks(cfg, best)

    assert [c.monitor for c in ckpts] == ["val/total_loss", "val/CE_epoch", "val/dice_score_mean"]
    # the primary keeps the historic name; the others are siblings
    assert [c.filename for c in ckpts] == [
        "best_supervised", "best_supervised_CE", "best_supervised_dice_score_mean"
    ]
    assert len({c.filename for c in ckpts}) == 3
    assert [c.mode for c in ckpts] == ["min", "min", "max"]
    assert all(Path(c.dirpath) == tmp_path for c in ckpts)


def test_staged_layout_keeps_the_stage_index_in_sibling_names(tmp_path):
    cfg = _cfg(**CVPR_METRICS)
    ckpts, _ = build_monitor_callbacks(cfg, tmp_path / "best_semisupervised_K3.ckpt")
    assert [c.filename for c in ckpts] == [
        "best_semisupervised_K3", "best_semisupervised_K3_CE", "best_semisupervised_K3_dice_score_mean"
    ]


def test_early_stopping_is_decoupled_from_the_primary_metric(tmp_path):
    """The run stops on dice while the primary checkpoint still tracks total_loss."""
    cfg = _cfg(**CVPR_METRICS)
    ckpts, early = build_monitor_callbacks(cfg, tmp_path / "best_supervised.ckpt")

    assert early.monitor == "val/dice_score_mean"
    assert early.mode == "max"
    assert early.patience == 20
    assert ckpts[0].monitor == "val/total_loss", "primary checkpoint must be independent of stopping"


def test_early_stopping_defaults_to_primary_when_unset(tmp_path):
    cfg = _cfg(monitored_metric="val/CE_epoch", monitored_metrics=["val/dice_score_mean"])
    _, early = build_monitor_callbacks(cfg, tmp_path / "best_supervised.ckpt")
    assert early.monitor == "val/CE_epoch"


def test_per_metric_checkpoints_have_distinct_state_keys(tmp_path):
    """
    Lightning restores callback state by state_key. If two ModelCheckpoints shared one, a resume
    could load the wrong metric's best score -- the class of bug behind the Stage-2 selection mess.
    """
    cfg = _cfg(**CVPR_METRICS)
    ckpts, _ = build_monitor_callbacks(cfg, tmp_path / "best_supervised.ckpt")
    keys = [c.state_key for c in ckpts]
    assert len(set(keys)) == len(keys), keys


def test_checkpoint_directory_is_created(tmp_path):
    cfg = _cfg(**CVPR_METRICS)
    nested = tmp_path / "a" / "b"
    assert not nested.exists()
    build_monitor_callbacks(cfg, nested / "best_supervised.ckpt")
    assert nested.is_dir()


def test_duplicate_metric_is_not_checkpointed_twice(tmp_path):
    cfg = _cfg(monitored_metric="val/CE_epoch",
               monitored_metrics=["val/CE_epoch", "val/dice_score_mean"])
    ckpts, _ = build_monitor_callbacks(cfg, tmp_path / "best_supervised.ckpt")
    assert [c.monitor for c in ckpts] == ["val/CE_epoch", "val/dice_score_mean"]


# --- downstream: prediction output naming, evaluation discovery, reporting ----------------------
# C2 must not change the METHODOLOGY, only add rows. These guard that the per-metric checkpoints
# flow through prediction -> evaluation -> reporting as ordinary extra rows.

def test_prediction_output_folder_is_the_checkpoint_stem():
    """predict.py names the output dir `ckpt_path.stem`, so per-metric names stay distinct."""
    for name, expected in [
        ("best_supervised.ckpt", "best_supervised"),
        ("best_supervised_CE.ckpt", "best_supervised_CE"),
        ("best_supervised_dice_score_mean.ckpt", "best_supervised_dice_score_mean"),
    ]:
        assert Path(name).stem == expected


def test_evaluate_discovers_every_per_metric_prediction_folder(tmp_path, monkeypatch):
    from eps_seg.evaluate import _load_experiment_predictions

    class FakeExp:
        outputs_dir = tmp_path

    names = ["best_supervised", "best_supervised_CE", "best_supervised_dice_score_mean",
             "last_supervised"]
    for n in names:
        (tmp_path / "predictions" / n).mkdir(parents=True)
    (tmp_path / "predictions" / "not_a_dir.txt").write_text("x", encoding="utf-8")

    found = _load_experiment_predictions(FakeExp())
    assert sorted(found) == sorted(names), "each metric's predictions must be evaluated separately"
    assert "not_a_dir.txt" not in found


def test_dice_computation_is_independent_of_the_checkpoint_name():
    """Methodology unchanged: the metric depends only on the arrays, not on which monitor saved them."""
    import numpy as np
    from eps_seg.evaluate import _compute_dice_rows

    rng = np.random.default_rng(0)
    gt = rng.integers(0, 4, size=(32, 32))
    pred = gt.copy()
    pred[:4] = (pred[:4] + 1) % 4          # a few deliberate errors

    rows = [
        _compute_dice_rows(experiment_name="exp", checkpoint_name=name, test_key="k",
                           slice_idx=0, gt_slice=gt, pred_slice=pred, n_classes=4)
        for name in ("best_supervised", "best_supervised_CE", "best_supervised_dice_score_mean")
    ]
    scores = {r["average_dice_score"] for r in rows}
    assert len(scores) == 1, "the same arrays must score identically under any checkpoint name"
    assert {r["checkpoint"] for r in rows} == {
        "best_supervised", "best_supervised_CE", "best_supervised_dice_score_mean"
    }
