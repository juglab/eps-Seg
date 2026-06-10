import pytest
import torch

from eps_seg.config.models import LVAEConfig
from eps_seg.modules.lvae.layers import ConditionalPosterior


def test_eps_seg_plus_accepts_feature_spatial_size_ablations():
    for feature_spatial_size in ([32, 16, 8], [16, 8, 4], [8, 8, 8]):
        cfg = LVAEConfig(
            architecture="eps_seg_plus",
            n_layers=3,
            n_filters=[64, 64, 64],
            feature_spatial_size=list(feature_spatial_size),
        )

        assert cfg.feature_spatial_size == list(feature_spatial_size)


def test_eps_seg_plus_default_uses_all_three_natural_head_sizes():
    cfg = LVAEConfig(
        architecture="eps_seg_plus",
        n_layers=3,
        n_filters=[64, 64, 64],
        img_shape=[64, 64],
    )

    assert cfg.feature_spatial_size == [32, 16, 8]


def test_eps_seg_plus_rejects_disabled_segmentation_heads():
    with pytest.raises(ValueError, match="eps_seg_plus requires all"):
        LVAEConfig(
            architecture="eps_seg_plus",
            n_layers=3,
            n_filters=[64, 64, 64],
            feature_spatial_size=[0, 0, 8],
        )


def test_eps_seg_vanilla_only_accepts_default_feature_spatial_size():
    cfg = LVAEConfig(
        architecture="eps_seg_vanilla",
        n_layers=3,
        n_filters=[64, 64, 64],
        feature_spatial_size=[0, 0, 8],
    )

    assert cfg.feature_spatial_size == [0, 0, 8]

    with pytest.raises(ValueError, match="eps_seg_vanilla only supports"):
        LVAEConfig(
            architecture="eps_seg_vanilla",
            n_layers=3,
            n_filters=[64, 64, 64],
            feature_spatial_size=[32, 16, 8],
        )


def test_feature_spatial_size_values_must_be_non_negative():
    with pytest.raises(ValueError, match="feature_spatial_size values must be >= 0"):
        LVAEConfig(
            architecture="eps_seg_plus",
            n_layers=3,
            n_filters=[64, 64, 64],
            feature_spatial_size=[32, -1, 8],
        )


def test_plus_conditional_posterior_crops_classifier_input_only():
    layer = ConditionalPosterior(
        c_in=4,
        c_vars=2,
        n_components=3,
        conv_mult=2,
        seg_head_dim=4,
    )
    x = torch.randn(2, 4, 8, 8)

    qz_params, class_logits = layer(x)

    assert qz_params.shape == (2, 4, 8, 8)
    assert class_logits.shape == (2, 3)
