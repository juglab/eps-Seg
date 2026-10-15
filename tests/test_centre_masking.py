"""Centre masking (`mask_strategy`) — 2D and 3D, pure tensors, no dataset or cache.

Guards three properties that previously went wrong:
  1. zero-fill when mask_strategy == 0 (the published behaviour)
  2. the masked region can never contribute to its own fill (no leak)
  3. the annulus mean is PER SAMPLE and PER CHANNEL, not one scalar for the whole batch
"""

import pytest
import torch

from eps_seg.modules.masking import apply_centre_mask


def centre_index(patch_size: int, mask_size: int = 1) -> int:
    return (patch_size - mask_size) // 2


def centre_values(x: torch.Tensor, conv_mult: int, mask_size: int = 1) -> torch.Tensor:
    b = centre_index(x.shape[-1], mask_size)
    e = b + mask_size
    sl = [slice(None), slice(None)] + [slice(b, e)] * conv_mult
    return x[tuple(sl)].reshape(x.shape[0], x.shape[1], -1)


@pytest.mark.parametrize("conv_mult", [2, 3])
def test_zero_strategy_fills_zeros_and_leaves_rest_untouched(conv_mult):
    ps = 9
    shape = (2, 1) + (ps,) * conv_mult
    x = torch.randn(*shape) + 5.0
    out = apply_centre_mask(x, mask_size=1, ann_width=0, conv_mult=conv_mult)
    assert torch.all(centre_values(out, conv_mult) == 0.0)
    # everything outside the centre is unchanged
    b = centre_index(ps)
    sl = [slice(None), slice(None)] + [slice(b, b + 1)] * conv_mult
    ref, got = x.clone(), out.clone()
    ref[tuple(sl)] = 0.0
    got[tuple(sl)] = 0.0
    assert torch.equal(ref, got)


@pytest.mark.parametrize("conv_mult", [2, 3])
def test_centre_cannot_leak_into_its_own_fill(conv_mult):
    """Ring is 5.0, centre is 1000.0 -> fill must be exactly 5.0, never influenced by 1000."""
    ps, aw = 9, 2
    shape = (1, 1) + (ps,) * conv_mult
    x = torch.zeros(*shape)
    b = centre_index(ps)
    ring = [slice(None), slice(None)] + [slice(b - aw, b + 1 + aw)] * conv_mult
    x[tuple(ring)] = 5.0
    ctr = [slice(None), slice(None)] + [slice(b, b + 1)] * conv_mult
    x[tuple(ctr)] = 1000.0
    out = apply_centre_mask(x, mask_size=1, ann_width=aw, conv_mult=conv_mult)
    assert torch.allclose(centre_values(out, conv_mult), torch.tensor(5.0))


@pytest.mark.parametrize("conv_mult", [2, 3])
def test_annulus_mean_is_per_sample(conv_mult):
    """Two samples with different rings must receive DIFFERENT fills (not a batch-global scalar)."""
    ps, aw = 9, 2
    shape = (2, 1) + (ps,) * conv_mult
    x = torch.zeros(*shape)
    b = centre_index(ps)
    ring = [slice(None), slice(None)] + [slice(b - aw, b + 1 + aw)] * conv_mult
    x[tuple(ring)] = 1.0
    x[1] = x[1] * 0.0 + 0.0
    x[(1,) + tuple(ring[1:])] = 101.0
    ctr = [slice(None), slice(None)] + [slice(b, b + 1)] * conv_mult
    x[tuple(ctr)] = -99.0
    out = centre_values(apply_centre_mask(x, mask_size=1, ann_width=aw, conv_mult=conv_mult), conv_mult)
    assert torch.allclose(out[0].reshape(-1), torch.tensor(1.0)), out[0]
    assert torch.allclose(out[1].reshape(-1), torch.tensor(101.0)), out[1]
    # and explicitly NOT the batch-global mean, which is what the old implementation produced
    assert not torch.allclose(out[0].reshape(-1), torch.tensor(51.0))


@pytest.mark.parametrize("conv_mult", [2, 3])
def test_annulus_mean_is_per_channel(conv_mult):
    ps, aw = 9, 1
    shape = (1, 2) + (ps,) * conv_mult
    x = torch.zeros(*shape)
    b = centre_index(ps)
    ring = [slice(None), slice(None)] + [slice(b - aw, b + 1 + aw)] * conv_mult
    x[tuple(ring)] = 1.0
    x[(0, 1) + tuple(s for s in ring[2:])] = 7.0
    out = centre_values(apply_centre_mask(x, mask_size=1, ann_width=aw, conv_mult=conv_mult), conv_mult)
    assert torch.allclose(out[0, 0], torch.tensor(1.0))
    assert torch.allclose(out[0, 1], torch.tensor(7.0))


@pytest.mark.parametrize("conv_mult", [2, 3])
def test_oversized_annulus_falls_back_to_whole_unmasked_region(conv_mult):
    """ann_width > begin must use every unmasked voxel, still excluding the centre."""
    ps = 5
    shape = (1, 1) + (ps,) * conv_mult
    x = torch.full(shape, 3.0)
    b = centre_index(ps)
    ctr = [slice(None), slice(None)] + [slice(b, b + 1)] * conv_mult
    x[tuple(ctr)] = 500.0
    out = apply_centre_mask(x, mask_size=1, ann_width=99, conv_mult=conv_mult)
    assert torch.allclose(centre_values(out, conv_mult), torch.tensor(3.0))


@pytest.mark.parametrize("conv_mult", [2, 3])
def test_larger_mask_size(conv_mult):
    ps, ms, aw = 11, 3, 2
    shape = (1, 1) + (ps,) * conv_mult
    x = torch.zeros(*shape)
    b = centre_index(ps, ms)
    ring = [slice(None), slice(None)] + [slice(b - aw, b + ms + aw)] * conv_mult
    x[tuple(ring)] = 4.0
    ctr = [slice(None), slice(None)] + [slice(b, b + ms)] * conv_mult
    x[tuple(ctr)] = -1.0
    out = apply_centre_mask(x, mask_size=ms, ann_width=aw, conv_mult=conv_mult)
    got = centre_values(out, conv_mult, mask_size=ms)
    assert torch.allclose(got, torch.full_like(got, 4.0))


def test_input_is_not_mutated():
    x = torch.randn(2, 1, 9, 9)
    before = x.clone()
    apply_centre_mask(x, mask_size=1, ann_width=2, conv_mult=2)
    assert torch.equal(x, before)


def test_fill_creates_no_gradient_path():
    x = torch.randn(1, 1, 9, 9, requires_grad=True)
    out = apply_centre_mask(x, mask_size=1, ann_width=2, conv_mult=2)
    centre_values(out, 2).sum().backward()
    assert x.grad is None or torch.all(x.grad == 0.0)


@pytest.mark.parametrize("bad", [{"mask_size": 0}, {"ann_width": -1}])
def test_invalid_arguments_raise(bad):
    kwargs = {"mask_size": 1, "ann_width": 0, "conv_mult": 2} | bad
    with pytest.raises(ValueError):
        apply_centre_mask(torch.zeros(1, 1, 9, 9), **kwargs)
