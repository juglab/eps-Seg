"""Centre masking for the inpainting objective.

Shared by both backends (``eps_seg_vanilla`` and ``lvae``).

Params:

- ``mask_size``     -- the SIZE of the masked centre region. 
                       (NIPS Paper Table 8, the main results use 1x1).
- ``mask_strategy`` -- the FILL rule for that region, passed here as ``ann_width``:
                       0 means fill with zeros, as in original epsSeg (zero-masking).
                       N > 0 fills with the mean of an N-wide annulus around the region.
"""

from __future__ import annotations

import torch


def apply_centre_mask(
    x: torch.Tensor,
    *,
    mask_size: int,
    ann_width: int,
    conv_mult: int,
) -> torch.Tensor:
    """
    Mask the centre region of every patch in ``x`` and return the masked copy.

    ``x`` is ``[B, C, H, W]`` for ``conv_mult == 2`` and ``[B, C, D, H, W]`` otherwise.

    With ``ann_width == 0`` the centre is set to ``0``. With
    ``ann_width > 0`` it is set to the mean of an annulus of that width around the centre,
    computed **per sample and per channel** and **excluding the masked region itself**, so the
    value being reconstructed can never leak into the masked area. If the annulus would leave the patch,
    the whole unmasked remainder is used instead.

    The fill is detached.
    """
    if mask_size < 1:
        raise ValueError(f"mask_size must be >= 1, got {mask_size}")
    if ann_width < 0:
        raise ValueError(f"ann_width (mask_strategy) must be >= 0, got {ann_width}")

    x_masked = x.clone()
    patch_size = x.shape[-1]
    begin = (patch_size - mask_size) // 2
    end = begin + mask_size

    centre = [slice(None), slice(None)] + [slice(begin, end)] * conv_mult
    mask_binary = torch.zeros_like(x, dtype=torch.bool)
    mask_binary[tuple(centre)] = True

    if ann_width == 0:
        x_masked[mask_binary] = 0.0
        return x_masked

    if ann_width <= begin:
        ring = [slice(None), slice(None)] + [slice(begin - ann_width, end + ann_width)] * conv_mult
        average_mask = torch.zeros_like(x, dtype=torch.bool)
        average_mask[tuple(ring)] = True
        average_mask &= ~mask_binary          # never let the masked region feed its own fill
    else:
        # the annulus would be larger than the patch: fall back to the whole unmasked remainder
        average_mask = ~mask_binary

    spatial = tuple(range(2, x.dim()))
    weights = average_mask.to(x.dtype)
    total = (x * weights).sum(dim=spatial, keepdim=True)
    count = weights.sum(dim=spatial, keepdim=True).clamp(min=1.0)
    fill = (total / count).detach()           # [B, C, 1, 1(, 1)] -- per sample, per channel

    return torch.where(mask_binary, fill.expand_as(x), x_masked)
