from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def build_top_prior_params(
    *,
    n_components: int,
    top_prior_param_shape: tuple[int, ...],
    conv_mult: int,
    learn_top_prior: bool,
    mu_init: float,
) -> tuple[nn.Parameter, torch.Tensor]:
    """
    Build top-layer GMM prior parameters and a mask selecting active mean slots.
    """
    total_channels = top_prior_param_shape[1]
    spatial_dims = top_prior_param_shape[-conv_mult:]
    channels_per_component = total_channels // (2 * n_components)

    chunk_values = torch.zeros(
        (n_components, channels_per_component) + spatial_dims,
        dtype=torch.float32,
    )
    chunk_mask = torch.zeros_like(chunk_values)

    chunk_size = channels_per_component // n_components
    for component_idx in range(n_components):
        start_idx = component_idx * chunk_size
        end_idx = (component_idx + 1) * chunk_size
        chunk_values[component_idx, start_idx:end_idx] = mu_init
        chunk_mask[component_idx, start_idx:end_idx] = 1.0

    mus = chunk_values.view((1, n_components * channels_per_component) + spatial_dims)
    mus_mask = chunk_mask.view((1, n_components * channels_per_component) + spatial_dims)
    sigmas = torch.zeros_like(mus)
    prior_params = nn.Parameter(
        torch.cat([mus, sigmas], dim=1),
        requires_grad=learn_top_prior,
    )
    return prior_params, mus_mask


class TopPriorSchedulingMixin:
    """
    Shared top-prior scheduling behavior for top stochastic layers.

    Expected attributes on subclasses:
    - ``is_top_layer``
    - ``top_prior_params``
    - ``top_prior_mu_mask``
    - ``training_mode``
    """

    def _init_top_prior_schedule(
        self,
        *,
        schedule: str,
        mu_init: float,
        mu_supervised_max: Optional[float],
        mu_semisupervised_min: Optional[float],
        mu_step_epochs: int,
    ) -> None:
        self.top_prior_schedule = schedule
        self.top_prior_mu_init = float(mu_init)
        self.top_prior_mu_supervised_max = mu_supervised_max
        self.top_prior_mu_semisupervised_min = mu_semisupervised_min
        self.top_prior_mu_step_epochs = int(mu_step_epochs)
        self._semisupervised_start_mu: Optional[float] = None
        self._schedule_phase_start_epoch: Optional[int] = None
        self._schedule_phase_start_mu: Optional[float] = None
        self._schedule_phase_mode: Optional[str] = None

    def _get_current_top_prior_mu_value(self) -> Optional[float]:
        if not getattr(self, "is_top_layer", False):
            return None
        mu_channels = self.top_prior_params.shape[1] // 2
        mus = self.top_prior_params[:, :mu_channels]
        mask = self.top_prior_mu_mask > 0
        if mask.any():
            return float(mus[mask].mean().item())
        return self.top_prior_mu_init

    def _set_top_prior_mu_value(self, value: float) -> None:
        mu_channels = self.top_prior_params.shape[1] // 2
        with torch.no_grad():
            self.top_prior_params[:, :mu_channels].copy_(self.top_prior_mu_mask * value)

    def update_top_prior_scheduler(self, epoch: int) -> None:
        if not getattr(self, "is_top_layer", False):
            return

        schedule = getattr(self, "top_prior_schedule", "none")
        if schedule in {"none", "static"}:
            return
        if schedule != "dynamic":
            raise ValueError(f"Unknown top-prior schedule: {schedule}")

        current_value = self._get_current_top_prior_mu_value()
        if current_value is None:
            return

        if self._schedule_phase_mode != self.training_mode:
            self._schedule_phase_mode = self.training_mode
            self._schedule_phase_start_epoch = int(epoch)
            self._schedule_phase_start_mu = current_value

        phase_start_epoch = int(self._schedule_phase_start_epoch or 0)
        phase_start_mu = float(self._schedule_phase_start_mu if self._schedule_phase_start_mu is not None else current_value)
        step = max(0, int(epoch) - phase_start_epoch) // self.top_prior_mu_step_epochs
        if self.training_mode == "supervised":
            target_value = min(
                float(self.top_prior_mu_supervised_max),
                phase_start_mu + float(step),
            )
            self._semisupervised_start_mu = None
        elif self.training_mode == "semisupervised":
            target_value = max(
                float(self.top_prior_mu_semisupervised_min),
                phase_start_mu - float(step),
            )
        else:
            return

        if abs(current_value - target_value) > 1e-6:
            self._set_top_prior_mu_value(target_value)

    def get_top_prior_mu_value(self) -> Optional[float]:
        return self._get_current_top_prior_mu_value()
