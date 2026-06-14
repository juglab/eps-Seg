from __future__ import annotations

from typing import Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from eps_seg.config import LVAEConfig
from eps_seg.modules.lvae.layers import (
    BlurPool,
    BottomUpDeterministicResBlock,
    MergeLayer,
    TopDownDeterministicResBlock,
)
from eps_seg.modules.lvae.likelihoods import GaussianLikelihood
from eps_seg.modules.lvae.utils import (
    Interpolate,
    compute_cl_loss,
    compute_kl_loss,
    crop_img_tensor,
    cross_entropy_from_probs,
    pad_img_tensor,
)
from eps_seg.modules.top_prior import TopPriorSchedulingMixin, build_top_prior_params


class VanillaTopDownLayer(TopPriorSchedulingMixin, nn.Module):
    def __init__(
        self,
        layer_number: int,
        z_dim: int,
        seg_head_dim: int,
        n_res_blocks: int,
        n_filters: int,
        is_top_layer: bool = False,
        downsampling_steps: Optional[int] = None,
        conv_mult: int = 2,
        nonlin=None,
        skip_connection_merge_type: Optional[str] = None,
        batchnorm: bool = True,
        dropout: Optional[float] = None,
        enable_top_down_residual: bool = False,
        skip_connection: bool = True,
        res_block_type: Optional[str] = None,
        gated=None,
        grad_checkpoint: bool = False,
        learn_top_prior: bool = False,
        top_prior_param_shape=None,
        n_components: int = 4,
        training_mode: str = "supervised",
        top_prior_schedule: str = "none",
        top_prior_mu_init: float = 10.0,
        top_prior_mu_supervised_max: Optional[float] = None,
        top_prior_mu_semisupervised_min: Optional[float] = None,
        top_prior_mu_step_epochs: int = 10,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.training_mode = training_mode
        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.seg_head_dim = seg_head_dim
        self.enable_top_down_residual = enable_top_down_residual
        self.learn_top_prior = learn_top_prior
        self.n_components = n_components
        self.top_prior_param_shape = top_prior_param_shape
        self.skip_connection = skip_connection
        self.conv_mult = conv_mult
        self._init_top_prior_schedule(
            schedule=top_prior_schedule,
            mu_init=top_prior_mu_init,
            mu_supervised_max=top_prior_mu_supervised_max,
            mu_semisupervised_min=top_prior_mu_semisupervised_min,
            mu_step_epochs=top_prior_mu_step_epochs,
        )

        if self.is_top_layer:
            self.top_prior_params = self._get_top_prior_params()

        dws_left = downsampling_steps
        block_list = []
        for _ in range(n_res_blocks):
            do_resample = False
            if dws_left > 0:
                do_resample = True
                dws_left -= 1

            block_list.append(
                TopDownDeterministicResBlock(
                    n_filters,
                    n_filters,
                    conv_mult=conv_mult,
                    nonlin=nonlin,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                    grad_checkpoint=grad_checkpoint,
                )
            )
        self.deterministic_block = nn.Sequential(*block_list)

        if is_top_layer:
            self.stochastic = VanillaMixtureStochasticConvBlock(
                layer_number=layer_number,
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                conv_mult=conv_mult,
                n_components=self.n_components,
                training_mode=training_mode,
                seg_head_dim=self.seg_head_dim,
            )
        else:
            self.stochastic = VanillaNormalStochasticConvBlock(
                layer_number=layer_number,
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                conv_mult=conv_mult,
                training_mode=training_mode,
            )
            self.skip_connection_merger = MergeLayer(
                channels=n_filters,
                merge_type=skip_connection_merge_type,
                conv_mult=conv_mult,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                grad_checkpoint=grad_checkpoint,
            )
            if enable_top_down_residual:
                self.top_down_residual = MergeLayer(
                    channels=n_filters,
                    merge_type="residual",
                    conv_mult=conv_mult,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    grad_checkpoint=grad_checkpoint,
                )

    def update_mode(self, mode):
        self.training_mode = mode
        if mode == "semisupervised":
            self._semisupervised_start_mu = None
        self._schedule_phase_start_epoch = None
        self._schedule_phase_start_mu = None
        self._schedule_phase_mode = None
        self.stochastic.update_mode(mode)

    def _get_top_prior_params(self) -> nn.Parameter:
        if not self.is_top_layer:
            raise ValueError("Top prior can only be initialized for the top mixture layer.")
        prior_params, mu_mask = build_top_prior_params(
            n_components=self.n_components,
            top_prior_param_shape=self.top_prior_param_shape,
            conv_mult=self.conv_mult,
            learn_top_prior=self.learn_top_prior,
            mu_init=self.top_prior_mu_init,
        )
        self.register_buffer("top_prior_mu_mask", mu_mask, persistent=False)
        return prior_params

    def forward(
        self,
        label,
        input_=None,
        skip_connection_input=None,
        inference_mode=False,
        bu_value=None,
        n_img_prior=None,
        use_mode=False,
        force_constant_output=False,
        forced_latent=None,
    ):
        if use_mode:
            print("TODO: use_mode is not implemented yet")
        if force_constant_output:
            print("TODO: force_constant_output is not implemented yet")
        if forced_latent is not None:
            print("TODO: forced_latent is not implemented yet")

        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        if self.is_top_layer:
            p_params = self.top_prior_params
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, *[-1] * len(p_params.shape[1:]))
        else:
            p_params = input_

        if inference_mode:
            if self.is_top_layer:
                q_params = bu_value
            else:
                if self.skip_connection:
                    q_params = self.skip_connection_merger(bu_value, p_params)
                else:
                    q_params = p_params
        else:
            q_params = None

        x, data_stoch = self.stochastic(p_params=p_params, q_params=q_params)

        if self.enable_top_down_residual and not self.is_top_layer:
            x = self.top_down_residual(x, skip_connection_input)

        x_pre_residual = x
        x = self.deterministic_block(x)
        data = {k: v for k, v in data_stoch.items()}
        return x, x_pre_residual, data


class VanillaBottomUpLayer(nn.Module):
    def __init__(
        self,
        layer_number: int,
        n_res_blocks: int,
        n_filters: int,
        downsampling_steps: int = 0,
        conv_mult: int = 2,
        nonlin=None,
        batchnorm: bool = True,
        dropout: Optional[float] = None,
        res_block_type: Optional[str] = None,
        gated=None,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        bu_blocks = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1

            bu_blocks.append(
                BottomUpDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    conv_mult=conv_mult,
                    nonlin=nonlin,
                    downsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                    grad_checkpoint=grad_checkpoint,
                )
            )
        self.net = nn.Sequential(*bu_blocks)
        self.layer_number = layer_number

    def forward(self, x):
        return self.net(x)


class VanillaBaseStochasticConvBlock(nn.Module):
    def __init__(
        self,
        layer_number,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        training_mode="supervised",
    ):
        super().__init__()
        self.layer_number = layer_number
        self.training_mode = training_mode
        assert kernel % 2 == 1
        self.pad = kernel // 2
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self.batch_size = 0
        self.conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(
            nn, f"Conv{conv_mult}d"
        )

    def update_mode(self, mode):
        self.training_mode = mode

    def _clamp_params(self, params):
        mu, lv = params.chunk(2, dim=1)
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        lv = torch.clamp(lv, min=-10.0, max=10.0)
        std = torch.where(lv < 0, (lv / 2).exp(), 1 + lv)
        return mu, lv, std


class VanillaNormalStochasticConvBlock(VanillaBaseStochasticConvBlock):
    def __init__(
        self,
        layer_number,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        training_mode="supervised",
    ):
        super().__init__(layer_number, c_in, c_vars, c_out, conv_mult, kernel, training_mode)
        self.conv_in_q = self.conv_type(c_in, 2 * c_vars, kernel, padding=self.pad)
        self.conv_out = self.conv_type(c_vars, c_out, kernel, padding=self.pad)

    def forward(self, p_params, q_params):
        self.batch_size = q_params.shape[0]
        p_mu, _, p_std = self._clamp_params(p_params)
        p = Normal(p_mu, p_std)
        q_params = self.conv_in_q(q_params)
        q_mu, _, q_std = self._clamp_params(q_params)
        q = Normal(q_mu, q_std)
        z = q.rsample()
        out = self.conv_out(z)
        return out, {
            "prior": p,
            "posterior": q,
            "mu": q_mu,
            "z": z,
            "class_logits": None,
            "class_probabilities": None,
        }


class VanillaMixtureStochasticConvBlock(VanillaBaseStochasticConvBlock):
    def __init__(
        self,
        layer_number,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        training_mode="supervised",
        n_components=4,
        seg_head_dim=2,
    ):
        super().__init__(layer_number, c_in, c_vars, c_out, conv_mult, kernel, training_mode)
        self.n_components = n_components
        self.seg_head_dim = seg_head_dim
        self.conditional_layer = ConditionalPrior(
            c_in=c_in,
            c_vars=c_vars,
            n_components=n_components,
            conv_mult=conv_mult,
            kernel=kernel,
            seg_head_dim=seg_head_dim,
        )
        self.conv_out = self.conv_type(c_vars, c_out, kernel, padding=self.pad)

    def forward(self, p_params, q_params):
        self.batch_size = q_params.shape[0]
        p_mu, _, p_std = self._clamp_params(p_params)
        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)
        p_components = [Normal(mu, std) for mu, std in zip(p_mu_chunks, p_std_chunks)]

        qz_params, class_logits, class_probs = self.conditional_layer(q_params)
        q_mu, _, q_std = self._clamp_params(qz_params)
        q = Normal(q_mu, q_std)
        z = q.rsample()
        out = self.conv_out(z)

        return out, {
            "prior": p_components,
            "posterior": q,
            "mu": q_mu,
            "z": z,
            "class_logits": class_logits,
            "class_probabilities": class_probs,
        }


class FeatureSubsetSelectionLayer(nn.Module):
    def __init__(
        self,
        layer_number: int,
        crop_size: Optional[Tuple[int, ...] | int] = None,
        enabled: bool = True,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.crop_size = crop_size
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.crop_size:
            return x
        spatial_dims = x.shape[2:]
        if isinstance(self.crop_size, int):
            crop_size = [self.crop_size] * len(spatial_dims)
        else:
            crop_size = list(self.crop_size)
        starts = [(sd - cs) // 2 for sd, cs in zip(spatial_dims, crop_size)]
        ends = [s + cs for s, cs in zip(starts, crop_size)]
        slices = [slice(None), slice(None)]
        slices += [slice(s, e) for s, e in zip(starts, ends)]
        return x[tuple(slices)]


class EmptyFeatures(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        spatial_rank = x.dim() - 2
        return x.new_empty((batch_size, channels) + (0,) * spatial_rank)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        conv_mult: int,
        hidden_channels: Optional[int] = None,
        kernel: int = 1,
        spatial_size: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        assert kernel % 2 == 1

        if hidden_channels is None:
            hidden_channels = in_channels

        conv_type = getattr(nn, f"Conv{conv_mult}d")
        self.level_nets = nn.ModuleList([])
        for size in spatial_size:
            if size > 0:
                self.level_nets.append(
                    nn.Sequential(
                        conv_type(in_channels, hidden_channels, kernel_size=kernel),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.level_nets.append(EmptyFeatures())

        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, x) -> torch.Tensor:
        flattened = []
        for feature_map, net in zip(x, self.level_nets):
            out = net(feature_map)
            out = out.flatten(start_dim=1)
            flattened.append(out)
        feat = torch.cat(flattened, dim=1)
        return self.classifier(feat)


class ConditionalPrior(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_vars: int,
        n_components: int,
        conv_mult: int,
        kernel: int = 3,
        seg_head_dim: int = 2,
    ):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2

        self.n_components = n_components
        self.seg_head_dim = seg_head_dim
        self.conv_mult = conv_mult
        self.conv_type = getattr(nn, f"Conv{conv_mult}d")

        self.qy_x = nn.Sequential(
            self.conv_type(c_in, c_vars, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(c_vars * (seg_head_dim**conv_mult), n_components),
        )
        self.gamma_layer = nn.Linear(n_components, c_in)
        self.beta_layer = nn.Linear(n_components, c_in)
        self.qz_xy = nn.Sequential(
            self.conv_type(c_in, 2 * c_vars, kernel, padding=pad),
            nn.ReLU(inplace=True),
            self.conv_type(2 * c_vars, 2 * c_vars, kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        class_logits = self.qy_x(x)
        class_probs = F.softmax(class_logits / temperature, dim=-1)
        gamma = self.gamma_layer(class_probs)
        beta = self.beta_layer(class_probs)
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        x_mod = gamma * x + beta
        qz_params = self.qz_xy(x_mod)
        return qz_params, class_logits, class_probs


class EpsSegVanilla(nn.Module):
    def __init__(self, cfg: LVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.training_mode = cfg.training_mode

        self.n_layers = cfg.n_layers
        self.z_dims = cfg.z_dims
        self.blocks_per_layer = cfg.blocks_per_layer
        self.conv_mult = cfg.conv_mult
        self.conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(
            nn, f"Conv{self.conv_mult}d"
        )
        self.nonlin: Type[Union[nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU]] = getattr(
            nn, cfg.nonlin
        )

        self.enable_top_down_residuals = cfg.enable_top_down_residuals
        self.skip_connections = cfg.skip_connections
        self.skip_connections_merge_type = cfg.skip_connections_merge_type
        self.batchnorm = cfg.use_batchnorm
        self.color_ch = cfg.color_channels
        self.n_filters = int(cfg.n_filters[0])
        self.dropout = cfg.dropout
        self.kl_free_bits = cfg.kl_free_bits
        self.learn_top_prior = cfg.learn_top_prior
        self.res_block_type = cfg.res_block_type
        self.use_gated_convs = cfg.use_gated_convs
        self.use_grad_checkpoint = cfg.grad_checkpoint
        self.no_initial_downscaling = cfg.no_initial_downscaling
        self.mask_size = cfg.mask_size
        self.use_contrastive_learning = cfg.use_contrastive_learning
        self.margin = cfg.margin
        self.n_components = cfg.n_components
        self.learnable_thetas = True
        self.seg_features = cfg.seg_features
        self.feature_spatial_size = cfg.feature_spatial_size
        self.input_array_shape = cfg.img_shape

        self.build_architecture()

    def build_architecture(self):
        self.downsample = [1] * self.n_layers
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not self.no_initial_downscaling:
            self.overall_downscale_factor *= 2

        stride = 1 if self.no_initial_downscaling else 2
        self.first_bottom_up = nn.Sequential(
            self.conv_type(self.color_ch, self.n_filters, 5, padding=2, stride=1),
            BlurPool(self.n_filters, stride=stride, dim=self.conv_mult),
            self.nonlin(),
            BottomUpDeterministicResBlock(
                c_in=self.n_filters,
                c_out=self.n_filters,
                conv_mult=self.conv_mult,
                nonlin=self.nonlin,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                res_block_type=self.res_block_type,
                grad_checkpoint=self.use_grad_checkpoint,
            ),
        )

        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])
        self.feature_selection_layers = nn.ModuleList([])
        self.head_z_dims = int(self.input_array_shape[-1] / (2**self.n_layers))

        for layer_idx in range(self.n_layers):
            is_top = layer_idx == self.n_layers - 1
            self.bottom_up_layers.append(
                VanillaBottomUpLayer(
                    layer_number=layer_idx,
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
                    downsampling_steps=self.downsample[layer_idx],
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.use_gated_convs,
                    grad_checkpoint=self.use_grad_checkpoint,
                )
            )

            crop_size = self.feature_spatial_size[layer_idx]
            if crop_size > 0:
                self.feature_selection_layers.append(
                    FeatureSubsetSelectionLayer(
                        layer_number=layer_idx,
                        crop_size=crop_size,
                        enabled=True,
                    )
                )
            else:
                self.feature_selection_layers.append(EmptyFeatures())

            self.top_down_layers.append(
                VanillaTopDownLayer(
                    layer_number=layer_idx,
                    z_dim=self.z_dims[layer_idx],
                    seg_head_dim=self.head_z_dims,
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[layer_idx],
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    skip_connection_merge_type=self.skip_connections_merge_type,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    enable_top_down_residual=self.enable_top_down_residuals[layer_idx],
                    skip_connection=self.skip_connections[layer_idx],
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(dim=self.conv_mult),
                    res_block_type=self.res_block_type,
                    gated=self.use_gated_convs,
                    grad_checkpoint=self.use_grad_checkpoint,
                    n_components=self.n_components,
                    training_mode=self.training_mode,
                    top_prior_schedule=self.cfg.top_prior_schedule,
                    top_prior_mu_init=self.cfg.top_prior_mu_init,
                    top_prior_mu_supervised_max=self.cfg.top_prior_mu_supervised_max,
                    top_prior_mu_semisupervised_min=self.cfg.top_prior_mu_semisupervised_min,
                    top_prior_mu_step_epochs=self.cfg.top_prior_mu_step_epochs,
                )
            )

        modules = []
        if not self.no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for _ in range(self.blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=self.n_filters,
                    c_out=self.n_filters,
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    res_block_type=self.res_block_type,
                    gated=self.use_gated_convs,
                    grad_checkpoint=self.use_grad_checkpoint,
                )
            )

        self.final_top_down = nn.Sequential(*modules)
        seg_channel = self.n_filters // 2 if self.seg_features == "mu" else self.n_filters
        self.segmentation_head = SegmentationHead(
            in_channels=seg_channel,
            n_classes=self.n_components,
            conv_mult=self.conv_mult,
            hidden_channels=seg_channel,
            kernel=1,
            spatial_size=self.feature_spatial_size,
        )
        self.likelihood = GaussianLikelihood(
            self.n_filters,
            self.color_ch,
            self.conv_mult,
        )

    def update_mode(self, mode):
        self.training_mode = mode
        for layer in self.top_down_layers:
            layer.update_mode(mode)

    def update_top_prior_scheduler(self, epoch: int):
        for layer in self.top_down_layers:
            layer.update_top_prior_scheduler(epoch)

    def get_top_prior_mu_value(self):
        for layer in reversed(self.top_down_layers):
            mu_value = layer.get_top_prior_mu_value()
            if mu_value is not None:
                return mu_value
        return None

    @property
    def global_step(self) -> int:
        return self._global_step

    def forward(
        self,
        x,
        y=None,
        validation_mode=False,
        confidence_threshold=0.99,
        mask_input: Optional[bool] = None,
        use_pseudo_labels: bool = False,
    ):
        cl = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        ce = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        probabilities = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        kl_layer = torch.tensor([], dtype=torch.float32, device=x.device)

        should_mask_input = (self.training or validation_mode) if mask_input is None else bool(mask_input)
        x_orig = x if should_mask_input else None
        x = self._mask_input(x) if should_mask_input else x

        img_size = x.size()[2:]
        x_pad = self.pad_input(x, self.conv_mult)
        bu_values = self.bottomup_pass(x_pad)
        out, td_data = self.topdown_pass(y, bu_values)

        if self.seg_features == "mu":
            logits, features = self.get_logits(td_data["mu"])
        elif self.seg_features == "bu":
            logits, features = self.get_logits(bu_values)
        else:
            raise KeyError(f"Unknown segmentation features type: {self.seg_features}")

        if self.training_mode == "semisupervised" and (self.training or use_pseudo_labels):
            pseudo_labels = self.get_pseudo_labels(
                td_data["mu"][-1], y, threshold=confidence_threshold
            )
        else:
            pseudo_labels = y

        probabilities = F.softmax(logits, dim=-1)
        out = crop_img_tensor(out, img_size)
        ll, likelihood_info = self.likelihood(out, x_orig if should_mask_input else x)

        inpainting_loss = None
        if should_mask_input:
            recons_sep = -ll
            inpainting_loss = self._centre_crop(recons_sep).mean()

        if self.training or validation_mode:
            if self.use_contrastive_learning:
                cl = compute_cl_loss(
                    mus=td_data["mu"],
                    labels=pseudo_labels if self.training_mode == "semisupervised" else y,
                    margin=self.margin,
                    learnable_thetas=self.learnable_thetas,
                )
            ce = cross_entropy_from_probs(
                probabilities,
                pseudo_labels if self.training_mode == "semisupervised" else y,
                ignore_index=-1,
            )
            kl_layer = compute_kl_loss(
                td_data["posterior"],
                td_data["prior"],
                label=pseudo_labels if self.training_mode == "semisupervised" else y,
                conv_mult=self.conv_mult,
            )

        return {
            "ll": ll,
            "z": td_data["z"],
            "posterior": td_data["posterior"],
            "prior": td_data["prior"],
            "mu": td_data["mu"],
            "kl_per_layer": kl_layer,
            "kl_layer": kl_layer,
            "kl": torch.mean(kl_layer.mean()) if kl_layer.numel() else torch.tensor(0.0, device=x.device),
            "cl": cl,
            "ce": ce,
            "out_mean": likelihood_info["mean"],
            "out_mode": likelihood_info["mode"],
            "out_sample": likelihood_info["sample"],
            "likelihood_params": likelihood_info["params"],
            "inpainting_loss": inpainting_loss,
            "class_probabilities": probabilities,
            "layers_logits": [logits],
            "pseudo_labels": pseudo_labels,
        }

    def bottomup_pass(self, x):
        x = self.first_bottom_up(x)
        bu_values = []
        for layer in self.bottom_up_layers:
            x = layer(x)
            bu_values.append(x)
        return bu_values

    def topdown_pass(
        self,
        label,
        bu_values=None,
        n_img_prior=None,
        mode_layers=None,
        constant_layers=None,
        forced_latent=None,
    ):
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0
        inference_mode = bu_values is not None

        if inference_mode != (n_img_prior is None):
            raise RuntimeError(
                "Number of images for top-down generation has to be given if and only if we're not doing inference"
            )
        if inference_mode and prior_experiment:
            raise RuntimeError(
                "Prior experiments are not compatible with inference mode"
            )

        z = [None] * self.n_layers
        prior = [None] * self.n_layers
        posterior = [None] * self.n_layers
        mu = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        out = None
        for layer_idx in reversed(range(self.n_layers)):
            try:
                bu_value = bu_values[layer_idx]
            except TypeError:
                bu_value = None

            use_mode = layer_idx in mode_layers
            constant_out = layer_idx in constant_layers
            skip_input = out

            out, _, aux = self.top_down_layers[layer_idx](
                label,
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[layer_idx],
            )
            z[layer_idx] = aux["z"]
            prior[layer_idx] = aux["prior"]
            posterior[layer_idx] = aux["posterior"]
            mu[layer_idx] = aux["mu"]

        out = self.final_top_down(out)
        return out, {
            "z": z,
            "prior": prior,
            "posterior": posterior,
            "mu": mu,
        }

    def _mask_input(self, x: torch.Tensor) -> torch.Tensor:
        ann_width = self.cfg.mask_strategy
        x_masked = x.clone()
        patch_size = x.shape[-1]
        mask_size = self.mask_size
        begin = (patch_size - mask_size) // 2
        end = begin + mask_size

        mask_binary = torch.zeros_like(x).bool()
        if self.conv_mult == 2:
            mask_binary[:, :, begin:end, begin:end] = 1
        else:
            mask_binary[:, :, begin:end, begin:end, begin:end] = 1

        mask_value = 0.0
        if ann_width > 0:
            average_mask = torch.zeros_like(x).bool()
            if ann_width <= begin:
                if self.conv_mult == 2:
                    average_mask[
                        :, :, begin - ann_width : end + ann_width, begin - ann_width : end + ann_width
                    ] = 1
                else:
                    average_mask[
                        :,
                        :,
                        begin - ann_width : end + ann_width,
                        begin - ann_width : end + ann_width,
                        begin - ann_width : end + ann_width,
                    ] = 1
                average_mask[mask_binary] = 0
            else:
                average_mask = ~mask_binary
            mask_value = x[average_mask].mean().item()

        x_masked[mask_binary] = mask_value
        return x_masked

    def _centre_crop(self, x: torch.Tensor) -> torch.Tensor:
        patch_size = x.shape[-1]
        mask_size = self.mask_size
        begin = (patch_size - mask_size) // 2
        end = begin + mask_size
        if self.conv_mult == 2:
            return x[:, :, begin:end, begin:end]
        return x[:, :, begin:end, begin:end, begin:end]

    def pad_input(self, x, dim):
        return pad_img_tensor(x, self.get_padded_size(x.size(), dim))

    def get_padded_size(self, size, dim):
        dwnsc = self.overall_downscale_factor
        if len(size) in [2, 3, 4, 5]:
            size = size[-dim:]
        else:
            raise RuntimeError(
                f"input size must be either (N, C, H, W) or (N, C, Z, H, W) or (H, W) or (Z, H, W), but it has length {len(size)} (size={size})"
            )
        return list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

    def sample_prior(self, n_imgs, mode_layers=None, constant_layers=None):
        out, _ = self.topdown_pass(
            n_img_prior=n_imgs,
            mode_layers=mode_layers,
            constant_layers=constant_layers,
        )
        out = crop_img_tensor(out, self.input_array_shape)
        _, likelihood_data = self.likelihood(out, None)
        return likelihood_data["sample"]

    def get_top_prior_param_shape(self, dim, n_imgs=1):
        dwnsc = self.overall_downscale_factor
        size = self.get_padded_size(self.input_array_shape, dim)
        channels = self.z_dims[-1] * 2 * self.n_components
        if self.conv_mult == 2:
            return (n_imgs, channels, size[0] // dwnsc, size[1] // dwnsc)
        if self.conv_mult == 3:
            return (
                n_imgs,
                channels,
                size[0] // dwnsc,
                size[1] // dwnsc,
                size[2] // dwnsc,
            )
        raise AssertionError("Incorrect conv layer dimensions")

    def get_logits(self, features):
        feature_subset = []
        for layer_idx in range(len(features)):
            feature_subset.append(self.feature_selection_layers[layer_idx](features[layer_idx]))
        logits = self.segmentation_head(feature_subset)
        return logits, feature_subset

    def get_pseudo_labels(self, innermost_mu, label, threshold=0.99):
        batch_size = innermost_mu.shape[0]
        group_size = 0
        while label[group_size + 1] == -1:
            group_size += 1
        group_size += 1
        num_groups = batch_size // group_size
        anchors = torch.arange(
            0, num_groups * group_size, group_size, device=label.device
        )

        q_mu_anchors = innermost_mu[anchors]
        labels_anchors = label[anchors]

        if self.conv_mult == 2:
            sums = torch.zeros(
                self.n_components,
                innermost_mu.size(-3),
                innermost_mu.size(-2),
                innermost_mu.size(-1),
                device=label.device,
            )
            counts = torch.zeros(self.n_components, 1, 1, 1, device=label.device)
            sum_dims = (2, 3, 4)
        else:
            sums = torch.zeros(
                self.n_components,
                innermost_mu.size(-4),
                innermost_mu.size(-3),
                innermost_mu.size(-2),
                innermost_mu.size(-1),
                device=label.device,
            )
            counts = torch.zeros(self.n_components, 1, 1, 1, 1, device=label.device)
            sum_dims = (2, 3, 4, 5)

        for class_idx in range(self.n_components):
            mask = labels_anchors == class_idx
            if mask.any():
                sums[class_idx] = q_mu_anchors[mask].sum(dim=0)
                counts[class_idx] = mask.sum()

        means = sums / counts.clamp(min=1)
        diff = innermost_mu.unsqueeze(1) - means.unsqueeze(0)
        dists = (diff * diff).sum(dim=sum_dims)
        logits = -dists / 200
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)
        conf, pseudo = probs.max(dim=1)
        accept = conf > threshold
        pseudo[~accept] = -1
        pseudo[anchors] = label[anchors].long()
        return pseudo
