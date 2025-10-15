import numpy as np
import torch
from torch import nn
from typing import Type, Union
from eps_seg.modules.lvae.likelihoods import GaussianLikelihood  

from eps_seg.modules.lvae.utils import (
    crop_img_tensor,
    pad_img_tensor,
    Interpolate,
    free_bits_kl,
    compute_cl_loss,
)
from eps_seg.modules.lvae.layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    BottomUpDeterministicResBlock,
    BlurPool,
)


class LadderVAE(nn.Module):
    def __init__(
        self,
        z_dims,
        device,
        data_mean,
        data_std,
        color_ch=1,
        noiseModel=None,
        blocks_per_layer=5,
        conv_mult=3,
        nonlin=nn.ELU,
        merge_type="residual",
        batchnorm=True,
        stochastic_skip=True,
        n_filters=64,
        dropout=0.2,
        free_bits=0.0,
        learn_top_prior=False,
        img_shape=None,
        res_block_type="bacdbacd",
        gated=True,
        grad_checkpoint=False,
        no_initial_downscaling=True,
        analytical_kl=True,
        mode_pred=False,
        use_uncond_mode_at=[],
        mask_size=5,
        contrastive_learning=False,
        margin=50,
        lambda_contrastive=0.5,
        stochastic_block_type="normal",
        conditional=False,
        condition_type="mlp",
        n_components=4,
        training_mode="supervised",
        labeled_ratio=0.75,
    ):
        super().__init__()
        self.training_mode = training_mode
        self.color_ch = color_ch
        self.z_dims = z_dims
        self.blocks_per_layer = blocks_per_layer
        # Get class of convolutional layer
        self.conv_mult = conv_mult
        # Standard convolution
        self.conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(
            nn, f"Conv{self.conv_mult}d"
        )
        self.up_conv_type: Type[Union[nn.ConvTranspose2d, nn.ConvTranspose3d]] = getattr(
            nn, f"ConvTranspose{self.conv_mult}d"
        )

        # Get class of nonlinear activation from string description
        self.nonlin: Type[Union[nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU]] = nonlin
        self.n_layers = len(self.z_dims)
        self.stochastic_skip = stochastic_skip
        self.n_filters = n_filters
        self.dropout = dropout
        self.free_bits = free_bits
        self.learn_top_prior = learn_top_prior
        self.input_array_shape = tuple(img_shape)
        self.res_block_type = res_block_type
        self.gated = gated
        self.device = device
        self.data_mean = torch.Tensor([data_mean]).to(self.device)
        self.data_std = torch.Tensor([data_std]).to(self.device)
        self.noiseModel = noiseModel
        self.mode_pred = mode_pred
        self.use_uncond_mode_at = use_uncond_mode_at
        self._global_step = 0
        self.mask_size = mask_size
        self.contrastive_learning = contrastive_learning
        self.margin = margin
        self.lambda_contrastive = lambda_contrastive
        self.prior_type = stochastic_block_type
        self.n_components = n_components

        assert self.data_std is not None, "Data std is not specified"
        assert self.data_mean is not None, "Data mean is not specified"
        assert self.conv_mult in [
            2,
            3,
        ], "Please specify correct conv layers dimension, 2 or 3"
        assert self.color_ch in [
            1,
            2,
            3,
        ], "Please specify correct number of input channels"
        if self.noiseModel is None:
            self.likelihood_form = "gaussian"
        else:
            self.likelihood_form = "noise_model"

        self.downsample = [1] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not no_initial_downscaling:  # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.blocks_per_layer
        assert len(self.downsample) == self.n_layers

        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        stride = 1 if no_initial_downscaling else 2
        self.first_bottom_up = nn.Sequential(
            # self.conv_type(color_ch, n_filters, 5, padding=2, stride=stride),
            self.conv_type(color_ch, n_filters, 5, padding=2, stride=1),  # No stride here
            BlurPool(n_filters, stride=stride),  # Add BlurPool for downsampling
            self.nonlin(),
            BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                conv_mult=self.conv_mult,
                nonlin=self.nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                grad_checkpoint=grad_checkpoint,
            ),
        )

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=self.downsample[i],
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                    grad_checkpoint=grad_checkpoint,
                )
            )

            # Add top-down stochastic layer at level i.

            self.top_down_layers.append(
                TopDownLayer(
                    z_dim=z_dims[i],
                    n_layers=self.n_layers,
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    merge_type=merge_type,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    stochastic_skip=stochastic_skip,
                    learn_top_prior=learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(
                        dim=self.conv_mult
                    ),
                    res_block_type=res_block_type,
                    gated=gated,
                    grad_checkpoint=grad_checkpoint,
                    analytical_kl=analytical_kl,
                    stochastic_block_type=stochastic_block_type,
                    conditional=conditional,
                    condition_type=condition_type,
                    n_components=n_components,
                    training_mode=training_mode,
                    labeled_ratio=labeled_ratio,
                )
            )

        # Final top-down layer
        modules = list()
        if not no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for i in range(blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                    grad_checkpoint=grad_checkpoint,
                )
            )
        self.final_top_down = nn.Sequential(*modules)

        # Define likelihood
        if self.likelihood_form == "gaussian":
            self.likelihood = GaussianLikelihood(n_filters, color_ch, conv_mult)
        # In eps-Seg we use only use a Gaussian likelihood
        # elif self.likelihood_form == "noise_model":
        #     self.likelihood = NoiseModelLikelihood(
        #         n_filters,
        #         color_ch,
        #         conv_mult,
        #         data_mean,
        #         data_std,
        #         noiseModel,
        #     )
        else:
            msg = "Unrecognized likelihood '{}'".format(self.likelihood_form)
            raise RuntimeError(msg)

    def increment_global_step(self):
        """Increments global step by 1."""
        self._global_step += 1

    def update_mode(self, mode):
        """Update training mode and propagate to all submodules."""
        print(f"Updating model mode from {self.training_mode} to {mode}")
        self.training_mode = mode
        for layer in self.top_down_layers:
            layer.update_mode(mode)

    @property
    def global_step(self) -> int:
        """Global step."""
        return self._global_step

    # TODO: check forward function
    def forward(self, x, y=None, x_orig=None, confidence_threshold=0.99):
        img_size = x.size()[2:]
        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x, self.conv_mult)
        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        # Top-down inference/generation
        out, td_data = self.topdown_pass(
            y, bu_values, confidence_threshold=confidence_threshold
        )
        # Restore original image size
        out = crop_img_tensor(out, img_size)
        # Log likelihood and other info (per data point)

        cl = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        kl = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if x_orig is not None:
            ll, likelihood_info = self.likelihood(out, x_orig)
        else:
            ll, likelihood_info = self.likelihood(out, x)
        if self.mode_pred is False:
            # kl[i] for each i has length batch_size
            # resulting kl shape: (batch_size, layers)
            kl = torch.stack(td_data["kl"]).sum(0)
            if self.free_bits > 0:
                kl = free_bits_kl(kl, self.free_bits)
        q = None
        if self.contrastive_learning and self.mode_pred is False:
            cl, q = compute_cl_loss(
                mus=td_data["mu"],
                logvars=td_data["logvar"],
                pis=td_data["pi"] if "pi" in td_data else None,
                labels=td_data["pseudo_labels"],
                # labels=y,
                margin=self.margin,
                lambda_contrastive=self.lambda_contrastive,
                training_mode=self.training_mode,
                prior=self.prior_type,
            )
        output = {
            "ll": ll,
            "z": td_data["z"],
            "mu": td_data["mu"],
            "kl": kl,
            "cl": cl,
            "logp": td_data["logprob_p"],
            "out_mean": likelihood_info["mean"],
            "out_mode": likelihood_info["mode"],
            "out_sample": likelihood_info["sample"],
            "likelihood_params": likelihood_info["params"],
            "ce": td_data["ce"][-1],
            "entropy": td_data["entropy"][-1],
            "pi": td_data["pi"][-1] if "pi" in td_data else None,
            "q": q,
        }
        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            x = self.bottom_up_layers[i](x)
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
        confidence_threshold=0.99,
    ):
        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = (
                "Number of images for top-down generation has to be given "
                "if and only if we're not doing inference"
            )
            raise RuntimeError(msg)
        if inference_mode and prior_experiment:
            msg = (
                "Prior experiments (e.g. sampling from mode) are not"
                " compatible with inference mode"
            )
            raise RuntimeError(msg)

        # Sampled latent variables at each layer
        z = [None] * self.n_layers

        # KL divergence of each layer
        kl = [0.0] * self.n_layers
        ce = [0.0] * self.n_layers
        entropy = [0.0] * self.n_layers

        mu = [None] * self.n_layers
        logvar = [None] * self.n_layers
        pi = [None] * self.n_layers
        pseudo_labels = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass
        logprob_p = 0.0

        # Top-down inference/generation loop
        out = None
        for i in reversed(range(self.n_layers)):
            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers
            use_uncond_mode = i in self.use_uncond_mode_at

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, _, aux = self.top_down_layers[i](
                label,
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred,
                use_uncond_mode=use_uncond_mode,
                considence_threshold=confidence_threshold,
            )
            z[i] = aux["z"]  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux["kl"]  # (batch, )
            ce[i] = aux["cross_entropy"]
            entropy[i] = aux["entropy"]
            mu[i] = aux["mu"]
            pi[i] = aux["pi"] if "pi" in aux else None
            pseudo_labels[i] = aux["pseudo_labels"] if "pseudo_labels" in aux else None
            if self.mode_pred is False:
                logprob_p += aux["logprob_p"].mean()  # mean over batch
            else:
                logprob_p = None

        # Final top-down layer
        out = self.final_top_down(out)
        data = {
            "z": z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            "kl": kl,  # list of tensors with shape (batch, )
            "logprob_p": logprob_p,  # scalar, mean over batch
            "mu": mu,
            "logvar": logvar,
            "pi": pi,
            "ce": ce,
            "entropy": entropy,
            "pseudo_labels": pseudo_labels,
        }
        return out, data

    def _mask_input(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out the centre mask according to self.mask_size."""
        x_masked = x.clone()
        ps = x.shape[-1]
        ms = self.mask_size
        b = (ps - ms) // 2
        e = b + ms
        if self.conv_mult == 2:
            x_masked[:, :, b:e, b:e] = 0
        else:
            x_masked[:, :, b:e, b:e, b:e] = 0
        return x_masked

    def _centre_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Crop the centre window used for inpainting loss."""
        ps = x.shape[-1]
        ms = self.mask_size
        b = (ps - ms) // 2
        e = b + ms
        if self.conv_mult == 2:
            return x[:, :, b:e, b:e]
        else:
            return x[:, :, b:e, b:e, b:e]

    def _step_common(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        threshold: float = 0.99,
        amp: bool = True,
    ) -> dict:
        """
        Shared compute for train/val: runs forward with masked input and returns all
        component losses + useful tensors.
        """
        from torch.cuda.amp import autocast

        # 1) mask input
        x_masked = self._mask_input(x).to(x.device)

        # 2) forward
        with autocast(enabled=amp):
            out = self(x=x_masked, y=y, x_orig=x, threshold=threshold)

        # 3) inpainting loss is centre of -loglikelihood
        recons_sep = -out["ll"]
        inpainting_loss = self._centre_crop(recons_sep).mean()

        # 4) other components (guard cl for NaN)
        kl = out["kl"]
        cl = out.get("cl", torch.tensor(0.0, dtype=torch.float32, device=x.device))
        if torch.isnan(cl):
            cl = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        ce = out.get("ce", torch.tensor(0.0, dtype=torch.float32, device=x.device))
        entropy = out.get(
            "entropy", torch.tensor(0.0, dtype=torch.float32, device=x.device)
        )

        return {
            "inpainting_loss": inpainting_loss,
            "kl_loss": kl,
            "cl_loss": cl,
            "ce": ce,
            "entropy": entropy,
            "q": out.get("q"),
            "out_mean": out.get("out_mean"),
            "out_sample": out.get("out_sample"),
        }

    def training_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        threshold: float = 0.99,
        amp: bool = True,
    ) -> dict:
        """Return per-batch component losses/metrics for training."""
        return self._step_common(x, y, threshold=threshold, amp=amp)

    @torch.no_grad()
    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, *, amp: bool = True
    ) -> dict:
        """Return per-batch component losses/metrics for validation."""
        # If threshold annealing is only for semi-supervised, you can just use 0.99 here
        return self._step_common(x, y, threshold=0.99, amp=amp)

    def pad_input(self, x, dim):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size(), dim)  # TODO check !
        x = pad_img_tensor(x, size)

        return x

    def get_padded_size(self, size, dim):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, W) or (N, C, Z, H, W) or (H, W) or (Z, H, W)
        :return: 2-tuple (H, W)
        """
        # TODO check!!!
        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor
        # Make size argument into (heigth, width) or (depth, heigth, width)
        if len(size) in [2, 3, 4, 5]:
            size = size[-dim:]
        else:
            msg = (
                f"input size must be either (N, C, H, W) or (N, C, Z, H, W) or (H, W) or (Z, H, W), but it "
                f"has length {len(size)} (size={size})"
            )
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)
        return padded_size

    def sample_prior(self, n_imgs, mode_layers=None, constant_layers=None):
        # Generate from prior
        out, _ = self.topdown_pass(
            n_img_prior=n_imgs, mode_layers=mode_layers, constant_layers=constant_layers
        )
        out = crop_img_tensor(out, self.input_array_shape)

        # Log likelihood and other info (per data point)
        _, likelihood_data = self.likelihood(out, None)

        return likelihood_data["sample"]

    def get_top_prior_param_shape(self, dim, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.input_array_shape, dim)
        c = self.z_dims[-1] * 2  # mu and logvar
        if self.prior_type == "mixture":
            c *= self.n_components
        if self.conv_mult == 2:
            h = sz[0] // dwnsc
            w = sz[1] // dwnsc
            top_layer_shape = (n_imgs, c, h, w)
        elif self.conv_mult == 3:
            z = sz[0] // dwnsc
            h = sz[1] // dwnsc
            w = sz[2] // dwnsc
            assert len(self.input_array_shape) >= 3, (
                "Depth dimension not specified for input array"
            )
            top_layer_shape = (n_imgs, c, z, h, w)
        else:
            raise AssertionError("Incorrect conv layer dimensions")
        return top_layer_shape
