import numpy as np
import torch
from torch import nn
from typing import Type, Union
from eps_seg.modules.lvae.likelihoods import GaussianLikelihood
from torchinfo import summary
import torch.nn.functional as F


from eps_seg.modules.lvae.utils import (
    crop_img_tensor,
    pad_img_tensor,
    Interpolate,
    free_bits_kl,
    compute_cl_loss,
    compute_ce_loss,
    compute_kl_loss,
)
from eps_seg.modules.lvae.layers import (
    TopDownLayer,
    BottomUpLayer,
    TopDownDeterministicResBlock,
    BottomUpDeterministicResBlock,
    BlurPool,
    FeatureSubsetSelectionLayer,
    SegmentationHead,
)
from eps_seg.config import LVAEConfig


class LadderVAE(nn.Module):
    def __init__(self, cfg: LVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.training_mode = cfg.training_mode

        self.n_layers = cfg.n_layers
        self.z_dims = cfg.z_dims
        self.blocks_per_layer = cfg.blocks_per_layer
        self.conv_mult = cfg.conv_mult

        # Standard convolution
        self.conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(
            nn, f"Conv{self.conv_mult}d"
        )
        self.up_conv_type: Type[Union[nn.ConvTranspose2d, nn.ConvTranspose3d]] = getattr(
            nn, f"ConvTranspose{self.conv_mult}d"
        )
        self.nonlin: Type[Union[nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU]] = getattr(
            nn, cfg.nonlin
        )

        self.enable_top_down_residuals = cfg.enable_top_down_residuals
        self.skip_connections = cfg.skip_connections
        self.skip_connections_merge_type = cfg.skip_connections_merge_type
        self.batchnorm = cfg.use_batchnorm
        self.color_ch = cfg.color_channels
        self.n_filters = cfg.n_fiters
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

        # Derived paramters
        self.input_array_shape = cfg.img_shape
        self.likelihood_form = "gaussian"

        # assert self.data_std is not None, "Data std is not specified"
        # assert self.data_mean is not None, "Data mean is not specified"
        assert self.conv_mult in [
            2,
            3,
        ], "Please specify correct conv layers dimension, 2 or 3"
        assert self.color_ch in [
            1,
            2,
            3,
        ], "Please specify correct number of input channels"

        self.build_architecture()

    def build_architecture(self):
        self.downsample = [1] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not self.no_initial_downscaling:  # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.blocks_per_layer
        assert len(self.downsample) == self.n_layers

        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        stride = 1 if self.no_initial_downscaling else 2
        self.first_bottom_up = nn.Sequential(
            # self.conv_type(color_ch, n_filters, 5, padding=2, stride=stride),
            self.conv_type(
                self.color_ch, self.n_filters, 5, padding=2, stride=1
            ),  # No stride here
            BlurPool(
                self.n_filters, stride=stride, dim=self.conv_mult
            ),  # Add BlurPool for downsampling
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

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])
        self.feature_selection_layers = nn.ModuleList([])

        # Z dimensions for stochastic layers are downscaled by factor 2
        self.head_z_dims = int(
            self.input_array_shape[-1] / (2**self.n_layers)
        )  # if isotropic, better to get size of x

        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            new_layer = BottomUpLayer(
                layer_number=i,
                n_res_blocks=self.blocks_per_layer,
                n_filters=self.n_filters,
                downsampling_steps=self.downsample[i],
                conv_mult=self.conv_mult,
                nonlin=self.nonlin,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                res_block_type=self.res_block_type,
                gated=self.use_gated_convs,
                grad_checkpoint=self.use_grad_checkpoint,
            )
            self.bottom_up_layers.append(new_layer)

            new_layer = FeatureSubsetSelectionLayer(
                layer_number=i,
                crop_size=self.feature_spatial_size[i],
                enabled=True,
            )
            self.feature_selection_layers.append(new_layer)

            # Add top-down stochastic layer at level i.
            # FIXME: Review commented out parameters and reimplement
            self.top_down_layers.append(
                TopDownLayer(
                    layer_number=i,
                    z_dim=self.z_dims[i],
                    seg_head_dim=self.head_z_dims,
                    # n_layers=self.n_layers,
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=self.n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    conv_mult=self.conv_mult,
                    nonlin=self.nonlin,
                    skip_connection_merge_type=self.skip_connections_merge_type,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    enable_top_down_residual=self.enable_top_down_residuals[i],
                    skip_connection=self.skip_connections[i],
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(
                        dim=self.conv_mult
                    ),
                    res_block_type=self.res_block_type,
                    gated=self.use_gated_convs,
                    grad_checkpoint=self.use_grad_checkpoint,
                    n_components=self.n_components,
                    training_mode=self.training_mode,
                )
            )

        # Final top-down layer
        modules = list()
        if not self.no_initial_downscaling:
            modules.append(Interpolate(scale=2))
        for i in range(self.blocks_per_layer):
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
        self.segmentation_head = SegmentationHead(
            in_channels=self.n_filters,
            n_classes=self.n_components,
            conv_mult=self.conv_mult,
            hidden_channels=int(self.n_filters / 2),
            n_layers=self.n_layers,
            kernel=1,
        )
        # Define likelihood
        self.likelihood = GaussianLikelihood(
            self.n_filters, self.color_ch, self.conv_mult
        )

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

    def forward(self, x, y=None, validation_mode=False, confidence_threshold=0.99):
        """
        Forward pass through the LVAE model.

        Args:
            x: Unmasked Image - Input tensor of shape (batch_size, channels, height, width)
            y: Optional labels tensor
            validation_mode: Whether we are in validation mode (used to mask input or not and compute losses)
            confidence_threshold: Confidence threshold for assigning pseudo-labels
        """
        # TODO: Masking can also be handled outside the model (in LightningModule), but it would need to also move loss computation there
        # TODO: Find a way to also check it during validation (but not during prediction) to match original behaviour
        mask_input = self.training or validation_mode
        x_orig = x if mask_input else None
        x = self._mask_input(x) if mask_input else x

        img_size = x.size()[2:]
        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x, self.conv_mult)
        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)

        # Top-down inference/generation
        out, td_data = self.topdown_pass(y, bu_values)

        # get logits from segmentation head
        if self.seg_features == "mu":
            logits, features = self.get_logits(td_data["mu"])
        elif self.seg_features == "bu":
            logits, features = self.get_logits(bu_values)
        else:
            KeyError(f"Unknown segmentation features type: {self.seg_features}")

        if self.training_mode == "semisupervised" and self.training:
            # get pseudo-labels
            pseudo_labels = self.get_pseudo_labels(
                td_data["mu"][-1], y, threshold=confidence_threshold
            )
        else:
            pseudo_labels = y

        # Restore original image size
        out = crop_img_tensor(out, img_size)
        # Log likelihood and other info (per data point)

        cl = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        ce = torch.tensor(0.0, dtype=torch.float32, device=x.device)

        # If original (unmasked) input is given, use it for likelihood computation, otherwise use masked input
        ll, likelihood_info = self.likelihood(out, x_orig if mask_input else x)

        inpainting_loss = None
        if mask_input:
            # 3) inpainting loss is centre of -loglikelihood
            # FIXME: This "out" is not a dictionary.
            recons_sep = -ll
            inpainting_loss = self._centre_crop(recons_sep).mean()

        if self.training or validation_mode:  # TODO: Merge with above condition?
            # kl[i] for each i has length batch_size
            # resulting kl shape: (batch_size, layers)
            # kl = torch.stack(td_data["kl"]).sum(0)
            # if self.kl_free_bits > 0:
            #     kl = free_bits_kl(kl, self.kl_free_bits)

            if self.use_contrastive_learning:
                cl = compute_cl_loss(
                    mus=td_data["mu"],
                    labels=pseudo_labels if self.training_mode == "semisupervised" else y,
                    margin=self.margin,
                    learnable_thetas=self.learnable_thetas,
                )
            ce = compute_ce_loss(
                logits, pseudo_labels if self.training_mode == "semisupervised" else y
            )

            probabilities = F.softmax(logits, dim=-1)

            kl_layer = compute_kl_loss(
                td_data["posterior"],
                td_data["prior"],
                probabilities,
                label=pseudo_labels if self.training_mode == "semisupervised" else y,
                conv_mult=self.conv_mult,
            )

        output = {
            "ll": ll,
            "z": td_data["z"],
            "posterior": td_data["posterior"],
            "prior": td_data["prior"],
            "mu": td_data["mu"],
            "kl_layer": kl_layer,
            "kl": kl_layer.mean(),
            "cl": cl,
            "ce": ce,
            "out_mean": likelihood_info["mean"],
            "out_mode": likelihood_info["mode"],
            "out_sample": likelihood_info["sample"],
            "likelihood_params": likelihood_info["params"],
            "inpainting_loss": inpainting_loss,
            "class_probabilities": probabilities,
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

        prior = [None] * self.n_layers
        posterior = [None] * self.n_layers
        mu = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # log p(z) where z is the sample in the topdown pass

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
            )
            z[i] = aux["z"]  # sampled variable at this layer (batch, ch, h, w)
            # kl[i] = aux["kl"]  # (batch, )
            # ce[i] = aux.get("cross_entropy", None)
            prior[i] = aux["prior"]
            posterior[i] = aux["posterior"]
            mu[i] = aux["mu"]
            # class_prob[i] = aux["class_probabilities"] if "class_probabilities" in aux else None

            # if self.training:
            #     logprob_p += aux["logprob_p"].mean()  # mean over batch
            # else:
            #     logprob_p = None

        # Final top-down layer
        out = self.final_top_down(out)
        data = {
            "z": z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            # "kl": kl,  # list of tensors with shape (batch, )
            # "logprob_p": logprob_p,  # scalar, mean over batch
            "prior": prior,
            "posterior": posterior,
            "mu": mu,
            # "logvar": logvar,
            # "class_probabilities": class_prob,
            # "cross_entropy": ce,
        }
        return out, data

    def _mask_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies masking to the input tensor x according to the specified masking strategy.
        """
        ann_width = self.cfg.mask_strategy  # Depth of the annulus for average masking

        x_masked = x.clone()
        ps = x.shape[-1]
        ms = self.mask_size
        b = (ps - ms) // 2
        e = b + ms

        mask_binary = torch.zeros_like(x).bool()
        if self.conv_mult == 2:
            mask_binary[:, :, b:e, b:e] = 1
        else:
            mask_binary[:, :, b:e, b:e, b:e] = 1

        mask_value = 0.0
        average_mask = None
        if ann_width > 0:  # If ann_width == 0, fill mask with zeros
            average_mask = torch.zeros_like(x).bool()
            if ann_width <= b:
                # Take an annulus around the masked region
                if self.conv_mult == 2:
                    average_mask[
                        :, :, b - ann_width : e + ann_width, b - ann_width : e + ann_width
                    ] = 1
                    average_mask[mask_binary] = 0  # Exclude the masked region itself
                else:
                    average_mask[
                        :,
                        :,
                        b - ann_width : e + ann_width,
                        b - ann_width : e + ann_width,
                        b - ann_width : e + ann_width,
                    ] = 1
                    average_mask[mask_binary] = 0  # Exclude the masked region itself
            else:
                # If the annulus would go out of bounds, use the entire unmasked area
                average_mask = ~mask_binary
            mask_value = x[average_mask].mean().item()

        x_masked[mask_binary] = mask_value

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
        # For mixture model, we need parameters for each component
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

    def get_logits(self, features):
        """
        Get segmentation logits from the model given bottom-up values.

        Args:
            features: List of bottom-up feature (bu_values or mu) tensors from each layer.
        """
        feature_subset = []
        for i in range(len(features)):
            feature_subset.append(self.feature_selection_layers[i](features[i]))
        logits = self.segmentation_head(feature_subset)
        return logits, feature_subset

    def get_pseudo_labels(self, mu, label, threshold=0.99):
        batch_size = mu[-1].shape[0]
        group_size = 0
        while(label[group_size + 1] == -1):
            group_size += 1
        group_size += 1 # one anchor and it's neighbors
        num_groups = batch_size // group_size
        anchors = torch.arange(
            0, num_groups * group_size, group_size, device=label.device
        )

        q_mu_anchors = mu[anchors]
        labels_anchors = label[anchors]

        # Compute class means from labeled anchor samples
        if self.conv_mult == 2:
            sums = torch.zeros(
                self.n_components,
                mu.size(-3),
                mu.size(-2),
                mu.size(-1),
                device=label.device,
            )
        else:
            sums = torch.zeros(
                self.n_components,
                mu.size(-4),
                mu.size(-3),
                mu.size(-2),
                mu.size(-1),
                device=label.device,
            )
        counts = (
            torch.zeros(self.n_components, 1, 1, 1, device=label.device)
            if self.conv_mult == 2
            else torch.zeros(self.n_components, 1, 1, 1, 1, device=label.device)
        )

        for c in range(self.n_components):
            mask = labels_anchors == c
            if mask.any():
                sums[c] = q_mu_anchors[mask].sum(dim=0)
                counts[c] = mask.sum()

        means = sums / counts.clamp(min=1)

        # Compute distances and logits for pseudo-labeling
        diff = mu.unsqueeze(1) - means.unsqueeze(0)
        dists = (diff * diff).sum(dim=(2, 3, 4) if self.conv_mult == 2 else (2, 3, 4, 5))
        logits = -dists / 200
        logits = logits - logits.max(dim=1, keepdim=True).values

        y = F.softmax(logits)

        # Generate pseudo labels with confidence thresholding
        conf, pseudo = y.max(dim=1)
        accept = conf > threshold
        pseudo[~accept] = -1
        pseudo[anchors] = label[anchors].long()

        return pseudo  # TODO: fix me: logit here is not used for ceoss entropy
