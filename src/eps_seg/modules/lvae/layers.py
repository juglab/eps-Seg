import torch
from torch import nn
from typing import Type, Union
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def no_cp(func, inp):
    return func(inp)

class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.

    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(
        self,
        channels,
        conv_mult,
        nonlin,
        kernel=None,
        groups=1,
        batchnorm=True,
        block_type=None,
        dropout=None,
        gated=None,
        grad_checkpoint=False,
    ):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        pad = [k // 2 for k in kernel]
        dropout = dropout if not grad_checkpoint else None
        self.cp = checkpoint if grad_checkpoint else no_cp
        # TODO Might need to update batchnorm stats calculation for grad checkpointing

        conv_layer: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")
        batchnorm_layer_type: Type[Union[nn.BatchNorm2d, nn.BatchNorm3d]] = getattr(
            nn, f"BatchNorm{conv_mult}d"
        )
        dropout_layer_type: Type[Union[nn.Dropout2d, nn.Dropout3d]] = getattr(
            nn, f"Dropout{conv_mult}d"
        )
        modules = []

        if block_type == "cabdcabd":
            for i in range(2):
                conv = conv_layer(
                    channels, channels, kernel[i], padding=pad[i], groups=groups
                )
                modules.append(conv)
                modules.append(nonlin())
                if batchnorm:
                    modules.append(batchnorm_layer_type(channels))
                if dropout is not None:
                    modules.append(dropout_layer_type(dropout))

        elif block_type == "bacdbac":
            for i in range(2):
                if batchnorm:
                    modules.append(batchnorm_layer_type(channels))
                modules.append(nonlin())
                conv = conv_layer(
                    channels, channels, kernel[i], padding=pad[i], groups=groups
                )
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(dropout_layer_type(dropout))

        elif block_type == "bacdbacd":
            for i in range(2):
                if batchnorm:
                    modules.append(batchnorm_layer_type(channels))
                modules.append(nonlin())
                conv = conv_layer(
                    channels, channels, kernel[i], padding=pad[i], groups=groups
                )
                modules.append(conv)
                if dropout is not None:
                    modules.append(dropout_layer_type(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        if gated:
            modules.append(GateLayer(channels, 1, conv_layer, nonlin))
        self.block = nn.Sequential(*modules)

    def forward(self, inp):
        # return self.cp(self.block, inp) + inp
        if torch.onnx.is_in_onnx_export():
            return self.block(inp) + inp  # No checkpointing during ONNX export
        else:
            return self.cp(self.block, inp) + inp  # Use checkpointing during training

class ResidualGatedBlock(ResidualBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)

class GateLayer(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size, conv_type, nonlin=nn.LeakyReLU):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = conv_type(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate


class TopDownLayer(nn.Module):
    """
    Top-down layer, including stochastic sampling, KL computation, and small
    deterministic ResNet with upsampling.

    The architecture when doing inference is roughly as follows:
    p_params = output of top-down layer above
    bu = inferred bottom-up value at this layer
    q_params = merge(bu, p_params)
    z = stochastic_layer(q_params)
    possibly get skip connection from previous top-down layer
    top-down deterministic ResNet

    When doing generation only, the value bu is not available, the
    merge layer is not used, and z is sampled directly from p_params.

    If this is the top layer, at inference time, the uppermost bottom-up value
    is used directly as q_params, and p_params are defined in this layer
    (while they are usually taken from the previous layer), and can be learned.

    Args:
        - stochastic_skip (bool): whether to use skip connection around stochastic block
        - skip_connection (bool): whether to use skip connection at this layer (from encoder to decoder)
        - seg_head_dim (int): 
            dimension of the segmentation head - Only used in the top layers

    """

    def __init__(
        self,
        z_dim,
        seg_head_dim,
        n_res_blocks,
        n_filters,
        is_top_layer=False,
        downsampling_steps=None,
        conv_mult=2,
        nonlin=None,
        skip_connection_merge_type=None,
        batchnorm=True,
        dropout=None,
        enable_top_down_residual=False,
        skip_connection=True,
        res_block_type=None,
        gated=None,
        grad_checkpoint=False,
        learn_top_prior=False,
        top_prior_param_shape=None,
        n_components=4,  # Used only for Mixture block
        training_mode="supervised",
        stochastic_block_type="mixture",
    ):
        super().__init__()
        self.training_mode = training_mode
        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.seg_head_dim = seg_head_dim
        self.enable_top_down_residual = enable_top_down_residual
        self.learn_top_prior = learn_top_prior
        self.n_components = n_components
        self.top_prior_param_shape = top_prior_param_shape
        self.stochastic_block_type = stochastic_block_type
        self.skip_connection = skip_connection

        if self.is_top_layer:
            self.top_prior_params = self._get_top_prior_params()

        # Downsampling steps left to do in this layer
        dws_left = downsampling_steps

        # Define deterministic top-down block: sequence of deterministic
        # residual blocks with downsampling when needed.
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

        # Define stochastic block with convolutions
        # Select stochastic block based on the argument

        if is_top_layer:
            self.stochastic = MixtureStochasticConvBlock(
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                conv_mult=conv_mult,
                n_components=self.n_components,
                training_mode=training_mode,
                seg_head_dim=self.seg_head_dim,
            )
        else:
            self.stochastic = NormalStochasticConvBlock(
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                conv_mult=conv_mult,
                training_mode=training_mode,
            )

        if not is_top_layer:
            # Merge layer, combine bottom-up inference with top-down
            # generative to give posterior parameters
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

            # Skip connection that goes around the stochastic top-down layer
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
        print(f"Updating TopDownLayer mode from {self.training_mode} to {mode}")
        self.training_mode = mode
        self.stochastic.update_mode(mode)

    def _get_top_prior_params(self) -> Union[None, nn.Parameter]:
        # Define top layer prior parameters, possibly learnable
        if self.is_top_layer and self.stochastic_block_type == "mixture":
            return self._initialize_gmm_prior(
                self.n_components, self.top_prior_param_shape, self.learn_top_prior
            )
        else:
            raise ValueError(
                "Top prior can only be initialized for mixture stochastic block type."
            )

    def _initialize_gmm_prior(self, n_components, top_prior_param_shape, learn_top_prior):
        # TODO write tests for this function
        # Extract spatial dimensions and channels
        total_channels = top_prior_param_shape[1]  # Total number of channels
        spatial_res = top_prior_param_shape[2]  # Spatial resolution

        # Each GMM component uses an equal fraction of the channels
        channels_per_component = total_channels // (
            2 * n_components
        )  # Half for mus, half for sigmas

        # Initialize the tensor for means (mus)
        chunk_values = torch.zeros(
            (n_components, channels_per_component, spatial_res, spatial_res)
        )
        # TODO hardcoded 2.0, better initialization strategy?
        # Dynamically assign values to means
        chunk_size = channels_per_component // n_components
        for i in range(n_components):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunk_values[i, start_idx:end_idx] = (
                2.0  # Equidistant initialization for means
            )

        # Reshape means into the required format
        mus = chunk_values.view(
            1, n_components * channels_per_component, spatial_res, spatial_res
        )

        # Initialize standard deviations (sigmas) as zeros (or another value if needed)
        sigmas = torch.zeros_like(mus)

        # Concatenate mus and sigmas along the channel dimension
        prior_params = torch.cat([mus, sigmas], dim=1)

        # Convert to nn.Parameter
        # Convert prior_params to nn.Parameter
        prior_params = nn.Parameter(prior_params, requires_grad=learn_top_prior)

        return prior_params

    def forward(
        self,
        label,
        input_=None,
        skip_connection_input=None,
        inference_mode=False,
        bu_value=None,
        n_img_prior=None,
        confidence_threshold=0.5,
        use_mode=False,
        force_constant_output=False,
        forced_latent=None,
        mode_pred=None,
    ):
        """
        Forward pass through top-down layer.
        Args:
            label: segmentation label (if available)
            input_: input from layer above (prior parameters)
            skip_connection_input: input from previous top-down layer
            inference_mode: whether to run in inference mode (True) or
                generative mode (False)
            bu_value: bottom-up value at this layer (inference mode only)
            n_img_prior: number of images to sample from prior (generative mode only)
            confidence_threshold: threshold for pseudo-labeling in unsupervised mode
            use_mode: whether to use mode of distribution instead of sampling

        """
        if use_mode:
            print("TODO: use_mode is not implemented yet")
        if force_constant_output:
            print("TODO: force_constant_output is not implemented yet")
        if forced_latent is not None:
            print("TODO: forced_latent is not implemented yet")
        if mode_pred is not None:
            print("TODO: mode_pred is not implemented yet")
        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")
        # If top layer, define parameters of prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params
            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(
                    n_img_prior, *[-1] * len(p_params.shape[1:])
                )  # TODO check dims!

        # Else the input from the layer above is the prior parameters
        else:
            p_params = input_
        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if inference_mode:
            if self.is_top_layer:
                q_params = bu_value
            else:
                if self.skip_connection:
                    q_params = self.skip_connection_merger(bu_value, p_params)
                else:
                    q_params = p_params
                    

        # In generative mode, q is not used
        else:
            q_params = None

        # Sample from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        if self.is_top_layer:
            x, data_stoch = self.stochastic(
                p_params=p_params, q_params=q_params,
                label=label, confidence_threshold=confidence_threshold
            )
        else:
            x, data_stoch = self.stochastic(
                p_params=p_params, q_params=q_params
            )


        # Skip connection from previous layer
        if self.enable_top_down_residual and not self.is_top_layer:
            x = self.top_down_residual(x, skip_connection_input)

        # Save activation before residual block: could be the skip
        # connection input in the next layer
        x_pre_residual = x

        # Last top-down block (sequence of residual blocks)
        x = self.deterministic_block(x)
        data = {k: v for k, v in data_stoch.items()}
        return x, x_pre_residual, data


class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference, roughly the same as the
    small deterministic Resnet in top-down layers. Consists of a sequence of
    bottom-up deterministic residual blocks with downsampling.
    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        downsampling_steps=0,
        conv_mult=2,
        nonlin=None,
        batchnorm=True,
        dropout=None,
        res_block_type=None,
        gated=None,
        grad_checkpoint=False,
        device=None,
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
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(*bu_blocks).to(self.device)

    def forward(self, x):
        return self.net(x)


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).

    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through strided convolution.

    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.

    Other parameters: kernel size, nonlinearity, and groups of the internal
    residual block; whether batch normalization and dropout are performed;
    whether the residual path has a gate layer at the end. There are a few
    residual block structures to choose from.
    """

    def __init__(
        self,
        mode,
        c_in,
        c_out,
        conv_mult=2,
        nonlin=nn.LeakyReLU,
        resample=False,
        res_block_kernel=None,
        groups=1,
        batchnorm=True,
        res_block_type=None,
        dropout=None,
        min_inner_channels=None,
        gated=None,
        grad_checkpoint=False,
    ):
        super().__init__()
        assert mode in ["top-down", "bottom-up"]
        if min_inner_channels is None:
            min_inner_channels = 0
        inner_filters = max(c_out, min_inner_channels)

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")
        upsample_conv: Type[Union[nn.ConvTranspose2d, nn.ConvTranspose3d]] = getattr(
            nn, f"ConvTranspose{conv_mult}d"
        )

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == "bottom-up":  # downsample
                self.pre_conv = conv_type(
                    in_channels=c_in,
                    out_channels=inner_filters,
                    kernel_size=3,
                    padding=1,
                    # stride=2,
                    # groups=groups,
                )
                # self.pre_conv = conv_type(c_in, c_out, kernel_size=3, padding=1)
                self.blurpool = BlurPool(inner_filters, stride=2)
            elif mode == "top-down":  # upsample
                self.pre_conv = upsample_conv(
                    in_channels=c_in,
                    out_channels=inner_filters,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    groups=groups,
                    output_padding=1,
                )
        elif c_in != inner_filters:
            self.pre_conv = conv_type(c_in, inner_filters, 1, groups=groups)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            conv_mult=conv_mult,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
            grad_checkpoint=grad_checkpoint,
        )

        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = conv_type(inner_filters, c_out, 1, groups=groups)
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        if hasattr(self, "blurpool"):
            x = self.blurpool(x)  # Apply BlurPool
        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):
    def __init__(self, *args, upsample=False, **kwargs):
        kwargs["resample"] = upsample
        super().__init__("top-down", *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):
    def __init__(self, *args, downsample=False, **kwargs):
        kwargs["resample"] = downsample
        super().__init__("bottom-up", *args, **kwargs)


class MergeLayer(nn.Module):
    """
    Merge two 4D/5D input tensors by concatenating along dim=1 and passing the
    result through 1) a convolutional 1x1 layer, or 2) a residual block
    """

    def __init__(
        self,
        channels,
        merge_type,
        conv_mult=2,
        nonlin=nn.LeakyReLU,
        batchnorm=True,
        dropout=None,
        res_block_type=None,
        grad_checkpoint=False,
    ):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3
        assert len(channels) == 3

        # Standard convolution case
        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")

        # Handle the "merge_type" logic
        if merge_type == "linear":
            if conv_mult == 0:
                self.layer = conv_type(
                    channels[0] + channels[1], channels[2], kernel_size=1, padding=0
                )
            else:
                self.layer = conv_type(channels[0] + channels[1], channels[2], 1)
        elif merge_type == "residual":
            if conv_mult == 0:
                self.layer = nn.Sequential(
                    conv_type(
                        channels[0] + channels[1], channels[2], kernel_size=1, padding=0
                    ),
                    ResidualGatedBlock(
                        channels[2],
                        conv_mult,
                        nonlin,
                        batchnorm=batchnorm,
                        dropout=dropout,
                        block_type=res_block_type,
                        grad_checkpoint=grad_checkpoint,
                    ),
                )
            else:
                self.layer = nn.Sequential(
                    conv_type(channels[0] + channels[1], channels[2], 1, padding=0),
                    ResidualGatedBlock(
                        channels[2],
                        conv_mult,
                        nonlin,
                        batchnorm=batchnorm,
                        dropout=dropout,
                        block_type=res_block_type,
                        grad_checkpoint=grad_checkpoint,
                    ),
                )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        return self.layer(x)


class ResidualMerger(MergeLayer):
    """
    By default for now simply a merge layer.
    """

    merge_type = "residual"

    def __init__(
        self,
        channels,
        conv_mult,
        nonlin,
        batchnorm,
        dropout,
        res_block_type,
        grad_checkpoint=False,
    ):
        super().__init__(
            channels,
            self.merge_type,
            conv_mult,
            nonlin,
            batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            grad_checkpoint=grad_checkpoint,
        )


class BlurPool(nn.Module):
    """
    BlurPool Layer: Applies a blur filter before downsampling to reduce aliasing artifacts.
    """

    def __init__(self, channels, stride=2):
        """
        Args:
            channels (int): Number of input channels.
            stride (int): Downsampling factor.
        """
        super(BlurPool, self).__init__()
        self.stride = stride

        # Define a simple low-pass filter (approximating Gaussian)
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()  # Normalize kernel

        # Expand kernel to all input channels
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        # Apply blur filter
        x = torch.nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=x.shape[1]
        )
        # Perform downsampling
        return x[:, :, :: self.stride, :: self.stride]


class BaseStochasticConvBlock(nn.Module):
    """Base class for all stochastic conv blocks with shared functionality."""

    def __init__(
        self,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        training_mode="supervised",
    ):
        super().__init__()
        self.training_mode = training_mode
        assert kernel % 2 == 1
        self.pad = kernel // 2
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars
        self.batch_size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(
            nn, f"Conv{conv_mult}d"
        )

    def update_mode(self, mode):
        self.training_mode = mode

    def _clamp_params(self, params):
        """Clamp parameters to prevent numerical instability."""
        mu, lv = params.chunk(2, dim=1)
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        lv = torch.clamp(lv, min=-10.0, max=10.0)
        std = torch.where(lv < 0, (lv / 2).exp(), 1 + lv)
        return mu, lv, std

    def _compute_logprob(self, p, z):
        """Compute log probability of z under distribution p."""
        if isinstance(p, Normal):
            logprob = p.log_prob(z)
        else:
            logprob = torch.stack([p_i.log_prob(z) for p_i in p], dim=-1)
        return logprob


class NormalStochasticConvBlock(BaseStochasticConvBlock):
    """
    Stochastic Conv Block for non-top layers.
    Always uses normal distribution (no mixture) and unconditional.
    """

    def __init__(
        self,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        training_mode="supervised",
    ):
        super().__init__(c_in, c_vars, c_out, conv_mult, kernel, training_mode)

        self.conv_in_q = self.conv_type(c_in, 2 * c_vars, kernel, padding=self.pad)
        self.conv_out = self.conv_type(c_vars, c_out, kernel, padding=self.pad)

    def forward(self, p_params, q_params):
        self.batch_size = q_params.shape[0]

        # Process prior parameters
        p_mu, _, p_std = self._clamp_params(p_params)
        p = Normal(p_mu, p_std)

        # Define q(z)
        q_params = self.conv_in_q(q_params)
        q_mu, q_lv, q_std = self._clamp_params(q_params)
        q = Normal(q_mu, q_std)

        # Sample and compute output
        z = q.rsample()
        out = self.conv_out(z)

        # Compute KL divergence
        kl = kl_divergence(q, p).mean()

        # Compute log probabilities
        logprob_p = self._compute_logprob(p, z)
        logprob_q = self._compute_logprob(q, z)

        data = {
            "z": z,
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": logprob_p,
            "logprob_q": logprob_q,
            "kl": kl,
            "mu": q_mu,
            "lv": q_lv,
        }

        return out, data


class MixtureStochasticConvBlock(BaseStochasticConvBlock):
    """
    Top layer stochastic block with Gaussian Mixture Model and conditioning.
    Uses q(y|x) and q(z|x,y) with FiLM modulation.
    Always conditional, always mixture prior.



    """

    def __init__(
        self,
        c_in,
        c_vars,
        c_out,
        conv_mult,
        kernel=3,
        training_mode="supervised",
        n_components=4,
        seg_head_dim=2,  # Spatial dim of feature map for segmentation head q(y|x)
    ):
        super().__init__(c_in, c_vars, c_out, conv_mult, kernel, training_mode)
        self.n_components = n_components
        self.seg_head_dim = seg_head_dim
        self.temperature = 1.0  # Initial temperature for Gumbel-Softmax
        self.constant = 200  # Scaling constant for distances
        self.group_size = 8
        self.prior_probs = torch.ones(n_components, device=self.device) / n_components

        # q(y|x) network -> segmentation head
        self.qy_x = nn.Sequential(
            self.conv_type(c_in, c_vars, kernel, padding=self.pad),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c_vars * (seg_head_dim**2), n_components),
        )

        # q(z|x,y) network
        self.qz_xy = nn.Sequential(
            self.conv_type(c_in, 2 * c_vars, kernel, padding=self.pad),
            nn.ReLU(),
            self.conv_type(2 * c_vars, 2 * c_vars, kernel, padding=self.pad),
        )

        # FiLM modulation layers
        self.gamma_layer = nn.Linear(n_components, c_in)
        self.beta_layer = nn.Linear(n_components, c_in)

        self.conv_out = self.conv_type(c_vars, c_out, kernel, padding=self.pad)

    def forward(self, p_params, q_params, label, confidence_threshold=None):
        """
        
        Outputs:
        out: output tensor after sampling and conv
        data: 
            pi: `q(y|x)` class probabilities
        
        
        """
        self.batch_size = q_params.shape[0]

        # Process prior parameters (mixture of Gaussians)
        p_mu, _, p_std = self._clamp_params(p_params)
        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)
        p_components = [Normal(mu, std) for mu, std in zip(p_mu_chunks, p_std_chunks)]

        # Get q(y|x)
        qy_logits = self.qy_x(q_params)

        # FiLM modulation: condition z on y
        gamma = self.gamma_layer(qy_logits).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_layer(qy_logits).unsqueeze(-1).unsqueeze(-1)
        q_modulated = gamma * q_params + beta

        # Get q(z|x,y)
        qz_params = self.qz_xy(q_modulated)
        q_mu, q_lv, q_std = self._clamp_params(qz_params)
        q = Normal(q_mu, q_std)
        z = q.rsample()

        # Initialize outputs
        y = None
        cross_entropy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        pseudo_label = label

        # Handle different training scenarios
        if label is None:
            # Inference mode: use softmax for y
            y = F.softmax(qy_logits, dim=1)
            # TODO: check if this is the right way to handle KL in inference
            # Class is computed outside
            kl = 0.0
        elif self.training_mode == "semisupervised":
            # Semi-supervised learning: use pseudo-labeling for unlabeled data
            y, pseudo_label, cross_entropy = self._semisupervised_forward(
                label, q_mu, qy_logits, confidence_threshold
            )
            kl = self._compute_kl_mixture(q, p_components, label=pseudo_label)
        elif self.training_mode == "supervised":
            # Supervised learning: use Gumbel-Softmax
            y = F.gumbel_softmax(qy_logits, tau=self.temperature, hard=False)
            self._update_temperature()
            kl = self._compute_kl_mixture(q, p_components, label=label)
            cross_entropy = F.cross_entropy(qy_logits, label.long(), ignore_index=-1)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

        # Add regularization terms
        js_div = self._compute_js_div(y) if y is not None else 0
        kl = kl + js_div

        out = self.conv_out(z)
        logprob_p = self._compute_logprob(p_components, z)
        logprob_q = self._compute_logprob(q, z)

        data = {
            "z": z,
            "p_params": p_params,
            "q_params": q_params,
            "logprob_p": logprob_p,
            "logprob_q": logprob_q,
            "kl": kl,
            "mu": q_mu,
            "lv": q_lv,
            "class_probabilities": y,
            "cross_entropy": cross_entropy,
            "pseudo_labels": pseudo_label,
        }

        return out, data

    def _semisupervised_forward(self, label, q_mu, confidence_threshold):
        """Handle semi-supervised training logic with pseudo-labeling."""

        num_groups = self.batch_size // self.group_size
        anchors = torch.arange(
            0, num_groups * self.group_size, self.group_size, device=self.device
        )

        q_mu_anchors = q_mu[anchors]
        labels_anchors = label[anchors]

        # Compute class means from labeled anchor samples
        sums = torch.zeros(
            self.n_components,
            q_mu.size(1),
            q_mu.size(2),
            q_mu.size(3),
            device=self.device,
        )
        counts = torch.zeros(self.n_components, 1, 1, 1, device=self.device)

        for c in range(self.n_components):
            mask = labels_anchors == c
            if mask.any():
                sums[c] = q_mu_anchors[mask].sum(dim=0)
                counts[c] = mask.sum()

        means = sums / counts.clamp(min=1)

        # Compute distances and logits for pseudo-labeling
        diff = q_mu.unsqueeze(1) - means.unsqueeze(0)
        dists = (diff * diff).sum(dim=(2, 3, 4))
        logits = -dists / self.constant
        logits = logits - logits.max(dim=1, keepdim=True).values

        y = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        self._update_temperature()

        # Generate pseudo labels with confidence thresholding
        conf, pseudo = y.max(dim=1)
        pseudo[anchors] = label[anchors].long()
        accept = conf > confidence_threshold
        pseudo[~accept] = -1

        cross_entropy = F.cross_entropy(logits, pseudo, ignore_index=-1)

        return y, pseudo, cross_entropy

    def _compute_kl_mixture(self, q, p_components, label=None, y_pred=None):
        """Compute KL divergence for mixture model."""
        kl_divergences = [
            kl_divergence(q, p_i).mean(dim=(1, 2, 3)) for p_i in p_components
        ]
        kl_divergences = torch.stack(kl_divergences, dim=-1)

        if label is not None:
            valid_mask = label >= 0
            if valid_mask.any():
                kl = kl_divergences[valid_mask, label[valid_mask].long()].mean()
            else:
                kl = torch.tensor(0.0, device=self.device)
        elif y_pred is not None:
            kl = kl_divergences[range(self.batch_size), y_pred].mean()
        else:
            kl = kl_divergences.mean()

        return kl

    def _compute_js_div(self, y):
        """Compute Jensen-Shannon divergence between y and uniform prior."""
        m = 0.5 * (y + self.prior_probs)
        js_div = 0.5 * torch.sum(y * torch.log(y / (m + 1e-10)), dim=1) + 0.5 * torch.sum(
            self.prior_probs * torch.log(self.prior_probs / (m + 1e-10)), dim=1
        )
        return js_div.mean()

    def _update_temperature(self):
        """Update temperature for Gumbel-Softmax annealing."""
        self.temperature = max(0.5, self.temperature * 0.999)
