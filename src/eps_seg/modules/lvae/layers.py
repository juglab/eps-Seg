import torch
from torch import nn
from typing import Type, Union, Optional, Tuple
from torch.distributions import Normal
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
        layer_number: int,
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
            self.stochastic = NormalStochasticConvBlock(
                layer_number=layer_number,
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
        if self.is_top_layer:
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
        spatial_dims: tuple = top_prior_param_shape[
            -self.conv_mult :
        ]  # Spatial resolution

        # Each GMM component uses an equal fraction of the channels
        channels_per_component = total_channels // (
            2 * n_components
        )  # Half for mus, half for sigmas

        # Initialize the tensor for means (mus)
        chunk_values = torch.zeros(
            (
                n_components,
                channels_per_component,
            )
            + spatial_dims
        )
        # TODO hardcoded 5.0, better initialization strategy?
        # Dynamically assign values to means
        chunk_size = channels_per_component // n_components
        for i in range(n_components):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunk_values[i, start_idx:end_idx] = (
                5.0  # Equidistant initialization for means
            )

        mus = chunk_values.view(
            (
                1,
                n_components * channels_per_component,
            )
            + spatial_dims
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
        use_mode=False,
        force_constant_output=False,
        forced_latent=None,
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
        layer_number: int,
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
                self.blurpool = BlurPool(inner_filters, stride=2, dim=conv_mult)
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

    def __init__(self, channels, stride=2, dim=2):
        """
        Args:
            channels (int): Number of input channels.
            stride (int): Downsampling factor.
            dim (int): Dimensionality of the input (2 or 3).
        """
        super(BlurPool, self).__init__()
        self.stride = stride
        self.dim = dim

        # Define a simple low-pass filter (approximating Gaussian)
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)

        kernel = (
            kernel[:, None] * kernel[None, :]
            if dim == 2
            else kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
        )
        kernel = kernel / kernel.sum()  # Normalize kernel

        # Expand kernel to all input channels
        kernel = (
            kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            if dim == 2
            else kernel.view(1, 1, 3, 3, 3).repeat(channels, 1, 1, 1, 1)
        )

        self.register_buffer("kernel", kernel)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
        Returns:
            Tensor: Downsampled tensor of shape (N, C, H/stride, W/stride) for 2D or (N, C, D/stride, H/stride, W/stride) for 3D.

        """

        if self.dim == 2:
            x = F.conv2d(x, self.kernel, stride=1, padding=1, groups=x.shape[1])
            # Downsample
            return x[:, :, :: self.stride, :: self.stride]
        else:
            x = F.conv3d(x, self.kernel, stride=1, padding=1, groups=x.shape[1])
            # Downsample all spatial dimensions
            return x[:, :, :: self.stride, :: self.stride, :: self.stride]


class BaseStochasticConvBlock(nn.Module):
    """Base class for all stochastic conv blocks with shared functionality."""

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

        # Process prior parameters
        p_mu, _, p_std = self._clamp_params(p_params)
        p = Normal(p_mu, p_std)

        # Define q(z)
        q_params = self.conv_in_q(q_params)
        q_mu, _, q_std = self._clamp_params(q_params)
        q = Normal(q_mu, q_std)

        # Sample and compute output
        z = q.rsample()
        out = self.conv_out(z)

        data = {
            "prior": p,
            "posterior": q,
            "mu": q_mu,
            "z": z,
            "class_logits": None,
            "class_probabilities": None,
        }

        return out, data


class MixtureStochasticConvBlock(BaseStochasticConvBlock):
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
        seg_head_dim=2,  # Spatial dim of feature map for segmentation head q(y|x)
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

        # Process prior parameters (mixture of Gaussians)
        p_mu, _, p_std = self._clamp_params(p_params)
        p_mu_chunks = p_mu.chunk(self.n_components, dim=1)
        p_std_chunks = p_std.chunk(self.n_components, dim=1)
        p_components = [Normal(mu, std) for mu, std in zip(p_mu_chunks, p_std_chunks)]

        # Get q(y|x) from prior parameters
        # and q(z|x,y) from conditional prior
        qz_params, class_logits, class_probs = self.conditional_layer(q_params)
        q_mu, _, q_std = self._clamp_params(qz_params)
        q = Normal(q_mu, q_std)
        z = q.rsample()

        out = self.conv_out(z)

        data = {
            "prior": p_components,
            "posterior": q,
            "mu": q_mu,
            "z": z,
            "class_logits": class_logits,
            "class_probabilities": class_probs,
        }

        return out, data



class FeatureSubsetSelectionLayer(nn.Module):
    """
    Layer that selects a spatial subset of the feature map.

    - Works for 2D (B, C, H, W) and 3D (B, C, D, H, W).
    - Keeps all channels; only crops spatial dims.
    - Typically used as a center crop, but can also use explicit indices.

    Args
    ----
    crop_size : tuple or None
        Spatial size to crop to.
        - For 2D: (h, w)
        - For 3D: (d, h, w)
        If None: passthrough.
    center_crop : bool
        If True, performs center crop with given crop_size.
    spatial_start : tuple or None
        If center_crop=False, you can specify explicit starting indices.
        - For 2D: (h0, w0)
        - For 3D: (d0, h0, w0)
    enabled : bool
        If False, always passthrough.
    """

    def __init__(
        self,
        layer_number: int,
        crop_size: Optional[Tuple[int, ...]] = None,
        enabled: bool = True,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.crop_size = crop_size
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) or (B, C, D, H, W)
        """
        spatial_dims = x.shape[2:]   # everything after channel
    
        starts = [(sd - self.crop_size) // 2 for sd in spatial_dims]
        ends   = [s + self.crop_size for s in starts]

        # Build slices dynamically
        slices = [slice(None), slice(None)]   # keep B, C
        slices += [slice(s, e) for s, e in zip(starts, ends)]

        return x[tuple(slices)]

class SegmentationHead(nn.Module):
    """Segmentation head that mirrors the original API but fixes batch mixing.

    The implementation keeps the same arguments and pooling behaviour as the
    original version, with per-level convolutions followed by global average
    pooling. The only functional difference is concatenating pooled vectors
    along the feature dimension (dim=1) instead of the batch dimension to avoid
    shape mismatches during classification.
    """

    def __init__(self,
                 in_channels: int,
                 n_classes: int,
                 conv_mult: int,
                 hidden_channels: int = None,
                 n_layers: int = 3,
                 kernel: int = 1):
        super().__init__()
        assert kernel % 2 == 1

        if hidden_channels is None:
            hidden_channels = in_channels

        self.conv_mult = conv_mult
        conv_type = getattr(nn, f"Conv{conv_mult}d")

        self.level_nets = nn.ModuleList([
            nn.Sequential(
                conv_type(in_channels, hidden_channels, kernel_size=kernel),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_layers)
        ])

        # self.classifier = nn.Linear(n_layers * hidden_channels, n_classes)
        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool each level and concatenate along the feature dimension.

        Args:
            x: Iterable of feature maps with shape (B, C, H, W) for 2D or
                (B, C, D, H, W) for 3D.

        Returns:
            Logits of shape (B, n_classes).
        """

        flattened = []
        for z, net in zip(x, self.level_nets):
            out = net(z)                      # (B, C, H, W) or (B, C, D, H, W)
            out = out.flatten(start_dim=1)    # (B, C*H*W) or (B, C*D*H*W)
            flattened.append(out)

        feat = torch.cat(flattened, dim=1)
        logits = self.classifier(feat)
        return logits



class ConditionalPrior(nn.Module):
    """
    Conditional prior network that predicts parameters for a conditional prior distribution:

        q(y|x)      : class logits/probs
        FiLM(x, y)  : feature modulation
        q(z|x, y)   : Gaussian parameters (mu, logvar) for posterior

    This does NOT sample z or compute KL; it just outputs q-params + class info.
    """

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

        # q(y|x) head (similar to old qy_x)
        self.qy_x = nn.Sequential(
            self.conv_type(c_in, c_vars, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(c_vars * (seg_head_dim**conv_mult), n_components),
        )

        # FiLM modulation layers
        self.gamma_layer = nn.Linear(n_components, c_in)
        self.beta_layer = nn.Linear(n_components, c_in)

        # q(z|x, y) network (similar to old qz_xy)
        self.qz_xy = nn.Sequential(
            self.conv_type(c_in, 2 * c_vars, kernel, padding=pad),
            nn.ReLU(inplace=True),
            self.conv_type(2 * c_vars, 2 * c_vars, kernel, padding=pad),
        )

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        x: feature map used to infer q(y|x) and q(z|x,y), shape (B, c_in, ...)

        Returns:
            qz_params      : (B, 2 * c_vars, ...)
            class_logits   : (B, n_components)
            class_probs    : (B, n_components)
        """
        # q(y|x)
        class_logits = self.qy_x(x)  # (B, n_components)
        class_probs = F.softmax(class_logits / temperature, dim=-1)

        # FiLM modulation
        gamma = self.gamma_layer(class_probs)  # (B, c_in)
        beta = self.beta_layer(class_probs)  # (B, c_in)

        # Broadcast to spatial dims
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        x_mod = gamma * x + beta

        # q(z|x, y)
        qz_params = self.qz_xy(x_mod)

        return qz_params, class_logits, class_probs

