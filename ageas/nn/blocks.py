#!/usr/bin/env python3
"""Reusable neural network building blocks.

Provides the residual MLP, ResNet-style 1D-conv, and recurrent blocks that
:class:`~ageas.nn.NN_Classifier` and :class:`~ageas.nn.Mixer_Classifier`
stack to form their feature extractor stages.
"""
import torch.nn as nn

__all__ = [
    'Basic',
    'Residual_Encoder',
    'Factorized_Residual_Mixer',
    'ResNet_Basic',
    'RNN_Basic',
    'RNN_Residual_Encoder',
]


def get_block(name: str) -> type:
    """Resolve a block class by its public name.

    Args:
        name: Name of the block class. Must be listed in :data:`__all__`.

    Returns:
        The matching block class.

    Raises:
        ValueError: If ``name`` is not in :data:`__all__`.
    """
    if name in __all__:
        return globals()[name]
    raise ValueError(
        f"Block '{name}' not found. Available blocks: {__all__}"
    )


class Basic(nn.Module):
    """Basic single-layer block: ``Linear -> Norm -> ReLU``.

    Attributes:
        layer: Linear projection.
        norm: Normalisation layer.
        activation: ReLU activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1000,
        bias: bool = True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ) -> None:
        """Initialize a Basic block.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            bias: If ``True``, include bias in the linear layer.
            norm_layer: Normalisation layer class.
            **kwargs: Ignored extra arguments.
        """
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm = norm_layer(out_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Residual_Encoder(nn.Module):
    """Two-layer residual MLP encoder with a learnable skip connection.

    Projects the input through a latent dimension of size
    ``(in_dim + out_dim) // latent_dim_divisor`` and adds a residual branch,
    with a linear projection if ``in_dim != out_dim`` and an identity
    otherwise.

    Attributes:
        encoder0: First linear projection (input → latent).
        norm0: Normalisation after the first projection.
        encoder1: Second linear projection (latent → output).
        residual: Skip connection (linear or identity).
        activation: ReLU activation.
        norm1: Normalisation after the residual add.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim_divisor: int = 2,
        out_dim: int = 1000,
        bias: bool = True,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ) -> None:
        """Initialize a Residual_Encoder.

        Args:
            in_dim: Input feature dimension.
            latent_dim_divisor: Divisor for computing the latent dimension
                from ``(in_dim + out_dim)``.
            out_dim: Output feature dimension.
            bias: If ``True``, include biases in the linear layers.
            norm_layer: Normalisation layer class.
            **kwargs: Ignored extra arguments.
        """
        super().__init__()
        latent_dim = (in_dim + out_dim) // latent_dim_divisor
        self.encoder0 = nn.Linear(in_dim, latent_dim, bias=bias)
        self.norm0 = norm_layer(latent_dim)
        self.encoder1 = nn.Linear(latent_dim, out_dim, bias=bias)
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=bias)
            if in_dim != out_dim
            else nn.Sequential()
        )
        self.activation = nn.ReLU(inplace=True)
        self.norm1 = norm_layer(out_dim)

    def forward(self, x):
        out = self.encoder0(x)
        out = self.norm0(out)
        out = self.activation(out)
        out = self.encoder1(out)
        out = out + self.residual(x)
        out = self.norm1(out)
        out = self.activation(out)
        return out


class Factorized_Residual_Mixer(nn.Module):
    """Factorized residual mixer block.

    Replaces standard local convolutions with global spatial mixing and
    factorises channel mixing (1×1 convolutions) from global spatial mixing
    (sequence-length convolutions). Expects input shape
    ``(Batch, Channels, Length)``.

    Note:
        Bottleneck in torchvision places the stride for downsampling at the
        3×3 convolution (``self.conv2``) while the original implementation
        places it at the first 1×1 convolution (``self.conv1``) per *Deep
        Residual Learning for Image Recognition*
        (https://arxiv.org/abs/1512.03385). This variant (ResNet V1.5)
        improves accuracy.

    Attributes:
        expansion: Channel expansion factor for the output.
        global_conv: Whether to use global (sequence-length) convolution.
        out_len: Output sequence length after striding.
    """

    expansion = 4

    def __init__(
        self,
        len_in: int,
        inplanes: int,
        planes: int,
        global_conv: bool = True,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        bias: bool = False,
        dilation: int = 1,
        norm_layer=nn.BatchNorm1d,
        **kwargs,
    ) -> None:
        """Initialize a Factorized_Residual_Mixer block.

        Args:
            len_in: Input sequence length.
            inplanes: Number of input channels.
            planes: Base number of output channels (expanded by
                :attr:`expansion`).
            global_conv: If ``True``, use a kernel spanning the full input
                length for spatial mixing.
            stride: Stride for the spatial mixing convolution.
            downsample: Optional downsampling module for the skip connection.
            groups: Number of convolution groups.
            base_width: Base width for grouped convolutions.
            bias: If ``True``, include biases in the convolutions.
            dilation: Dilation factor for local convolutions.
            norm_layer: Normalisation layer class.
            **kwargs: Ignored extra arguments.
        """
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.global_conv = global_conv
        self.out_len = len_in // stride

        if self.global_conv:
            padding = 0
            temp_planes = width * self.out_len
            kernel_size = len_in
        else:
            padding = dilation
            temp_planes = width
            kernel_size = 3

        # 1. Channel Mixing (cross-channel, independent of spatial position)
        self.conv1 = nn.Conv1d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)

        # 2. Global Spatial Mixing (cross-position, fully connected along sequence)
        self.conv2 = nn.Conv1d(
            width,
            temp_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=width,
            bias=bias,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)

        # 3. Channel Mixing (refinement)
        self.conv3 = nn.Conv1d(
            width, planes * self.expansion, kernel_size=1, stride=1, bias=bias
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.global_conv:
            out = out.reshape(out.shape[0], -1, self.out_len)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet_Basic(nn.Module):
    """Basic ResNet 1D block, optionally using global convolution.

    When ``global_conv=True`` the convolution kernel spans the whole input
    sequence so that each output position aggregates information across the
    full feature axis instead of only a local 3-wide receptive field.

    Attributes:
        expansion: Channel expansion factor (1 for basic blocks).
        global_conv: Whether to use global (sequence-length) convolution.
    """

    expansion = 1

    def __init__(
        self,
        len_in: int,
        inplanes: int,
        planes: int,
        global_conv: bool = True,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        norm_layer=nn.BatchNorm1d,
        **kwargs,
    ) -> None:
        """Initialize a ResNet_Basic block.

        Args:
            len_in: Input sequence length.
            inplanes: Number of input channels.
            planes: Number of output channels.
            global_conv: If ``True``, use a global (sequence-length) kernel.
            stride: Convolution stride.
            downsample: Optional downsampling module for the skip connection.
            groups: Must be 1 for this block type.
            bias: If ``True``, include biases.
            dilation: Dilation factor (must be 1 for this block type).
            norm_layer: Normalisation layer class.
            **kwargs: Ignored extra arguments.

        Raises:
            ValueError: If ``groups != 1``.
            NotImplementedError: If ``dilation > 1``.
        """
        super().__init__()
        if groups != 1:
            raise ValueError('ResNet_Basic only supports groups=1')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in ResNet_Basic')

        self.global_conv = global_conv
        if global_conv:
            padding = 0
            temp_planes = planes * len_in
            kernel_size = len_in
        else:
            padding = dilation
            temp_planes = planes
            kernel_size = 3

        self.conv1 = nn.Conv1d(
            inplanes,
            temp_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes,
            temp_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.global_conv:
            out = out.reshape(out.shape[0], -1, identity.shape[-1])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.global_conv:
            out = out.reshape(out.shape[0], -1, identity.shape[-1])
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class RNN_Basic(nn.Module):
    """Basic recurrent block backed by ``nn.RNN``, ``nn.LSTM``, or ``nn.GRU``.

    For bidirectional layers, ``out_dim`` must be even; the per-direction
    hidden size is ``out_dim // 2`` so that the concatenated output keeps the
    requested dimensionality.

    Attributes:
        hidden_size: Hidden size per direction.
        bidirectional: 2 if bidirectional, else 1.
        num_layers: Number of recurrent layers.
        layer_type: One of ``'RNN'``, ``'LSTM'``, ``'GRU'``.
        layer: The underlying recurrent module.
        norm: Normalisation layer applied to the output.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1000,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = True,
        proj_size: int = 0,
        norm_layer=nn.LayerNorm,
        block_layer_type: str = 'RNN',
        **kwargs,
    ) -> None:
        """Initialize an RNN_Basic block.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension. Must be even when
                ``bidirectional=True``.
            num_layers: Number of recurrent layers.
            nonlinearity: Activation for RNN (``'tanh'`` or ``'relu'``).
            bias: If ``True``, include recurrent biases.
            batch_first: If ``True``, input/output tensors are
                ``(batch, seq, feature)``.
            dropout: Dropout probability applied between recurrent layers.
            bidirectional: If ``True``, use a bidirectional variant.
            proj_size: LSTM projection size (only used for LSTM).
            norm_layer: Normalisation layer class applied to outputs.
            block_layer_type: One of ``'RNN'``, ``'LSTM'``, ``'GRU'``.
            **kwargs: Ignored extra arguments.

        Raises:
            ValueError: If ``bidirectional=True`` and ``out_dim`` is odd,
                or if ``block_layer_type`` is not recognised.
        """
        super().__init__()
        if bidirectional:
            assert out_dim % 2 == 0, \
                'out_dim must be divisible by 2 for bidirectional RNN'
            self.hidden_size = out_dim // 2
            self.bidirectional = 2
        else:
            self.hidden_size = out_dim
            self.bidirectional = 1
        self.num_layers = num_layers
        self.layer_type = block_layer_type

        common = dict(
            input_size=in_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        if self.layer_type == 'RNN':
            self.layer = nn.RNN(**common, nonlinearity=nonlinearity)
        elif self.layer_type == 'LSTM':
            self.layer = nn.LSTM(**common, proj_size=proj_size)
        elif self.layer_type == 'GRU':
            self.layer = nn.GRU(**common)
        else:
            raise ValueError(
                f"layer_type '{self.layer_type}' not supported. "
                "Choose 'RNN', 'LSTM', or 'GRU'."
            )

        self.norm = norm_layer(out_dim)

    def _main_forward(self, x) -> tuple:
        """Run the recurrent layer from a fresh (zero) state.

        Args:
            x: ``(batch, seq_len, in_dim)`` tensor, or the ``(out, h, c)``
                tuple of a preceding block. Only its output is used: blocks
                are connected by their outputs, never by hidden state.

        Returns:
            Tuple ``(x, out, h, c)``: the input tensor, the recurrent output,
            and the final hidden (and, for LSTM, cell) state.
        """
        if isinstance(x, tuple):
            x = x[0]
        if self.layer_type == 'LSTM':
            out, (h, c) = self.layer(x)
        else:
            out, h = self.layer(x)
            c = None
        return x, out, h, c

    def forward(self, x) -> tuple:
        """Forward pass over a token sequence.

        Args:
            x: ``(batch, seq_len, in_dim)`` tensor or a preceding block's
                ``(out, h, c)`` tuple.

        Returns:
            Tuple ``(out, h, c)`` with the normalised output.
        """
        _, out, h, c = self._main_forward(x)
        return (self.norm(out), h, c)


class RNN_Residual_Encoder(RNN_Basic):
    """RNN encoder with a learnable residual skip from inputs to outputs.

    Extends :class:`RNN_Basic` by adding a linear residual projection when
    ``in_dim != out_dim`` (identity otherwise) to ease optimisation in
    deeper recurrent stacks.

    Attributes:
        residual: Linear skip connection (or ``nn.Sequential()`` identity).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1000,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """Initialize an RNN_Residual_Encoder.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            bias: If ``True``, include bias in the residual projection.
            **kwargs: Forwarded to :class:`RNN_Basic`.
        """
        super().__init__(in_dim=in_dim, out_dim=out_dim, bias=bias, **kwargs)
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=bias)
            if in_dim != out_dim
            else nn.Sequential()
        )

    def forward(self, x) -> tuple:
        """Forward pass with residual skip connection.

        Args:
            x: ``(batch, seq_len, in_dim)`` tensor or a preceding block's
                ``(out, h, c)`` tuple.

        Returns:
            Tuple ``(out, h, c)`` with the normalised output.
        """
        x, out, h, c = self._main_forward(x)
        return (self.norm(out + self.residual(x)), h, c)
