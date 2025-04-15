from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.models.transformer.transformerXL import RelPositionMultiheadAttention



class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
        normalize_before (bool): whether to use layer_norm before the first block.
        causal (bool): Whether to use causal convolution in conformer encoder
            layer. This MUST be True when using dynamic_chunk_training and streaming decoding.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        normalize_before: bool = True,
        causal: bool = False,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel, causal=causal)

        self.norm_ff_macaron = nn.LayerNorm(d_model)  # for the macaron style FNN module
        self.norm_ff = nn.LayerNorm(d_model)  # for the FNN module
        self.norm_mha = nn.LayerNorm(d_model)  # for the MHA module

        self.ff_scale = 0.5

        self.norm_conv = nn.LayerNorm(d_model)  # for the CNN module
        self.norm_final = nn.LayerNorm(d_model)  # for the final output of the block

        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        # macaron style feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff_macaron(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(src))
        if not self.normalize_before:
            src = self.norm_ff_macaron(src)

        # multi-headed self-attention module
        residual = src
        if self.normalize_before:
            src = self.norm_mha(src)

        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = residual + self.dropout(src_att)
        if not self.normalize_before:
            src = self.norm_mha(src)

        # convolution module
        residual = src
        if self.normalize_before:
            src = self.norm_conv(src)

        src, _ = self.conv_module(src, src_key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src)

        if not self.normalize_before:
            src = self.norm_conv(src)

        # feed forward module
        residual = src
        if self.normalize_before:
            src = self.norm_ff(src)
        src = residual + self.ff_scale * self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm_ff(src)

        if self.normalize_before:
            src = self.norm_final(src)

        return src


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: Tensor) -> Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).
        causal (bool): Whether to use causal convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.causal = causal

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.lorder = kernel_size - 1
        padding = (kernel_size - 1) // 2
        if self.causal:
            padding = 0

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(
        self,
        x: Tensor,
        cache: Optional[Tensor] = None,
        right_context: int = 0,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
            cache: The cache of depthwise_conv, only used in real streaming
                decoding.
            right_context:
              How many future frames the attention can see in current chunk.
              Note: It's not that each individual frame has `right_context` frames
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
        if self.causal and self.lorder > 0:
            if cache is None:
                # Make depthwise_conv causal by
                # manualy padding self.lorder zeros to the left
                x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                assert not self.training, "Cache should be None in training time"
                assert cache.size(0) == self.lorder
                x = torch.cat([cache.permute(1, 2, 0), x], dim=2)
                if right_context > 0:
                    cache = x.permute(2, 0, 1)[
                        -(self.lorder + right_context) : (-right_context),  # noqa
                        ...,
                    ]
                else:
                    cache = x.permute(2, 0, 1)[-self.lorder :, ...]  # noqa

        x = self.depthwise_conv(x)
        # x is (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        if cache is None:
            cache = torch.empty(0)

        return x.permute(2, 0, 1), cache