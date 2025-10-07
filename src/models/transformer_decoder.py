from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from src.models.transformer.transformerXL import TransformerXL, RelPositionalEncoding
from src.models.transformer.conformer import ConformerEncoderLayer
from src.models.transformer.transformerxl_token import TransformerXLWithClsToken
from src.models.transformer.mask import diagonal_mask


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        input_dim,
        decoder_layer_num=2,
        pos_embed_strategy="learnable",
        seq_len=1000,
        attn_drop=0,
        num_heads=12,
        mlp_ratio=1,
    ) -> None:
        super(TransformerDecoder, self).__init__()

        self.input_dim = input_dim

        # self.projector = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))

        self.blocks = nn.ModuleList([
            Block(input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm, attn_drop=attn_drop)
            for i in range(decoder_layer_num)
        ])

        self.pos_embed_strategy = pos_embed_strategy
        if pos_embed_strategy == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, input_dim))
            torch.nn.init.normal_(self.pos_embed, std=.02)
        elif pos_embed_strategy == "sincos":
            self.register_buffer("pos_embed", self.get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(seq_len)))
        else:
            raise RuntimeError("Unknown pos_embed_strategy")

    def forward(self, x):
        B, T, C = x.shape
        pos_emd = self.linear_pos(self.pos_embed[:T, :])
        for block in self.blocks:
            x = block(x)
        return x

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """  Getting sincos position embedding
            embed_dim: output dimension for each position
            pos: a list of positions to be encoded: size (M,)
            out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)  ##dimention index
        omega /= embed_dim / 2.  ##normed dimention index
        omega = 1. / 10000**omega  # (D/2,)##why 10000

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return torch.Tensor(emb)


class TransformerXLDecoder(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len=1000,
        window_len=None,
        decoder_layer_num=2,
        attn_drop=0,
        num_heads=12,
        mlp_ratio=1,
    ) -> None:
        super(TransformerXLDecoder, self).__init__()
        self.pos_embedding = RelPositionalEncoding(d_model=input_dim, dropout_rate=0, max_len=seq_len)
        self.encoder_blocks = nn.ModuleList([
            TransformerXL(input_dim,
                          num_heads=num_heads,
                          mlp_ratio=mlp_ratio,
                          norm_layer=nn.LayerNorm,
                          attn_drop=attn_drop) for i in range(decoder_layer_num)
        ])

        if isinstance(window_len, int):
            att_mask = diagonal_mask(seq_len=seq_len, mask_width=window_len)
        elif isinstance(window_len, Sequence):
            assert len(window_len) == num_heads
            att_mask = []
            for width in window_len:
                att_mask.append(diagonal_mask(seq_len=seq_len, mask_width=width))
            att_mask = torch.stack(att_mask, 0)
        else:
            att_mask = None
        # Buffers won’t be returned in model.parameters(),
        # so that the optimizer won’t have a change to update them.
        self.register_buffer("att_mask", att_mask)

    def forward(self, x):
        x, pos_emb = self.pos_embedding(x)
        B = x.shape[0]
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        for block in self.encoder_blocks:
            if (self.att_mask is not None) and (not isinstance(self.att_mask, Sequence)) and self.att_mask.ndim == 3:
                att_mask = self.att_mask.repeat(B, 1, 1)
            else:
                att_mask = self.att_mask
            x = block(x, pos_emb, att_mask=att_mask)  #  # (T, N, C)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        return x


class ConformerDecoder(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len=1000,
        window_len=None,
        decoder_layer_num=2,
        attn_drop=0,
        num_heads=12,
        mlp_ratio=1,
        kenrel_size=31,
        dilation=1,
    ) -> None:
        super().__init__()
        self.pos_embedding = RelPositionalEncoding(d_model=input_dim, dropout_rate=0, max_len=seq_len)
        self.blocks = nn.ModuleList([
            ConformerEncoderLayer(
                input_dim,
                num_heads,
                dim_feedforward=int(input_dim * mlp_ratio),
                cnn_module_kernel=kenrel_size,
                dilation=dilation,
                dropout=attn_drop,
            ) for i in range(decoder_layer_num)
        ])

        att_mask = diagonal_mask(seq_len=seq_len, mask_width=window_len) if window_len is not None else None
        # Buffers won’t be returned in model.parameters(),
        # so that the optimizer won’t have a change to update them.
        self.register_buffer("att_mask", att_mask)

    def forward(self, x):
        x, pos_emb = self.pos_embedding(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        for block in self.blocks:
            x = block(src=x, pos_emb=pos_emb, src_mask=self.att_mask)  #  # (T, N, C)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        return x
