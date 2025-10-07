from typing import Optional

import torch.nn as nn
from torch import Tensor


class CrossAttentionFirstDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._sa_block(self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm2(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm3(x + self._ff_block(x))
        return x


class QueryBasedAudioTaggingDecoder(nn.Module):

    def __init__(self, n_layers, d_model, nhead, dim_ffn, activation="gelu"):
        super().__init__()
        decoder_layer = CrossAttentionFirstDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            activation=activation,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, feat_encoder: Tensor, queries: Tensor, tgt_mask=None):
        out = self.decoder(queries, feat_encoder, tgt_mask=tgt_mask)
        return out
