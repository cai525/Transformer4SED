import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class FrequencyWiseTranformerPooling(nn.Module):
    """
        Ref: K. Li, Y. Song, L. -R. Dai, I. McLoughlin, X. Fang and L. Liu, 
        "AST-SED: An Effective Sound Event Detection Method Based on Audio Spectrogram Transformer,
        " ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.linear_emb = nn.Linear(1, embed_dim)
        self.frequency_transformer = nn.ModuleList(
            Block(dim=embed_dim, num_heads=4, mlp_ratio=4, norm_layer=nn.LayerNorm) for i in range(2))
        self.frequency_transformer_norm = nn.LayerNorm(768)

    def forward(self, x):
        tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
        x = torch.cat([tag_token, x], dim=1)
        for block in self.frequency_transformer:
            x = block(x)
        x = self.frequency_transformer_norm(x)
        x = x[:, 0, :]
        return x


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling, self).__init__()
        pass

    def forward(self, input):
        #input shape, B,T,C
        return torch.mean(input, dim=1)  #return B,C
