import torch
import torch.nn as nn

from timm.models.vision_transformer import Block


class FrequencyWiseTranformerPooling(nn.Module):

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


class AttentionPooling(nn.Module):

    def __init__(self, embed_dim, num_head=4):
        super().__init__()
        self.f_att_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.f_att_token, std=.02)
        self.frequency_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_head, batch_first=True)

    def forward(self, x):
        x, _ = self.frequency_att(
            query=self.f_att_token.repeat(x.shape[0], 1, 1),
            key=x,
            value=x,
        )
        return x.squeeze(1)


class ActivateAttention(nn.Module):

    def __init__(self, dim, num_heads=6, qv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.f_q = nn.Linear(dim, dim, bias=qv_bias)
        self.f_k = nn.Linear(dim, dim, bias=True)
        self.f_v = nn.Linear(dim, dim, bias=qv_bias)
        self.activate = nn.GELU()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        assert query.shape[0] == key.shape[0] == value.shape[0]
        B = query.shape[0]
        assert query.shape[2] == key.shape[2] == value.shape[2]
        C = query.shape[2]
        N_q, N_k, N_v = query.shape[1], key.shape[1], value.shape[1]
        assert N_k == N_v
        
        q = self.f_q(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.f_k(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.f_v(value).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        k = self.activate(k)
        attn = (q @ k.transpose(-2, -1))* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ActivateAttentionPooling(nn.Module):
    
    def __init__(self, embed_dim, num_head=4):
        super().__init__()
        self.f_att_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.f_att_token, std=.02)
        self.frequency_att = ActivateAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, x):
        x = self.frequency_att(
            query=self.f_att_token.repeat(x.shape[0], 1, 1),
            key=x,
            value=x,
        )
        return x