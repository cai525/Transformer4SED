from typing import Sequence

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def diagonal_mask(seq_len, mask_width) -> Tensor:
    """ generate diagonal mask for attention matrix in MHSA(mult-head self-attention).
    The mask is a square matrix made up with bool value. The value which is near the diagnoal
    is ``False``, while other part is assigned to ``True``. 
    Args:
        seq_len : side length of the attention matrix.
        mask_width : width of the area that is assigned to ``False`` near the diagnoal.
    Ret:
        Tensor as the diagnoal attention mask, 2D mask :math:`(L, S)` where L is the target sequence length, S is 
        the source sequence length.attn_mask ensures that position i is allowed to attend the unmasked positions.
        positions with ``True`` are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
        is provided, it will be added to the attention weight.
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        mask[i, max(0, i - mask_width//2):i + mask_width//2] = False
    return mask


def passt_mask(size_f, size_t, mask_width) -> Tensor:
    """ generate diagonal mask for attention matrix in MHSA(mult-head self-attention) of PaSST model.
        The items out of the time range contorled by the mask_width parameter are assigned to `True`, 
        otherwise assigned to `True`.
        The input sequence to the PaSST must have the structure: [cls_token, dis_token, seq],
        and the seq can be viewed as [batch, frequency, time]
    Args:
        size_b, size_f, size_t: batch, frequency, and time length;
        mask_width : width of the area that is assigned to ``False`` near the diagnoal.
    Ret:
        Tensor as the diagnoal attention mask, 2D mask :math.attn_mask ensures that position i is allowed to attend
        the unmasked positions. Positions with ``True`` are not allowed to attend while ``False`` values will be 
        unchanged. If a FloatTensor is provided, it will be added to the attention weight.
    """
    seq_len = 2 + size_f*size_t     # consider cls_token and dis_token
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)      # set all the mask to false by default
    unit_mask = diagonal_mask(size_t, mask_width)               
    mask[2:, 2:] = unit_mask.repeat(size_f, size_f)
    return mask
    
    
        

class MlmModule:
    def __init__(self, mask_rate=0.15, mask_style=(0.8, 0.1, 0.1), strategy = "random", 
                 block_width=10, device=None) -> None:
        self.mask_rate = mask_rate
        self.mask_style = {
            "mask_token": mask_style[0],
            "random": mask_style[1],
            "self": mask_style[2]
        }
        self.strategy = strategy
        self.device = device
        self.block_width = block_width
    
    def setence_mask(self, token_seq:Tensor, mask_token:Tensor):
        B,T,C = token_seq.shape
        mask_id_seq = self.get_mask_id_seq(B, T)
        # TODO: replace the for loop to matrix operation
        token_seq_new = token_seq.clone()
        for i in range(B):
            for j in range(T):
                if not mask_id_seq[i,j]:
                    continue
                p = random.random()
                if p < self.mask_style["mask_token"]:
                    token_seq_new[i, j, :] = mask_token
                elif p < self.mask_style["mask_token"] + self.mask_style["random"]:
                    b = random.randint(0, B-1)
                    t = random.randint(0, T-1)        
                    token_seq_new[i, j, :] = token_seq[b, t, :]
                else:
                    pass
        return token_seq_new, mask_id_seq
                

    def get_mask_id_seq(self, batch_len, seq_len):
        if self.strategy == "random":
            id_seq = self.random_mask(batch_len, seq_len)
        elif self.strategy == "block":
            id_seq = self.block_mask(batch_len, seq_len, self.block_width)
        else:
            raise ValueError("Unknown mask strategy")
        return id_seq
    
    def random_mask(self, batch_len, seq_len):
        noise = torch.rand(batch_len, seq_len, device=self.device)
        id_seq = (noise <= self.mask_rate)
        return id_seq
    
    def block_mask(self, batch_len, seq_len,  block_width=10):
        num_seg = seq_len // block_width
        noise = torch.rand(batch_len, num_seg, device=self.device)
        noise_sort, _ = noise.sort()
        threshold = noise_sort[:, int(num_seg*self.mask_rate)]
        id_seq = torch.zeros(batch_len, seq_len, dtype=bool, device=self.device)
        id_seq[:, :num_seg*block_width] = (noise <= torch.unsqueeze(threshold, dim=-1)).repeat_interleave(block_width, dim=1)
        return id_seq
            
    
    def draw_mask(self, mask:Sequence[bool]):
        fig, ax = plt.subplots()
        ax.imshow(mask, 'gray', interpolation='none', aspect='auto')
        fig.show()
    


if __name__ == "__main__":
    mlm_util = MlmModule(mask_rate=0.15)
    mask = mlm_util.block_mask(batch_len=2,
                               seq_len=1000,
                               block_width=10)
    mask = list(mask[0, :])
    mlm_util.draw_mask(mask)
    
