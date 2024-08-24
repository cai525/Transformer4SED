import torch
import torch.nn.functional as F
from torch import nn
import random


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size,start_c=0,end_c=0):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.max_drop_prob=drop_prob
        self.block_size = block_size
        self.count=0
        self.start_c=start_c
        self.end_c=end_c

    def set_drop_prob(self):
        if  self.end_c==0:
            self.drop_prob=self.max_drop_prob
        else:

            if self.count<self.start_c:
                self.drop_prob=0.
            elif self.count<=self.end_c and self.count>=self.start_c:
                self.drop_prob=self.max_drop_prob*(self.count-self.start_c)/(self.end_c-self.start_c)
            else:
                self.drop_prob=self.max_drop_prob

        self.count = self.count + 1





    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training :
            return x

        if self.training:
            self.set_drop_prob()
        # print("self.drop_prob{},self.count{}".format(self.drop_prob, self.count))

        if  self.drop_prob == 0.:
                return x

        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)




def spec_augment_const_width(mel_spectrogram, F_size=0, T_size=0, F_num=0, T_num=0):
    T_MAX = mel_spectrogram.shape[1]
    F_MAX = mel_spectrogram.shape[2]

    warped_mel_spectrogram = mel_spectrogram

    for i in range(F_num):
        f = int(F_size)
        f0 = random.randint(0, F_MAX - f)
        warped_mel_spectrogram[:, :, f0:f0 + f] = 0

    for i in range(T_num):
        t = int(T_size)
        t0 = random.randint(0, T_MAX - t)
        warped_mel_spectrogram[:, t0:t0 + t, :] = 0

    return warped_mel_spectrogram



class SpecAug_Module(nn.Module):

    def __init__(self, F_size=0,F_num=0):
        super(SpecAug_Module, self).__init__()

        self.F_size = F_size
        self.F_num = F_num

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.F_size == 0 or self.F_num == 0:
            return x
        else:


            F_MAX = x.shape[3]

            assert self.F_size<F_MAX, \
                "Expected F_size<F_MAX"

            spec_mask = (torch.ones(x.shape))

            # place mask on input device
            spec_mask = spec_mask.to(x.device)


            for batch_idx in range(x.shape[0]):
                for i in range(self.F_num):
                    f = int(self.F_size)
                    f0 = random.randint(0, F_MAX - f)
                    spec_mask[batch_idx,:, :, f0:f0 + f] = 0

            # apply specAug mask
            out = x * spec_mask

            # scale output
            out = out * spec_mask.numel() / spec_mask.sum()

            return out






if __name__=="__main__":

    # (bsize, n_feats, depth, height, width)
    x = torch.rand(11, 32, 640, 64)
    drop_block =DropBlock2D(block_size=10, drop_prob=0.2,start_c=2,end_c=5)
    for i in range(10):
        regularized_x = drop_block(x)
        # drop_block
        # print(drop_block.drop_prob)
        print((regularized_x == 0).sum() * 1.0 / regularized_x.view(1, -1).shape[1])

    # x = torch.rand(11, 32, 640, 64)
    #
    # M=SpecAug_Module(F_size=5,F_num=5)
    # M.train()
    # spected_x=M(x)
    # print((spected_x == 0).sum() * 1.0 / spected_x.view(1, -1).shape[1])
    #
    # M.eval()
    # spected_x=M(x)
    # print((spected_x == 0).sum() * 1.0 / spected_x.view(1, -1).shape[1])
