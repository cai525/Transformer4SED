from functools import partial

import torch.nn as nn
import torch
from torchvision.ops import drop_block2d
from timm.models.resnet import downsample_avg

from src.models.cnn.base import GLU, ContextGating


class CNN1d(nn.Module):

    def __init__(self, in_channels, out_channel, kernel_size):
        super().__init__()
        self.conv_1d = nn.Conv1d(in_channels, out_channel, kernel_size)
        self.act_layer = nn.GELU()
        self.norm_layer = nn.BatchNorm1d(num_features=out_channel)

    def forward(self, x):
        x = self.conv_1d(x)
        x = self.norm_layer(x)
        x = self.act_layer(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channel,
            mid_channel,
            out_channel,
            stride=(1, 1),
            downsample=None,
            dilation=1,
            first_dilation=None,
            drop_block=None,
            drop_path=None,
    ):
        super(BasicBlock, self).__init__()
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(in_channel,
                               mid_channel,
                               kernel_size=3,
                               stride=stride[0],
                               padding=first_dilation,
                               dilation=first_dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            mid_channel,
            out_channel,
            kernel_size=3,
            stride=stride[0],
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.act2 = nn.GELU()
        self.downsample = downsample
        self.drop_block = drop_block
        self.drop_path = drop_path

    def forward(self, x):
        residual = x
        # original resnet
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)

        return x


class ResNetV2Block(BasicBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn1 = nn.BatchNorm2d(kwargs['in_channel'])
        self.bn2 = nn.BatchNorm2d(kwargs['mid_channel'])

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        return x


class SqueezeFreq(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.squeeze(-1)


class UnsqueezeFreq(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)


class ResNet(nn.Module):

    def __init__(self,
                 n_in_channel,
                 kernel_size=[3, 3, 3],
                 padding=[1, 1, 1],
                 stride=[1, 1, 1, 1, 1],
                 nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 res_block_type: str = "basic",
                 drop_block: bool = False,
                 drop_block_width=[3, 3, 3],
                 cnn_1d_dict: dict = None):
        """
            Initialization of CNN network s
        
        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
            resnetv2: whether to use resnetv2
        """
        super(ResNet, self).__init__()
        assert len(kernel_size) == len(padding) == len(stride) == len(nb_filters) == len(pooling)
        assert len(kernel_size) % 2 == 0
        self.nb_filters = nb_filters
        cnn = nn.Sequential()
        if res_block_type == "resnetv2":
            cnn.add_module(
                "pre_cov",
                nn.Conv2d(in_channels=1, out_channels=nb_filters[0], kernel_size=3),
            )
            n_in_channel = 16

        def res_block(i, dropblock=None, res_block_type="basic"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i + 1]
            nMid = nb_filters[i]
            downsample = None if nIn == nOut else downsample_avg(nIn, nOut, 1)
            cnn.add_module(
                "res_block {0}".format(i),
                self._get_res_block(
                    res_block_type=res_block_type,
                    in_channel=nIn,
                    mid_channel=nMid,
                    out_channel=nOut,
                    stride=(stride[i], stride[i + 1]),
                    downsample=downsample,
                    drop_block=dropblock,
                ),
            )

        # 128x862x64
        for i in range(0, len(nb_filters), 2):
            res_block(
                i,
                dropblock=None if not drop_block else \
                    partial(drop_block2d, p=drop_block, block_size=drop_block_width[i]),
                res_block_type=res_block_type,
            )
            cnn.add_module(
                "pooling{0}".format(i),
                nn.AvgPool2d(pooling[i + 1]),
            )
        if cnn_1d_dict:
            cnn.add_module(
                "squeeze_freq",
                SqueezeFreq(),
            )
            cnn_1d_len = len(cnn_1d_dict["kernel_size"])
            for i in range(cnn_1d_len):
                nIn = nb_filters[-1] if i == 0 else cnn_1d_dict["filters"][i - 1]
                nOut = cnn_1d_dict["filters"][i]
                cnn.add_module("cnn_1d_{0}".format(i),
                               CNN1d(
                                   in_channels=nIn,
                                   out_channel=nOut,
                                   kernel_size=cnn_1d_dict["kernel_size"][i],
                               ))
            cnn.add_module(
                "unsqueeze_freq",
                UnsqueezeFreq(),
            )

        self.cnn = cnn

    def _get_res_block(self, res_block_type, *args, **kwargs):
        if res_block_type == "basic":
            return BasicBlock(*args, **kwargs)
        elif res_block_type == "resnetv2":
            return ResNetV2Block(*args, **kwargs)
        else:
            raise NotImplementedError("Unknown resnet type {0}".format(res_block_type))

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.cnn(x)
        return x
