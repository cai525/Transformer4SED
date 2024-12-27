# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm
import torchaudio
import wget
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple, trunc_normal_
# override the timm package to relax the input shape constraint.
from src.models.transformer.pos_embed import get_1d_sincos_pos_embed_from_grid


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(self,
                 label_dim=527,
                 fstride=10,
                 tstride=10,
                 input_fdim=128,
                 input_tdim=1024,
                 imagenet_pretrain=True,
                 audioset_pretrain=False,
                 model_size='base384',
                 verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),
                                                                                  str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            self.f_dim, self.t_dim = f_dim, t_dim
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                    1, self.original_num_patches,
                    self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim,
                                                                         self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :,
                                                  int(self.oringal_hw / 2) - int(t_dim / 2):int(self.oringal_hw / 2) -
                                                  int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed,
                                                                    size=(self.oringal_hw, t_dim),
                                                                    mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :,
                                                  int(self.oringal_hw / 2) - int(f_dim / 2):int(self.oringal_hw / 2) -
                                                  int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:

            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError(
                    'currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.'
                )
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if os.path.exists('/home/cpf/CPF/AST-SED/pretrained_model/audioset_10_10_0.4593.pth') == False:
                raise Exception("Lack pretrained model file")
                # this model performs 0.4593 mAP on the audioset eval set
                # os.makedirs(Path('/home/mnt/likang/AST-SED-main/pretrained_model/audioset_10_10_0.4593.pth').parent, exist_ok=True)
                # audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                # wget.download(audioset_mdl_url, out='./pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('/home/cpf/CPF/AST-SED/pretrained_model/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527,
                                   fstride=10,
                                   tstride=10,
                                   input_fdim=128,
                                   input_tdim=1024,
                                   imagenet_pretrain=False,
                                   audioset_pretrain=False,
                                   model_size='base384',
                                   verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            self.f_dim, self.t_dim = f_dim, t_dim
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212,
                                                                        768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2):50 - int(t_dim / 2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def get_mutable_pos_embedding(self, total_dim):
        freq_dim = self.f_dim
        self.t_dim = time_dim = int(total_dim / freq_dim)
        pos_list = [self.v.pos_embed[:, 2 + f * time_dim:2 + (f + 1) * time_dim, :] for f in range(freq_dim)]
        pos_list.insert(0, self.v.pos_embed[:, :2, :])
        return torch.cat(pos_list, dim=1)

    @autocast()
    def forward(self, x, mutable_len=True):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        # import pdb
        # pdb.set_trace()
        output_dict = {}

        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        total_dim = x.shape[1]
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        pos_embed = self.v.pos_embed if not mutable_len else self.get_mutable_pos_embedding(total_dim)

        x = x + pos_embed
        x = self.v.pos_drop(x)
        for k, blk in enumerate(self.v.blocks):
            x = blk(x)
            output_dict['layer{}_out'.format(k + 1)] = x.transpose(1, 2).float()  #N,P,C->N,C,P
        x = self.v.norm(x)
        frame_embeds = x

        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)

        output_dict['globals'] = x.float()
        output_dict['frame'] = frame_embeds.transpose(1, 2).float()  #N,P,C->N,C,P
        output_dict['t_dim'] = self.t_dim
        output_dict['f_dim'] = self.f_dim
        return output_dict
        #return {"global": x.float(), "frame": frame_embeds.transpose(1, 2).float()}


class AugmentMelSTFT(nn.Module):

    def __init__(self,
                 n_mels=128,
                 sr=32000,
                 win_length=800,
                 hopsize=320,
                 n_fft=1024,
                 freqm=48,
                 timem=192,
                 htk=False,
                 fmin=0.0,
                 fmax=None,
                 norm=1,
                 fmin_aug_range=1,
                 fmax_aug_range=1000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer('window', torch.hann_window(win_length, periodic=False), persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):

        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x,
                       self.n_fft,
                       hop_length=self.hopsize,
                       win_length=self.win_length,
                       center=True,
                       normalized=False,
                       window=self.window,
                       return_complex=False)
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1, )).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1, )).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,
                                                                 self.n_fft,
                                                                 self.sr,
                                                                 fmin,
                                                                 fmax,
                                                                 vtln_low=100.0,
                                                                 vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:  #only in eval mode not apply specaugment, which means train mode teacher model also apply specaug
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec

    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length, self.hopsize)


class AST_SED(nn.Module):

    def __init__(self,
                 decode_ratio=10,
                 ast_feature_layer=10,
                 f_pool='frequency_wise_tranformer_encoder',
                 context_block='local_gru_decoder',
                 load_pretrained_model=True,
                 class_num=10):

        super(AST_SED, self).__init__()
        self.class_num = class_num

        # block = getattr(BLOCK, block_name)
        if load_pretrained_model:
            self.patch_transformer = ASTModel(label_dim=527,
                                              fstride=10,
                                              tstride=10,
                                              input_fdim=128,
                                              input_tdim=1024,
                                              imagenet_pretrain=True,
                                              audioset_pretrain=True,
                                              model_size='base384')

        else:

            self.patch_transformer = ASTModel(label_dim=527,
                                              fstride=10,
                                              tstride=10,
                                              input_fdim=128,
                                              input_tdim=1024,
                                              imagenet_pretrain=False,
                                              audioset_pretrain=False,
                                              model_size='base384')

        self.f_pool = f_pool
        self.ast_feature_layer = ast_feature_layer
        self.decode_ratio = decode_ratio
        self.context_block = context_block

        if self.ast_feature_layer == 'frame':
            self.out_norm = nn.Identity()
        else:
            self.out_norm = nn.LayerNorm(768)

        if f_pool == 'mean_pool':
            out_dim = 768
        elif f_pool == 'cat':
            ###to much, error
            self.shrinkblock = nn.Linear(768, 128)
            out_dim = 128 * (self.patch_transformer.f_dim)
        elif f_pool == 'frequency_wise_tranformer_encoder':
            self.linear_emb = nn.Linear(1, 768)
            self.frequency_transformer = nn.ModuleList(
                Block(dim=768, num_heads=4, mlp_ratio=4, norm_layer=nn.LayerNorm) for i in range(2))
            self.frequency_transformer_norm = nn.LayerNorm(768)
            out_dim = 768
        else:
            raise ValueError('invalid f_dim deal type')
        self.out_dim = out_dim

        if context_block in ['gru', 'local_gru_decoder']:
            self.GRU = nn.GRU(out_dim, out_dim, bidirectional=True, batch_first=True, num_layers=2, dropout=0)

            self.classifier = nn.Linear(2 * out_dim, class_num)

        elif context_block == 'transformer':

            self.transformer_decoder = nn.ModuleList(
                Block(dim=768, num_heads=4, mlp_ratio=4, norm_layer=nn.LayerNorm) for i in range(2))
            self.decoder_norm = nn.LayerNorm(768)
            self.classifier = nn.Linear(768, class_num)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            grid_size = self.patch_transformer.t_dim - 1  #remove 1 dim
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, grid_size, 768),
                                                  requires_grad=False)  # fixed sin-cos embedding
            pos_grid = np.arange(grid_size, dtype=np.float32)

            decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], pos=pos_grid)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        elif context_block == 'no':
            self.classifier = nn.Linear(out_dim, class_num)

        else:
            raise ValueError('invalid context block')

    def forward(self, input, mutable_len=True):

        other_dict = {}
        fbank = input
        input = input.transpose(1, 2)
        #input shape B,F,T i.e. 16x128x1024

        #patch-wise context modeling with AST
        ast_out_dict = self.patch_transformer(input, mutable_len)
        if self.ast_feature_layer == 'frame':
            ast_feature = ast_out_dict["frame"]
        else:
            ast_feature = ast_out_dict["layer{}_out".format(self.ast_feature_layer)]

        ast_feature = ast_feature.transpose(1, 2)  #N,C,P->N,P,C
        ast_feature = self.out_norm(ast_feature)

        #frequency-wise pooling
        if self.f_pool == 'mean_pool':
            B_, P_, C_ = ast_feature.shape
            ast_feature = ast_feature[:, 2:, :].reshape(B_, ast_out_dict['f_dim'], ast_out_dict['t_dim'], C_)
            frameout = torch.mean(ast_feature, dim=1)

        elif self.f_pool == 'cat':
            ast_feature = self.shrinkblock(ast_feature)
            B_, P_, C_ = ast_feature.shape
            ast_feature = ast_feature[:, 2:, :].reshape(B_, ast_out_dict['f_dim'], ast_out_dict['t_dim'], C_)
            frameout = ast_feature.transpose(1, 2).reshape(B_, ast_out_dict['t_dim'], -1)

        elif self.f_pool == 'frequency_wise_tranformer_encoder':
            B_, P_, C_ = ast_feature.shape
            ast_feature = ast_feature[:, 2:, :].reshape(B_, ast_out_dict['f_dim'], ast_out_dict['t_dim'], C_)
            ast_feature = ast_feature.transpose(1, 2).reshape(B_ * ast_out_dict['t_dim'], ast_out_dict['f_dim'], C_)
            tag_token = self.linear_emb(torch.ones(ast_feature.size(0), 1, 1).cuda())
            ast_feature = torch.cat([tag_token, ast_feature], dim=1)
            for block in self.frequency_transformer:
                ast_feature = block(ast_feature)
            ast_feature = self.frequency_transformer_norm(ast_feature)

            frameout = ast_feature[:, 0, :]
            frameout = frameout.reshape(B_, ast_out_dict['t_dim'], C_)

        else:
            raise ValueError('error f_dim pooling')

        # frameout = frameout[:,:100,:]#cut t_dim from 101 to 100
        embed_before_gru = frameout
        other_dict["embed_before_gru"] = embed_before_gru

        #context modeling or decoding
        if self.context_block == 'gru':
            self.GRU.flatten_parameters()
            x, hidden = self.GRU(frameout)
        elif self.context_block == 'local_gru_decoder':
            assert self.decode_ratio != 1
            frameout = frameout.transpose(1, 2)  # B,T,C->B,C,T
            frameout = F.interpolate(frameout, scale_factor=self.decode_ratio)
            frameout = frameout.transpose(1, 2)  # B,C,T->B,T,C
            self.GRU.flatten_parameters()
            x, hidden = self.GRU(frameout)
        elif self.context_block == 'transformer':
            x = frameout + self.decoder_pos_embed
            #x = frameout
            for block in self.transformer_decoder:
                x = block(x)
            x = self.decoder_norm(x)
        elif self.context_block == 'no':
            x = frameout
        #localization
        embed_before_classifier = x
        other_dict["sed_embed"] = embed_before_classifier
        x = self.classifier(x)
        other_dict["logit"] = x
        other_dict['fbank'] = fbank
        sed_out = torch.sigmoid(x)
        out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(out, 1e-7, 1.)
        return sed_out.transpose(1,2), at_out, other_dict


class ASTWithSlide(AST_SED):

    def __init__(self,
                 decode_ratio=10,
                 ast_feature_layer=10,
                 f_pool='frequency_wise_tranformer_encoder',
                 context_block='local_gru_decoder',
                 load_pretrained_model=True,
                 class_num=10):
        super().__init__(decode_ratio, ast_feature_layer, f_pool, context_block, load_pretrained_model, class_num)
        self.scale = 101/1024      

    def forward(self, input, win_width=256, step=16, mutable_len=True):
        device = input.device
        batch_size, _, input_len = input.shape
        
        # decode output shape :[batch, frame, shape]
        # For 10 seconds audio, frame length is 101
        patch_num_t = round(input_len*self.scale)
        embedding = torch.zeros([batch_size, patch_num_t, self.out_dim]).to(device)
        accumlator = torch.zeros([batch_size, patch_num_t, self.out_dim]).to(device)
        for w_left in range(0, input_len + step - win_width, step):
            w_right = min(w_left + win_width, input_len)
            out_left = round(w_left * self.scale)
            strong = self.encode(input[:, :, w_left:w_right])
            out_right = int(min(patch_num_t, out_left + strong.shape[1]))
            embedding[:, out_left:out_right, :] += strong
            accumlator[:,out_left:out_right, :] += 1

        embedding /= accumlator

        # decode
        rnn_out = self.rnn_decode(embedding[:,:100,:])
        sed_out = self.localization(rnn_out)
        # pooling
        weak_out = (sed_out * sed_out).sum(dim=2) / sed_out.sum(dim=2)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)
        return sed_out, weak_out

    def encode(self, input, mutable_len=True):
        input = input.transpose(1, 2)
        #input shape B,F,T i.e. 16x128x1024

        #patch-wise context modeling with AST
        ast_out_dict = self.patch_transformer(input, mutable_len)
        if self.ast_feature_layer == 'frame':
            ast_feature = ast_out_dict["frame"]
        else:
            ast_feature = ast_out_dict["layer{}_out".format(self.ast_feature_layer)]

        ast_feature = ast_feature.transpose(1, 2)  #N,C,P->N,P,C
        ast_feature = self.out_norm(ast_feature)

        #frequency-wise pooling
        if self.f_pool == 'mean_pool':
            B_, P_, C_ = ast_feature.shape
            ast_feature = ast_feature[:, 2:, :].reshape(B_, ast_out_dict['f_dim'], ast_out_dict['t_dim'], C_)
            frameout = torch.mean(ast_feature, dim=1)

        elif self.f_pool == 'cat':
            ast_feature = self.shrinkblock(ast_feature)
            B_, P_, C_ = ast_feature.shape
            ast_feature = ast_feature[:, 2:, :].reshape(B_, ast_out_dict['f_dim'], ast_out_dict['t_dim'], C_)
            frameout = ast_feature.transpose(1, 2).reshape(B_, ast_out_dict['t_dim'], -1)

        elif self.f_pool == 'frequency_wise_tranformer_encoder':
            B_, P_, C_ = ast_feature.shape
            ast_feature = ast_feature[:, 2:, :].reshape(B_, ast_out_dict['f_dim'], ast_out_dict['t_dim'], C_)
            ast_feature = ast_feature.transpose(1, 2).reshape(B_ * ast_out_dict['t_dim'], ast_out_dict['f_dim'], C_)
            tag_token = self.linear_emb(torch.ones(ast_feature.size(0), 1, 1).cuda())
            ast_feature = torch.cat([tag_token, ast_feature], dim=1)
            for block in self.frequency_transformer:
                ast_feature = block(ast_feature)
            ast_feature = self.frequency_transformer_norm(ast_feature)

            frameout = ast_feature[:, 0, :]
            frameout = frameout.reshape(B_, ast_out_dict['t_dim'], C_)

        else:
            raise ValueError('error f_dim pooling')

        return frameout

    def rnn_decode(self, frameout: torch.Tensor):
        assert self.decode_ratio != 1
        frameout = frameout.transpose(1, 2)  # B,T,C->B,C,T
        frameout = F.interpolate(frameout, scale_factor=self.decode_ratio)
        frameout = frameout.transpose(1, 2)  # B,C,T->B,T,C
        self.GRU.flatten_parameters()
        output, hidden = self.GRU(frameout)
        return output

    def localization(self, input: torch.Tensor):
        input = self.classifier(input)
        sed_out = torch.sigmoid(input)
        return sed_out.transpose(1, 2)


class AST_AT(nn.Module):

    def __init__(self, ast_feature_layer=12, load_pretrained_model=True):

        super(AST_AT, self).__init__()
        # block = getattr(BLOCK, block_name)
        if load_pretrained_model:
            self.patch_transformer = ASTModel(label_dim=527,
                                              fstride=10,
                                              tstride=10,
                                              input_fdim=128,
                                              input_tdim=1024,
                                              imagenet_pretrain=True,
                                              audioset_pretrain=True,
                                              model_size='base384')

        else:

            self.patch_transformer = ASTModel(label_dim=527,
                                              fstride=10,
                                              tstride=10,
                                              input_fdim=128,
                                              input_tdim=1024,
                                              imagenet_pretrain=False,
                                              audioset_pretrain=False,
                                              model_size='base384')

        self.ast_feature_layer = ast_feature_layer
        self.out_norm = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, 10)

    def forward(self, input):

        other_dict = {}
        fbank = input
        input = input.transpose(1, 2)
        #input shape B,F,T i.e. 16x128x1024

        ast_out_dict = self.patch_transformer(input)
        if self.ast_outlayer == "frame_mean":
            ast_feature = torch.mean(ast_out_dict["frame"].transpose(1, 2)[:, 2:, :], dim=1)
        elif self.ast_outlayer == 'token_mean':
            ast_feature = torch.mean(ast_out_dict["frame"].transpose(1, 2)[:, :2, :], dim=1)
        else:
            ast_feature = ast_out_dict["layer{}_out".format(self.ast_feature_layer)]
            ast_feature = ast_feature.transpose(1, 2)  # N,C,P->N,P,C
            ast_feature = self.out_norm(ast_feature)
            ast_feature = torch.mean(ast_feature[:, 2:, :], dim=1)
        x = ast_feature
        x = self.classifier(x)
        other_dict["logit"] = x
        other_dict['fbank'] = fbank
        at_out = torch.sigmoid(x)
        sed_out = at_out.unsqueeze(1).repeat(1, 1000, 1)
        return sed_out.transpose(1, 2), at_out, other_dict


class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()
        pass

    def forward(self, input):
        #input shape, B,T,C
        return torch.mean(input, dim=1)  #return B,C


class th_learner_block(nn.Module):

    def __init__(self, input_logit_dim=10, input_embedding_dim=1536, output_dim=10, middle_dim=128):
        super(th_learner_block, self).__init__()
        self.logit_projector = nn.Sequential(nn.Linear(input_logit_dim, middle_dim), nn.ReLU())
        self.embedding_projector = nn.Sequential(nn.Linear(input_embedding_dim, middle_dim), nn.ReLU())
        self.th_learner = nn.Sequential(nn.Linear(middle_dim, middle_dim), nn.ReLU(), nn.Linear(middle_dim, middle_dim),
                                        nn.ReLU(), nn.Linear(middle_dim, output_dim))
        self.th_sigmoid = nn.Sigmoid()

    def forward(self, logit, embedding):
        #input dimension (B,C) or (B,T,C)
        logit = logit.detach()
        embed = embedding.detach()
        x1_at = self.logit_projector(logit)
        x2_at = self.embedding_projector(embed)

        th_learner_input = x1_at + x2_at
        th_learner_embedding = self.th_learner(th_learner_input)
        th_learner_output = self.th_sigmoid(th_learner_embedding)

        #input shape, B,T,C
        return th_learner_output


if __name__ == '__main__':
    # input_tdim = 100
    # ast_mdl = ASTModel(input_tdim=input_tdim)
    # # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    # test_input = torch.rand([10, input_tdim, 128])
    # test_output = ast_mdl(test_input)
    # # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    # print(test_output.shape)
    #
    # input_tdim = 256
    # ast_mdl = ASTModel(input_tdim=input_tdim, label_dim=50, audioset_pretrain=True)
    # # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    # test_input = torch.rand([10, input_tdim, 128])
    # test_output = ast_mdl(test_input)
    # # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    # print(test_output.shape)

    #test_input = torch.rand([10, 1, 320000])
    #passe_sed=PaSST_SED()
    #output=passe_sed(test_input)
    pass