from functools import reduce

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from src.preprocess.augmentMelSTFT import AugmentMelSTFT
from src.models.passt.passt import PaSST
from src.models.passt.passt_win import PasstWithSlide
from src.models.passt.tool_block import GAP
from src.models.transformer_decoder import TransformerDecoder, TransformerXLDecoder, ConformerDecoder
from src.models.transformer.mask import MlmModule


class InterpolateModule(nn.Module):
    """ Used for Interpolate
    The module is defined as a module becasue we need to register the hook function
    here to visualize the features.
    """

    def __init__(self, mode="linear") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, seq: torch.Tensor, ratio):
        """
        Args:
            seq (B, T, C): sequence to be interpolated
            ratio: interpolate ratio
        """
        seq = seq.transpose(1, 2)  # B,T,C->B,C,T
        seq = torch.nn.functional.interpolate(seq, scale_factor=ratio, mode=self.mode)
        seq = seq.transpose(1, 2)  # B,C,T->B,T,C
        return seq


class PaSST_SED(nn.Module):

    def __init__(self,
                 decode_ratio=10,
                 interpolate_mode='linear',
                 passt_feature_layer=10,
                 embed_dim=768,
                 f_pool='mean_pool',
                 s_patchout_f=0,
                 s_patchout_t=0,
                 decoder='gru',
                 decoder_layer_num=2,
                 load_pretrained_model=True,
                 specaug='TF',
                 class_num=10,
                 at_adapter=False,
                 passt_at_feature_layer="frame_mean",
                 decoder_win_len=None,
                 mlm=False,
                 mlm_dict=dict()):

        super(PaSST_SED, self).__init__()
        # block = getattr(BLOCK, block_name)
        #setting mel feature extrator and specaug
        if specaug == 'TF':
            freqm = 48
            timem = 192
        elif specaug == 'F':
            freqm = 48
            timem = 0
        elif specaug == 'T':
            freqm = 0
            timem = 192
        else:
            freqm = 0
            timem = 0
        self.mel = AugmentMelSTFT(n_mels=128,
                                  sr=32000,
                                  win_length=800,
                                  hopsize=320,
                                  n_fft=1024,
                                  freqm=freqm,
                                  timem=timem,
                                  htk=False,
                                  fmin=0.0,
                                  fmax=None,
                                  norm=1,
                                  fmin_aug_range=10,
                                  fmax_aug_range=2000)

        #passt
        assert s_patchout_t == 0, "SED task do not support temporal patchout"

        if load_pretrained_model:

            self.patch_transformer = PaSST(u_patchout=0,
                                           s_patchout_t=s_patchout_t,
                                           s_patchout_f=s_patchout_f,
                                           img_size=(128, 998),
                                           patch_size=16,
                                           stride=10,
                                           in_chans=1,
                                           num_classes=527,
                                           embed_dim=embed_dim,
                                           depth=12,
                                           num_heads=12,
                                           mlp_ratio=4,
                                           qkv_bias=True,
                                           representation_size=None,
                                           distilled=True,
                                           drop_rate=0,
                                           attn_drop_rate=0,
                                           drop_path_rate=0.,
                                           norm_layer=None,
                                           act_layer=None,
                                           weight_init='')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            sd = torch.load('./pretrained_model/passt-s-f128-p16-s10-ap.476-swa.pt', map_location=device)
            self.patch_transformer.load_state_dict(sd, strict=False)
            pass

        else:
            self.patch_transformer = PaSST(
                u_patchout=0,
                s_patchout_t=0,
                s_patchout_f=0,
                img_size=(128, 998),
                patch_size=16,
                stride=10,
                in_chans=1,
                num_classes=527,
                embed_dim=embed_dim,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                representation_size=None,
                distilled=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0.,
                norm_layer=None,
                act_layer=None,
                weight_init='',
            )

        self.s_patchout_f = s_patchout_f

        self.f_pool_name = f_pool
        self.passt_feature_layer = passt_feature_layer
        self.decoder_name = decoder
        self.decode_ratio = decode_ratio
        self.class_num = class_num

        if self.passt_feature_layer == 'frame':
            self.out_norm = nn.Identity()
        else:
            self.out_norm = nn.LayerNorm(embed_dim)

        self.init_f_pool(pool_name=f_pool, embed_dim=embed_dim)
        self.interpolate_module = InterpolateModule(mode=interpolate_mode)
        self.slide_window_layer = nn.Identity() # Just for capture the output of the sliding windows outside the class
        self.mlm = mlm
        if mlm:
            self.init_mlm(device=device, out_dim=self.out_dim, mlm_dict=mlm_dict)

        self.init_decoder(self.out_dim, class_num, decoder_win_len, decoder_layer_num)

        self.at_adpater = at_adapter
        if self.at_adpater:
            self.passt_at_feature_layer = passt_at_feature_layer
            if self.passt_at_feature_layer in ['frame_mean', 'token_mean']:
                self.at_adpater = nn.Sequential(GAP(), nn.Linear(768, class_num))

            else:
                self.at_adpater = nn.Sequential(nn.LayerNorm(768), GAP(), nn.Linear(768, class_num))

    def init_f_pool(self, pool_name, embed_dim):
        if self.passt_feature_layer == "mlp_mix":
            self.linear_f1 = nn.Linear(embed_dim, embed_dim)
            self.linear_f2 = nn.Linear(embed_dim, embed_dim)
            self.linear_f3 = nn.Linear(embed_dim, embed_dim)
            self.fuse = nn.Linear(embed_dim * 3, embed_dim)

        if pool_name == 'mean_pool':
            self.out_dim = embed_dim

        elif pool_name == 'cat':
            self.shrinkblock = nn.Linear(embed_dim, 128)  #first reduce dimension, then concat
            self.out_dim = 128 * (self.patch_transformer.f_dim)

        elif pool_name == 'frequency_wise_tranformer_encoder':
            self.linear_emb = nn.Linear(1, embed_dim)
            self.frequency_transformer = nn.ModuleList(
                Block(dim=768, num_heads=4, mlp_ratio=4, norm_layer=nn.LayerNorm) for i in range(2))
            self.frequency_transformer_norm = nn.LayerNorm(768)
            self.out_dim = 768
        else:
            raise ValueError('invalid f_dim deal type')

    def init_decoder(self, out_dim, class_num, win_len, decoder_layer_num):
        self.decoder_layer_num = decoder_layer_num
        if self.decoder_name == 'gru':
            self.GRU = nn.GRU(out_dim,
                              out_dim,
                              bidirectional=True,
                              batch_first=True,
                              num_layers=decoder_layer_num,
                              dropout=0)
            self.classifier = nn.Linear(2 * out_dim, class_num)

        elif self.decoder_name == "PUP":
            if decoder_layer_num == 2:
                self.GRU1 = nn.GRU(out_dim, out_dim, bidirectional=True, batch_first=True, num_layers=1, dropout=0)
                self.GRU2 = nn.GRU(2 * out_dim, out_dim, bidirectional=True, batch_first=True, num_layers=1, dropout=0)
                self.decode_rate_list = [3, 3]
                assert reduce(lambda x, y: x * y,
                              self.decode_rate_list) == self.decode_ratio, "Decoder rate inconsistent!"
                self.classifier = nn.Linear(2 * out_dim, class_num)
            elif decoder_layer_num == 3:
                self.GRU1 = nn.GRU(out_dim, out_dim, bidirectional=True, batch_first=True, num_layers=1, dropout=0)
                self.GRU2 = nn.GRU(2 * out_dim, out_dim, bidirectional=True, batch_first=True, num_layers=1, dropout=0)
                self.GRU3 = nn.GRU(2 * out_dim, out_dim, bidirectional=True, batch_first=True, num_layers=1, dropout=0)
                self.decode_rate_list = [2.5, 2, 2]
                assert reduce(lambda x, y: x * y,
                              self.decode_rate_list) == self.decode_ratio, "Decoder rate inconsistent!"
                self.classifier = nn.Linear(2 * out_dim, class_num)
            else:
                raise ValueError("Argument decoder_layer_num must be 2 or 3!")

        elif self.decoder_name == "transformer":
            self.decoder = TransformerDecoder(input_dim=out_dim, decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(out_dim, class_num)
        elif self.decoder_name == "transformerXL":
            self.decoder = TransformerXLDecoder(input_dim=out_dim,
                                                seq_len=1000,
                                                window_len=win_len,
                                                decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(out_dim, class_num)
        elif self.decoder_name == "conformer":
            self.decoder = ConformerDecoder(input_dim=out_dim,
                                            seq_len=1000,
                                            window_len=win_len,
                                            decoder_layer_num=decoder_layer_num)
            self.classifier = nn.Linear(out_dim, class_num)
        elif self.decoder_name == 'no':
            self.classifier = nn.Linear(out_dim, class_num)
        else:
            raise ValueError('invalid decoder block')

    def init_mlm(self, device, out_dim, mlm_dict=dict()):
        self.mlm_tool = MlmModule(device=device, **mlm_dict)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.mlm_mlp = nn.Sequential(torch.nn.Linear(out_dim, out_dim), torch.nn.GELU(), torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(out_dim, out_dim))

    def f_pool(self, passt_out_dict):
        # shape of passt_feature is (N,C,P)
        if self.passt_feature_layer == 'frame':
            passt_feature = passt_out_dict['frame'][:, :, 2:]
            passt_feature = passt_feature.transpose(1, 2)  # N,C,P->N,P,C
        elif self.passt_feature_layer == "mlp_mix":
            feature_list = [passt_out_dict["layer{}_out".format(i)][:, :, 2:].transpose(1, 2)
                            for i in (8, 10, 12)]  # N,C,P->N,P,C
            passt_feature = self.fuse(torch.cat(feature_list, dim=-1))
            assert passt_feature.shape[2] == self.out_dim
        else:
            passt_feature = passt_out_dict["layer{}_out".format(self.passt_feature_layer)][:, :, 2:]
            passt_feature = passt_feature.transpose(1, 2)  # N,C,P->N,P,C
        passt_feature = self.out_norm(passt_feature)

        if self.f_pool_name == 'mean_pool':
            B_, P_, C_ = passt_feature.shape
            passt_feature = passt_feature.reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
            frameout = torch.mean(passt_feature, dim=1)
        elif self.f_pool_name == 'cat':
            passt_feature = self.shrinkblock(passt_feature)
            B_, P_, C_ = passt_feature.shape
            passt_feature = passt_feature.reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
            frameout = passt_feature.transpose(1, 2).reshape(B_, passt_out_dict['t_dim'], -1)

        elif self.f_pool_name == 'frequency_wise_tranformer_encoder':
            B_, P_, C_ = passt_feature.shape
            passt_feature = passt_feature.reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
            passt_feature = passt_feature.transpose(1, 2).reshape(B_ * passt_out_dict['t_dim'], passt_out_dict['f_dim'],
                                                                  C_)
            tag_token = self.linear_emb(torch.ones(passt_feature.size(0), 1, 1).cuda())
            passt_feature = torch.cat([tag_token, passt_feature], dim=1)
            for block in self.frequency_transformer:
                passt_feature = block(passt_feature)
            passt_feature = self.frequency_transformer_norm(passt_feature)
            frameout = passt_feature[:, 0, :]
            frameout = frameout.reshape(B_, passt_out_dict['t_dim'], C_)
        else:
            raise ValueError('error f_dim pooling')
        return frameout

    def decoder_step(self, x, other_dict, encoder_win: bool):
        # subsampling
        if not encoder_win:
            x = torch.cat((x, x[:, -1, :].unsqueeze(1)), dim=1)  #padding from 99 to 100
            if not encoder_win or self.decoder_name not in ['no', 'PUP']:
                x = self.interpolate_module(x, self.decode_ratio)

        # mask language model
        if self.mlm:
            other_dict["frame_before_mask"] = x
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        # time series model
        if self.decoder_name == 'gru':
            self.GRU.flatten_parameters()
            x, _ = self.GRU(x)


        elif self.decoder_name.find("transformer") >= 0:
            x = self.decoder(x)
        elif self.decoder_name == 'no':
            x = x
        return x

    def forward(self, input: torch.Tensor, encoder_win=False, mix_rate=0.5, win_param=[512, 49], temp_w=1):
        # Bx1xL  i.e. 10x1x320000
        other_dict = {}
        # fbank = input.squeeze(1).transpose(1, 2)
        org_input = input
        input = input.unsqueeze(1)
        # input shape B,F,T i.e. 10x1000x128

        #patch-wise context modeling
        passt_out_dict = self.patch_transformer(input)
        # pooling
        if not encoder_win:
            x = self.f_pool(passt_out_dict)
        else:
            # merge the global feature map and the local feature map
            slide_window_model = PasstWithSlide(net=self, win_param=win_param)
            x_local = self.slide_window_layer(slide_window_model(org_input))

            x_global = self.f_pool(passt_out_dict)
            x_global = torch.cat((x_global, x_global[:, -1, :].unsqueeze(1)), dim=1)  #padding from 99 to 100
            x_global = self.interpolate_module(x_global, self.decode_ratio)
            x = mix_rate * x_local + (1 - mix_rate) * x_global
        # decoder
        x = self.decoder_step(x, other_dict, encoder_win=encoder_win)

        if self.mlm:
            x = self.mlm_mlp(x)
            return x, other_dict
        embed_before_classifier = x

        other_dict["sed_embed"] = embed_before_classifier
        # localization
        x = self.classifier(x)
        other_dict["logit"] = x

        # other_dict['fbank'] = fbank
        sed_out = torch.sigmoid(x/temp_w)

        # linear-softmax pool
        out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        at_out = torch.clamp(out, 1e-7, 1.)
        other_dict['sed_logit'] = other_dict["logit"].transpose(1, 2)
        other_dict['at_logit'] = torch.logit(at_out)

        #################################################################################
        #                            Audio tagging branch
        #################################################################################
        if self.at_adpater:
            if self.passt_at_feature_layer == "frame_mean":
                at_embedding = passt_out_dict["frame"].transpose(1, 2)[:, 2:, :]  #B,P,C
            elif self.passt_at_feature_layer == 'token_mean':
                at_embedding = passt_out_dict["frame"].transpose(1, 2)[:, :2, :],
            else:
                at_embedding = passt_out_dict["layer{}_out".format(self.passt_at_feature_layer)].transpose(1, 2)[:,
                                                                                                                 2:, :]

            at_adapter_logit = self.at_adpater(at_embedding)
            #print(at_adapter_embedding.shape)
            at_adapter_out = torch.sigmoid(at_adapter_logit)
            other_dict['at_out_specific'] = at_adapter_out

        return sed_out.transpose(1, 2), at_out, other_dict
